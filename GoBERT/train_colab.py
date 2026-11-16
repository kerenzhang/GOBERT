"""Single-GPU training of the reference GoBERT, with resumable local checkpoints."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import obonet
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertConfig

from gobert import CustomDataCollatorForLanguageModeling, GoBERTModel_Ex_Im


class GoTermDataset(Dataset):
    """Encode the supplied space-separated GO IDs without splitting GO identifiers."""

    def __init__(self, path, tokenizer, max_length, limit=None):
        self.tokenizer = tokenizer
        self.max_terms = max_length - tokenizer.num_special_tokens_to_add()
        with Path(path).open(encoding="utf-8") as stream:
            self.lines = []
            for line in stream:
                if line.strip():
                    self.lines.append(line.strip())
                    if limit is not None and len(self.lines) >= limit:
                        break
        if not self.lines:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        terms = self.lines[index].split()[:self.max_terms]
        ids = self.tokenizer.convert_tokens_to_ids(terms)
        if self.tokenizer.unk_token_id in ids:
            raise ValueError(f"Unknown GO ID in dataset row {index + 1}")
        ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "token_type_ids": [0] * len(ids)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="data/unique_train_311536.txt")
    parser.add_argument("--validation_file", default="data/unique_valid_32014.txt")
    parser.add_argument("--tokenizer_name", default="hf_gobert")
    parser.add_argument("--semantic_path", default="data/e5-mistral-7b-instruct.npy")
    parser.add_argument("--output_dir", default="runs/colab")
    parser.add_argument("--num_train_epochs", type=int, default=18)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2.1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--ratiomlmloss", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--max_eval_batches", type=int)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--smoke_test", action="store_true", help="Use a tiny model and small subsets to verify the pipeline.")
    args = parser.parse_args()
    for name in ["num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "checkpointing_steps"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.max_seq_length < 3:
        parser.error("--max_seq_length must be at least 3")
    for name in ["max_train_steps", "max_eval_batches"]:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.smoke_test:
        args.num_train_epochs = 1
        args.max_train_steps = args.max_train_steps or 2
        args.max_eval_batches = args.max_eval_batches or 2
        args.gradient_accumulation_steps = 1
        if args.output_dir == "runs/colab":
            args.output_dir = "runs/colab-smoke"
    return args


def evaluate(model, loader, device, amp_dtype, max_batches, seed):
    """Use repeatable validation masks without changing the training RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    loss_sum = correct = masked = batches = 0
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        result = model(**batch)
                    selected = batch["labels"] != -100
                    correct += ((result["prediction_logits"].argmax(-1) == batch["labels"]) & selected).sum().item()
                    masked += selected.sum().item()
                    loss_sum += result["loss"].item()
                    batches += 1
                    if max_batches and batch_index + 1 >= max_batches:
                        break
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return {"loss": loss_sum / batches, "mlm_accuracy": correct / max(masked, 1), "masked_tokens": masked}


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, scaler, epoch, next_batch, step, best_accuracy, args):
    path.mkdir(parents=True, exist_ok=True)
    # The fixed sparse graph is external data; exports contain learned weights.
    model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)
    torch.save({
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "next_batch": next_batch, "step": step, "best_accuracy": best_accuracy,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args": vars(args),
    }, path / "training_state.pt")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("Select a GPU runtime in Colab before full training.")
    if args.mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16.")
    amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision] if device.type == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False, local_files_only=True)
    probe = tokenizer("GO:0005739 GO:0005783")["input_ids"]
    if tokenizer.unk_token_id in probe or len(probe) != 4:
        raise ValueError("GO tokenization failed; install requirements-colab.txt and restart the runtime.")
    if args.semantic_path:
        table = np.load(args.semantic_path, mmap_mode="r", allow_pickle=False)
        if table.ndim != 2 or table.shape[0] != len(tokenizer) - 5:
            raise ValueError("Semantic table row count must equal the GO vocabulary size (excluding five special tokens).")
        del table
    hidden = 64 if args.smoke_test else 1024
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=hidden, num_hidden_layers=1 if args.smoke_test else 10,
        num_attention_heads=4 if args.smoke_test else 16, intermediate_size=128 if args.smoke_test else 3072,
        hidden_act="gelu", hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.1,
        max_position_embeddings=len(tokenizer), type_vocab_size=2, initializer_range=0.02,
        pad_token_id=tokenizer.pad_token_id,
    )
    # A smoke run still exercises the real semantic projection and full GO vocabulary.
    model = GoBERTModel_Ex_Im(config, lam=args.ratiomlmloss, semantic_path=args.semantic_path)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if model.ex_label_buffer.shape != (len(tokenizer), len(tokenizer)):
        raise ValueError("Graph dimensions do not match the tokenizer.")
    graph = obonet.read_obo("go-basic.obo", ignore_obsolete=True)
    roots = [term for term in graph if not list(graph.successors(term))]
    vocab = tokenizer.get_vocab()
    collator = CustomDataCollatorForLanguageModeling(
        tokenizer, mlm_probability=args.mlm_probability, root_token_ids=tokenizer.convert_tokens_to_ids(roots),
        go_graph=graph, max_length=args.max_seq_length,
        token_id_go_id_dict={str(index): term for term, index in vocab.items()},
    )
    train_data = GoTermDataset(args.train_file, tokenizer, args.max_seq_length, 32 if args.smoke_test else None)
    valid_data = GoTermDataset(args.validation_file, tokenizer, args.max_seq_length, 8 if args.smoke_test else None)
    valid_loader = DataLoader(valid_data, batch_size=args.per_device_eval_batch_size, collate_fn=collator)
    batch_count = math.ceil(len(train_data) / args.per_device_train_batch_size)
    updates_per_epoch = math.ceil(batch_count / args.gradient_accumulation_steps)
    total_steps = args.num_train_epochs * updates_per_epoch
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in ["bias", "LayerNorm.weight"])], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0},
    ], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=0.5, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch = start_batch = step = 0
    best_accuracy = -1.0
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        # These settings define optimizer schedule, batch ordering, and architecture.
        for key in ["smoke_test", "seed", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "max_seq_length", "learning_rate", "weight_decay", "ratiomlmloss"]:
            if state["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume requires the original --{key} setting.")
        model.load_state_dict(torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch, step, best_accuracy = (state[key] for key in ["epoch", "next_batch", "step", "best_accuracy"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Device: {device}; train rows: {len(train_data)}; validation rows: {len(valid_data)}; effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print("Graph remains sparse. Reference neighborhood-target and masking behavior are retained.")
    if args.max_train_steps and step >= args.max_train_steps:
        print("Checkpoint has already reached --max_train_steps; remove or increase the limit to continue.")
        return
    for epoch in range(start_epoch, args.num_train_epochs):
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        offset = start_batch if epoch == start_epoch else 0
        loader = DataLoader(Subset(train_data, order[offset * args.per_device_train_batch_size:]), batch_size=args.per_device_train_batch_size, collate_fn=collator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        next_batch = offset
        progress = tqdm(enumerate(loader, start=offset), total=len(loader), desc=f"Epoch {epoch + 1}")
        for batch_index, batch in progress:
            if args.max_train_steps and step >= args.max_train_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, batch_count - group_start)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                result = model(**batch)
                loss = result["loss"] / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, batch {batch_index}")
            scaler.scale(loss).backward()
            running_loss += result["loss"].item()
            next_batch = batch_index + 1
            if next_batch % args.gradient_accumulation_steps == 0 or next_batch == batch_count:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    step += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss=result["loss"].item(), step=step)
                if step > 0 and step % args.checkpointing_steps == 0:
                    save_checkpoint(output / "checkpoint_last", model, tokenizer, optimizer, scheduler, scaler,
                                    epoch, next_batch, step, best_accuracy, args)
                if args.max_train_steps and step >= args.max_train_steps:
                    break
        metrics = evaluate(model, valid_loader, device, amp_dtype, args.max_eval_batches, args.seed + 10000)
        print(json.dumps({"epoch": epoch + 1, "step": step, "validation": metrics}))
        if next_batch >= batch_count:
            resume_epoch, resume_batch = epoch + 1, 0
        else:
            resume_epoch, resume_batch = epoch, next_batch
        improved = metrics["mlm_accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, metrics["mlm_accuracy"])
        save_checkpoint(output / "checkpoint_last", model, tokenizer, optimizer, scheduler, scaler, resume_epoch, resume_batch, step, best_accuracy, args)
        if improved:
            # The best export needs weights and tokenizer; resume from checkpoint_last.
            best_path = output / "checkpoint_best"
            best_path.mkdir(exist_ok=True)
            model.save_pretrained(best_path, safe_serialization=False)
            tokenizer.save_pretrained(best_path)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch + 1, "step": step, **metrics}) + "\n")
        if args.max_train_steps and step >= args.max_train_steps:
            break
    print(f"Saved model and resumable state under {output}")


if __name__ == "__main__":
    main()
