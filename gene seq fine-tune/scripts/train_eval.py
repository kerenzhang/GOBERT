from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene_seq_finetune.data import load_bundle
from gene_seq_finetune.metrics import summarize_predictions
from gene_seq_finetune.models import ExpressionDrugFusion, GoBERTExpressionFusion


TARGET_COLUMNS = ["AUC_PUBLISHED", "log2_IC50_PUBLISHED"]
MODEL_NAMES = ["gobert_expression_fusion", "expression_drug_fusion", "drug_mean"]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OMICS_DIR = Path("/content/drive/MyDrive/OMICS/gene seq")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_expression_features(bundle, fit_rows: np.ndarray, n_pcs: int):
    fit_model_ids = sorted(bundle.responses.iloc[fit_rows]["ModelID"].unique())
    fit_line_rows = np.asarray([bundle.model_to_row[value] for value in fit_model_ids], dtype=int)
    scaler = StandardScaler().fit(bundle.expression_by_model[fit_line_rows])
    standardized = scaler.transform(bundle.expression_by_model)
    n_components = min(n_pcs, max(1, len(fit_line_rows) - 1), standardized.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0).fit(standardized[fit_line_rows])
    pcs = pca.transform(standardized)
    pc_scaler = StandardScaler().fit(pcs[fit_line_rows])
    pcs = pc_scaler.transform(pcs).astype(np.float32)
    preprocessor = {"gene_scaler": scaler, "pca": pca, "pc_scaler": pc_scaler}
    return standardized.astype(np.float32), pcs, n_components, preprocessor


def index_arrays(bundle, rows: pd.DataFrame):
    return (
        np.asarray([bundle.model_to_row[value] for value in rows["ModelID"]], dtype=np.int64),
        np.asarray([bundle.drug_to_idx[value] for value in rows["DRUG_ID"]], dtype=np.int64),
        np.asarray([bundle.dataset_to_idx[value] for value in rows["DATASET"]], dtype=np.int64),
    )


def train_neural_model(kind, bundle, train_rows, validation_rows, test_rows,
                       standardized_expression, expression_pcs, n_components,
                       args, device, fold, checkpoint_dir):
    train_df = bundle.responses.iloc[train_rows]
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    target_mean = y_train.mean(axis=0)
    target_scale = y_train.std(axis=0)
    target_scale[target_scale < 1e-8] = 1.0

    if kind == "gobert_expression_fusion":
        model = GoBERTExpressionFusion(
            bundle.gene_embeddings, len(bundle.drug_to_idx), len(bundle.dataset_to_idx),
            projection_dim=args.projection_dim, drug_dim=args.drug_dim, dropout=args.dropout,
        ).to(device)
        feature_matrix = standardized_expression
    else:
        model = ExpressionDrugFusion(
            n_components, len(bundle.drug_to_idx), len(bundle.dataset_to_idx),
            hidden_dim=args.projection_dim, drug_dim=args.drug_dim, dropout=args.dropout,
        ).to(device)
        feature_matrix = expression_pcs

    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    train_idx = index_arrays(bundle, train_df)
    val_df = bundle.responses.iloc[validation_rows]
    val_idx = index_arrays(bundle, val_df)
    test_df = bundle.responses.iloc[test_rows]
    test_idx = index_arrays(bundle, test_df)
    y_scaled = (y_train - target_mean) / target_scale
    train_ds = TensorDataset(torch.from_numpy(train_idx[0]), torch.from_numpy(train_idx[1]),
                             torch.from_numpy(train_idx[2]), torch.from_numpy(y_scaled.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_tensors = tuple(torch.as_tensor(x, dtype=torch.long, device=device) for x in val_idx)
    val_targets = torch.as_tensor(
        (val_df[TARGET_COLUMNS].to_numpy(dtype=np.float32) - target_mean) / target_scale,
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_line, batch_drug, batch_dataset, batch_y in train_loader:
            batch_line, batch_drug = batch_line.to(device), batch_drug.to(device)
            batch_dataset, batch_y = batch_dataset.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature_tensor[batch_line], batch_drug, batch_dataset)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_prediction = model(feature_tensor[val_tensors[0]], val_tensors[1], val_tensors[2])
            val_loss = float(loss_fn(val_prediction, val_targets).item())
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, wait = epoch, 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No best checkpoint was recorded; check training/validation data")

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.inference_mode():
        test_line = torch.as_tensor(test_idx[0], dtype=torch.long, device=device)
        test_drug = torch.as_tensor(test_idx[1], dtype=torch.long, device=device)
        test_dataset = torch.as_tensor(test_idx[2], dtype=torch.long, device=device)
        test_prediction = model(feature_tensor[test_line], test_drug, test_dataset)
    prediction = test_prediction.cpu().numpy() * target_scale + target_mean

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_state = {key: value.detach().cpu() for key, value in best_state.items() if key != "gene_embeddings"}
    torch.save({
        "model_type": kind, "fold": fold, "model_state_dict": saved_state,
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        "gene_ids": bundle.gene_ids, "drug_to_idx": bundle.drug_to_idx,
        "dataset_to_idx": bundle.dataset_to_idx, "gene_embedding_file": str(args.gene_embeddings.resolve()),
        "best_epoch": best_epoch,
        "model_config": {"projection_dim": args.projection_dim, "drug_dim": args.drug_dim,
                         "n_components": n_components, "dropout": args.dropout},
    }, checkpoint_dir / f"{kind}_fold_{fold:02d}.pt")
    return prediction, best_epoch, best_loss


def drug_mean_predictions(bundle, train_rows, test_rows):
    train = bundle.responses.iloc[train_rows]
    test = bundle.responses.iloc[test_rows]
    global_mean = train[TARGET_COLUMNS].mean()
    means = train.groupby("DRUG_ID")[TARGET_COLUMNS].mean()
    return np.vstack([
        means.loc[drug].to_numpy(dtype=float) if drug in means.index else global_mean.to_numpy(dtype=float)
        for drug in test["DRUG_ID"]
    ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train grouped-CV GoBERT/expression drug-response baselines.")
    parser.add_argument("--expression", type=Path, default=DEFAULT_OMICS_DIR / "gene_expression_breast.csv",
                        help="Breast expression CSV (default: mounted Drive OMICS/gene seq folder)")
    parser.add_argument("--responses", type=Path, default=DEFAULT_OMICS_DIR / "drug_response_breast_matched.csv",
                        help="Matched drug-response CSV (default: mounted Drive OMICS/gene seq folder)")
    parser.add_argument("--gene-embeddings", type=Path,
                        default=DEFAULT_OMICS_DIR / "human_gene_gobert_embeddings.npz",
                        help="Packed NPZ made by export_gene_embeddings.py")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "breast_v1",
                        help="Output folder (defaults inside this project directory)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--drug-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    for input_path, label in ((args.expression, "expression matrix"),
                              (args.responses, "drug response table"),
                              (args.gene_embeddings, "GoBERT embedding archive")):
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Could not find {label}: {input_path}\n"
                "In Colab, mount Drive with `from google.colab import drive; drive.mount('/content/drive')`, "
                "then check the filenames under MyDrive/OMICS/gene seq or pass the correct path argument."
            )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    bundle = load_bundle(args.expression, args.responses, args.gene_embeddings)
    response = bundle.responses
    groups = response["ModelID"].to_numpy()
    outer = GroupKFold(n_splits=args.folds)
    model_names = ["gobert_expression_fusion", "expression_drug_fusion", "drug_mean"]
    oof = {name: np.full((len(response), 2), np.nan, dtype=np.float32) for name in model_names}
    fold_ids = np.full(len(response), -1, dtype=np.int32)
    logs = []

    for fold, (outer_train, test_rows) in enumerate(outer.split(np.zeros(len(response)), response["AUC_PUBLISHED"], groups), start=1):
        split = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed + fold)
        train_rel, val_rel = next(split.split(np.zeros(len(outer_train)), groups=groups[outer_train]))
        train_rows, validation_rows = outer_train[train_rel], outer_train[val_rel]
        standardized, pcs, n_components, preprocessor = make_expression_features(bundle, train_rows, args.n_pcs)
        fold_ids[test_rows] = fold
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dump({**preprocessor, "gene_ids": bundle.gene_ids}, checkpoint_dir / f"preprocessor_fold_{fold:02d}.joblib")

        for name in model_names[:2]:
            prediction, best_epoch, best_loss = train_neural_model(
                name, bundle, train_rows, validation_rows, test_rows, standardized, pcs,
                n_components, args, device, fold, checkpoint_dir,
            )
            oof[name][test_rows] = prediction
            logs.append({"fold": fold, "model": name, "n_train_samples": len(train_rows),
                         "n_validation_samples": len(validation_rows), "n_test_samples": len(test_rows),
                         "n_train_cell_lines": response.iloc[train_rows]["ModelID"].nunique(),
                         "n_validation_cell_lines": response.iloc[validation_rows]["ModelID"].nunique(),
                         "n_test_cell_lines": response.iloc[test_rows]["ModelID"].nunique(),
                         "best_epoch": best_epoch, "best_validation_mse_standardized": best_loss})
        oof["drug_mean"][test_rows] = drug_mean_predictions(bundle, train_rows, test_rows)
        print(f"Completed fold {fold}/{args.folds} on {device}; held out {response.iloc[test_rows]['ModelID'].nunique()} cell lines")

    if any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("Some samples did not receive an out-of-fold prediction")
    output = response[["sample_id", "ModelID", "DATASET", "DRUG_ID", "DRUG_NAME", "AUC_PUBLISHED", "IC50_PUBLISHED", "log2_IC50_PUBLISHED"]].copy()
    output = output.rename(columns={"AUC_PUBLISHED": "true_AUC_PUBLISHED", "IC50_PUBLISHED": "true_IC50_PUBLISHED",
                                    "log2_IC50_PUBLISHED": "true_log2_IC50_PUBLISHED"})
    output["fold"] = fold_ids
    for name, values in oof.items():
        output[f"pred_{name}_auc"] = values[:, 0]
        output[f"pred_{name}_log2_ic50"] = values[:, 1]
        output[f"pred_{name}_ic50"] = np.exp2(values[:, 1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    global_metrics, macro_metrics, per_drug_metrics = summarize_predictions(output, model_names)
    global_metrics.to_csv(args.output_dir / "metrics_global.csv", index=False)
    macro_metrics.to_csv(args.output_dir / "metrics_per_drug_macro.csv", index=False)
    per_drug_metrics.to_csv(args.output_dir / "metrics_per_drug.csv", index=False)
    pd.DataFrame(logs).to_csv(args.output_dir / "training_folds.csv", index=False)
    config = vars(args)
    config.update({"device_resolved": str(device), "n_samples": int(len(response)),
                   "n_cell_lines": int(response["ModelID"].nunique()), "n_drugs": int(response["DRUG_ID"].nunique()),
                   "n_aligned_genes": int(len(bundle.gene_ids)), "gobert_embedding_dim": int(bundle.gene_embeddings.shape[1])})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(global_metrics.to_string(index=False))
    print(f"Saved out-of-fold predictions and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
