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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene_seq_finetune.data import load_bundle
from gene_seq_finetune.metrics import summarize_predictions, summarize_within_drug
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


def drug_dataset_baselines(reference_rows: pd.DataFrame, query_rows: pd.DataFrame) -> np.ndarray:
    """Training-fold means per drug and dataset, with drug/global fallbacks."""
    group_means = reference_rows.groupby(["DRUG_ID", "DATASET"])[TARGET_COLUMNS].mean()
    drug_means = reference_rows.groupby("DRUG_ID")[TARGET_COLUMNS].mean()
    global_mean = reference_rows[TARGET_COLUMNS].mean().to_numpy(dtype=np.float32)
    result = np.empty((len(query_rows), len(TARGET_COLUMNS)), dtype=np.float32)
    for i, (drug, dataset) in enumerate(zip(query_rows["DRUG_ID"], query_rows["DATASET"])):
        key = (drug, dataset)
        if key in group_means.index:
            value = group_means.loc[key].to_numpy(dtype=np.float32)
        elif drug in drug_means.index:
            value = drug_means.loc[drug].to_numpy(dtype=np.float32)
        else:
            value = global_mean
        result[i] = value
    return result


def within_group_pairwise_rank_loss(prediction, target, drug_idx, dataset_idx, line_idx, tie_epsilon: float):
    """Pairwise logistic ranking loss, comparing only same-drug/dataset samples."""
    batch_size = target.shape[0]
    if batch_size < 2:
        return prediction.sum() * 0.0
    same_group = (
        (drug_idx[:, None] == drug_idx[None, :])
        & (dataset_idx[:, None] == dataset_idx[None, :])
        & (line_idx[:, None] != line_idx[None, :])
    )
    pair_mask = torch.triu(same_group, diagonal=1)
    losses = []
    for target_idx in range(target.shape[1]):
        target_diff = target[:, target_idx, None] - target[None, :, target_idx]
        keep = pair_mask & (target_diff.abs() > tie_epsilon)
        if keep.any():
            pred_diff = prediction[:, target_idx, None] - prediction[None, :, target_idx]
            direction = torch.sign(target_diff[keep])
            losses.append(F.softplus(-direction * pred_diff[keep]).mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def train_neural_model(kind, bundle, train_rows, validation_rows, test_rows,
                       standardized_expression, expression_pcs, n_components,
                       args, device, fold, checkpoint_dir):
    train_df = bundle.responses.iloc[train_rows]
    train_baseline = drug_dataset_baselines(train_df, train_df)
    val_df = bundle.responses.iloc[validation_rows]
    val_baseline = drug_dataset_baselines(train_df, val_df)
    test_df = bundle.responses.iloc[test_rows]
    test_baseline = drug_dataset_baselines(train_df, test_df)
    within_drug_mode = args.prediction_mode == "within_drug"
    y_train_raw = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    y_train = y_train_raw - train_baseline if within_drug_mode else y_train_raw
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
    val_idx = index_arrays(bundle, val_df)
    test_idx = index_arrays(bundle, test_df)
    y_scaled = (y_train - target_mean) / target_scale
    train_ds = TensorDataset(torch.from_numpy(train_idx[0]), torch.from_numpy(train_idx[1]),
                             torch.from_numpy(train_idx[2]), torch.from_numpy(y_scaled.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_tensors = tuple(torch.as_tensor(x, dtype=torch.long, device=device) for x in val_idx)
    val_y = val_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    if within_drug_mode:
        val_y = val_y - val_baseline
    val_targets = torch.as_tensor(
        (val_y - target_mean) / target_scale,
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
            regression_loss = loss_fn(prediction, batch_y)
            if within_drug_mode:
                ranking_loss = within_group_pairwise_rank_loss(
                    prediction, batch_y, batch_drug, batch_dataset, batch_line, args.ranking_tie_epsilon,
                )
            else:
                ranking_loss = prediction.sum() * 0.0
            loss = regression_loss + args.ranking_loss_weight * ranking_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_prediction = model(feature_tensor[val_tensors[0]], val_tensors[1], val_tensors[2])
            val_regression_loss = loss_fn(val_prediction, val_targets)
            if within_drug_mode:
                val_ranking_loss = within_group_pairwise_rank_loss(
                    val_prediction, val_targets, val_tensors[1], val_tensors[2], val_tensors[0],
                    args.ranking_tie_epsilon,
                )
            else:
                val_ranking_loss = val_prediction.sum() * 0.0
            val_loss = float((val_regression_loss + args.ranking_loss_weight * val_ranking_loss).item())
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
    predicted_target = test_prediction.cpu().numpy() * target_scale + target_mean
    prediction = predicted_target + test_baseline if within_drug_mode else predicted_target

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_state = {key: value.detach().cpu() for key, value in best_state.items() if key != "gene_embeddings"}
    group_baseline_records = (
        train_df.groupby(["DRUG_ID", "DATASET"])[TARGET_COLUMNS].mean().reset_index().to_dict(orient="records")
    )
    drug_baseline_records = train_df.groupby("DRUG_ID")[TARGET_COLUMNS].mean().reset_index().to_dict(orient="records")
    torch.save({
        "model_type": kind, "fold": fold, "prediction_mode": args.prediction_mode,
        "model_state_dict": saved_state,
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        "drug_dataset_baselines": group_baseline_records,
        "drug_baseline_fallbacks": drug_baseline_records,
        "global_baseline": train_df[TARGET_COLUMNS].mean().to_dict(),
        "gene_ids": bundle.gene_ids, "drug_to_idx": bundle.drug_to_idx,
        "dataset_to_idx": bundle.dataset_to_idx, "gene_embedding_file": str(args.gene_embeddings.resolve()),
        "best_epoch": best_epoch,
        "model_config": {"projection_dim": args.projection_dim, "drug_dim": args.drug_dim,
                         "n_components": n_components, "dropout": args.dropout,
                         "ranking_loss_weight": args.ranking_loss_weight,
                         "prediction_target": ("training-fold drug-dataset-centered residual"
                                               if within_drug_mode else "raw response")},
    }, checkpoint_dir / f"{kind}_fold_{fold:02d}.pt")
    return prediction, best_epoch, best_loss, test_baseline


def drug_mean_predictions(bundle, train_rows, test_rows, prediction_mode: str):
    train = bundle.responses.iloc[train_rows]
    test = bundle.responses.iloc[test_rows]
    if prediction_mode == "within_drug":
        return drug_dataset_baselines(train, test)
    global_mean = train[TARGET_COLUMNS].mean()
    drug_means = train.groupby("DRUG_ID")[TARGET_COLUMNS].mean()
    return np.vstack([
        drug_means.loc[drug].to_numpy(dtype=np.float32) if drug in drug_means.index
        else global_mean.to_numpy(dtype=np.float32)
        for drug in test["DRUG_ID"]
    ])


def main():
    parser = argparse.ArgumentParser(description="Train grouped-CV GoBERT/expression drug-response baselines.")
    parser.add_argument("--expression", type=Path, default=DEFAULT_OMICS_DIR / "gene_expression_breast.csv",
                        help="Breast expression CSV (default: mounted Drive OMICS/gene seq folder)")
    parser.add_argument("--responses", type=Path, default=DEFAULT_OMICS_DIR / "drug_response_breast_matched.csv",
                        help="Matched drug-response CSV (default: mounted Drive OMICS/gene seq folder)")
    parser.add_argument("--gene-embeddings", type=Path,
                        default=DEFAULT_OMICS_DIR / "human_gene_gobert_embeddings.npz",
                        help="Packed NPZ made by export_gene_embeddings.py")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output folder (defaults to a mode-specific folder inside this project)")
    parser.add_argument("--prediction-mode", choices=["global_cell_line", "within_drug"],
                        default="global_cell_line",
                        help="Global raw-response regression or drug/dataset-centered regression plus ranking")
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
    parser.add_argument("--ranking-loss-weight", type=float, default=0.1,
                        help="Weight for pairwise ranking loss within each drug and dataset")
    parser.add_argument("--ranking-tie-epsilon", type=float, default=1e-4,
                        help="Ignore response pairs closer than this after residual target scaling")
    parser.add_argument("--min-ranking-samples", type=int, default=4,
                        help="Minimum held-out cell lines in a drug/dataset/fold group for ranking metrics")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.output_dir is None:
        mode_folder = "breast_global_cell_line_v2" if args.prediction_mode == "global_cell_line" else "breast_within_drug_v1"
        args.output_dir = PROJECT_DIR / "results" / mode_folder
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
    baseline_oof = np.full((len(response), 2), np.nan, dtype=np.float32)
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
            prediction, best_epoch, best_loss, test_baseline = train_neural_model(
                name, bundle, train_rows, validation_rows, test_rows, standardized, pcs,
                n_components, args, device, fold, checkpoint_dir,
            )
            oof[name][test_rows] = prediction
            baseline_oof[test_rows] = test_baseline
            logs.append({"fold": fold, "model": name, "n_train_samples": len(train_rows),
                         "n_validation_samples": len(validation_rows), "n_test_samples": len(test_rows),
                         "n_train_cell_lines": response.iloc[train_rows]["ModelID"].nunique(),
                         "n_validation_cell_lines": response.iloc[validation_rows]["ModelID"].nunique(),
                         "n_test_cell_lines": response.iloc[test_rows]["ModelID"].nunique(),
                         "best_epoch": best_epoch, "best_validation_objective": best_loss})
        oof["drug_mean"][test_rows] = drug_mean_predictions(
            bundle, train_rows, test_rows, args.prediction_mode,
        )
        if np.isnan(baseline_oof[test_rows]).any():
            train_df = response.iloc[train_rows]
            baseline_oof[test_rows] = drug_dataset_baselines(train_df, response.iloc[test_rows])
        print(f"Completed fold {fold}/{args.folds} on {device}; held out {response.iloc[test_rows]['ModelID'].nunique()} cell lines")

    if any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("Some samples did not receive an out-of-fold prediction")
    output = response[["sample_id", "ModelID", "DATASET", "DRUG_ID", "DRUG_NAME", "AUC_PUBLISHED", "IC50_PUBLISHED", "log2_IC50_PUBLISHED"]].copy()
    output = output.rename(columns={"AUC_PUBLISHED": "true_AUC_PUBLISHED", "IC50_PUBLISHED": "true_IC50_PUBLISHED",
                                    "log2_IC50_PUBLISHED": "true_log2_IC50_PUBLISHED"})
    output["fold"] = fold_ids
    output["train_fold_drug_dataset_mean_auc"] = baseline_oof[:, 0]
    output["train_fold_drug_dataset_mean_log2_ic50"] = baseline_oof[:, 1]
    output["true_AUC_within_drug"] = output["true_AUC_PUBLISHED"].to_numpy() - baseline_oof[:, 0]
    output["true_log2_IC50_within_drug"] = output["true_log2_IC50_PUBLISHED"].to_numpy() - baseline_oof[:, 1]
    for name, values in oof.items():
        output[f"pred_{name}_auc"] = values[:, 0]
        output[f"pred_{name}_log2_ic50"] = values[:, 1]
        output[f"pred_{name}_ic50"] = np.exp2(values[:, 1])
        output[f"pred_{name}_auc_within_drug"] = values[:, 0] - baseline_oof[:, 0]
        output[f"pred_{name}_log2_ic50_within_drug"] = values[:, 1] - baseline_oof[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    global_metrics, macro_metrics, per_drug_metrics = summarize_predictions(output, model_names)
    within_summary, within_by_group = summarize_within_drug(
        output, model_names, min_rank_samples=args.min_ranking_samples,
    )
    global_metrics.to_csv(args.output_dir / "metrics_global.csv", index=False)
    macro_metrics.to_csv(args.output_dir / "metrics_per_drug_macro.csv", index=False)
    per_drug_metrics.to_csv(args.output_dir / "metrics_per_drug.csv", index=False)
    within_summary.to_csv(args.output_dir / "metrics_within_drug_summary.csv", index=False)
    within_by_group.to_csv(args.output_dir / "metrics_within_drug_by_group.csv", index=False)
    pd.DataFrame(logs).to_csv(args.output_dir / "training_folds.csv", index=False)
    config = vars(args)
    config.update({"device_resolved": str(device), "n_samples": int(len(response)),
                   "n_cell_lines": int(response["ModelID"].nunique()), "n_drugs": int(response["DRUG_ID"].nunique()),
                   "n_aligned_genes": int(len(bundle.gene_ids)), "gobert_embedding_dim": int(bundle.gene_embeddings.shape[1])})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(global_metrics.to_string(index=False))
    print("\nWithin-drug residual regression and ranking metrics:")
    print(within_summary.to_string(index=False))
    print(f"Saved out-of-fold predictions and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
