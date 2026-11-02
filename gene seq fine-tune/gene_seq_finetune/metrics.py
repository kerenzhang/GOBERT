from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 and np.std(y_true) and np.std(y_pred) else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and np.std(y_true) else np.nan
    return {"n": len(y_true), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2),
            "pearson_r": float(pearson), "spearman_rho": float(spearman)}


def summarize_predictions(predictions: pd.DataFrame, model_names: list[str]):
    targets = {
        "AUC_PUBLISHED": ("true_AUC_PUBLISHED", "auc"),
        "log2_IC50_PUBLISHED": ("true_log2_IC50_PUBLISHED", "log2_ic50"),
        "IC50_PUBLISHED": ("true_IC50_PUBLISHED", "ic50"),
    }
    global_rows, per_drug_rows, macro_rows = [], [], []
    for model in model_names:
        for target, (truth_col, suffix) in targets.items():
            pred_col = f"pred_{model}_{suffix}"
            global_rows.append({"model": model, "target": target, "aggregation": "global",
                                **regression_metrics(predictions[truth_col], predictions[pred_col])})
            drug_metrics = []
            for drug_id, group in predictions.groupby("DRUG_ID", sort=False):
                metric = regression_metrics(group[truth_col], group[pred_col])
                per_drug_rows.append({"model": model, "target": target, "DRUG_ID": drug_id,
                                      "DRUG_NAME": group["DRUG_NAME"].iloc[0], **metric})
                drug_metrics.append(metric)
            metric_df = pd.DataFrame(drug_metrics)
            macro_rows.append({"model": model, "target": target, "aggregation": "per_drug_macro_mean",
                               "n_drugs": len(metric_df),
                               **{key: float(metric_df[key].mean()) for key in ["rmse", "mae", "r2", "pearson_r", "spearman_rho"]}})
    return pd.DataFrame(global_rows), pd.DataFrame(macro_rows), pd.DataFrame(per_drug_rows)


def concordance_index(y_true, y_pred):
    """Fraction of non-tied response pairs ordered correctly (prediction ties count half)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) < 2:
        return np.nan, 0
    i, j = np.triu_indices(len(y_true), k=1)
    true_diff = y_true[i] - y_true[j]
    pred_diff = y_pred[i] - y_pred[j]
    comparable = np.abs(true_diff) > 1e-12
    if not comparable.any():
        return np.nan, 0
    concordant = np.sign(true_diff[comparable]) == np.sign(pred_diff[comparable])
    tied_prediction = np.abs(pred_diff[comparable]) <= 1e-12
    scores = concordant.astype(float)
    scores[tied_prediction] = 0.5
    return float(scores.mean()), int(comparable.sum())


def summarize_within_drug(predictions: pd.DataFrame, model_names: list[str], min_rank_samples: int = 4):
    """Summarize fold-centered regression and within-drug cell-line ranking.

    Ranking comparisons stay inside DRUG_ID x DATASET x held-out fold groups so
    predictions from separately trained folds are never ranked against each other.
    """
    targets = {
        "AUC_PUBLISHED": ("true_AUC_PUBLISHED", "true_AUC_within_drug", "auc", "auc_within_drug"),
        "log2_IC50_PUBLISHED": ("true_log2_IC50_PUBLISHED", "true_log2_IC50_within_drug", "log2_ic50", "log2_ic50_within_drug"),
    }
    group_cols = ["DRUG_ID", "DATASET", "fold"]
    groups = list(predictions.groupby(group_cols, sort=False, dropna=False))
    by_group_rows, summary_rows = [], []

    for model in model_names:
        for target, (truth_col, residual_truth_col, suffix, residual_suffix) in targets.items():
            pred_col = f"pred_{model}_{suffix}"
            residual_pred_col = f"pred_{model}_{residual_suffix}"
            pooled_reg = regression_metrics(predictions[residual_truth_col], predictions[residual_pred_col])
            metric_rows = []
            for group_key, group in groups:
                drug_id, dataset, fold = group_key
                y_resid = group[residual_truth_col].to_numpy(dtype=float)
                p_resid = group[residual_pred_col].to_numpy(dtype=float)
                reg = regression_metrics(y_resid, p_resid)
                rank_group = group.groupby("ModelID", as_index=False)[[truth_col, pred_col]].mean()
                y_raw = rank_group[truth_col].to_numpy(dtype=float)
                p_raw = rank_group[pred_col].to_numpy(dtype=float)
                if len(rank_group) >= min_rank_samples:
                    rho = spearmanr(y_raw, p_raw).statistic if np.std(y_raw) and np.std(p_raw) else np.nan
                    c_index, n_pairs = concordance_index(y_raw, p_raw)
                else:
                    rho, c_index, n_pairs = np.nan, np.nan, 0
                row = {
                    "model": model, "target": target, "DRUG_ID": drug_id,
                    "DATASET": dataset, "fold": fold, "n": len(group),
                    "n_cell_lines": int(group["ModelID"].nunique()),
                    "within_rmse": reg["rmse"], "within_mae": reg["mae"],
                    "within_pearson_r": reg["pearson_r"],
                    "within_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "concordance_index": c_index, "n_rank_pairs": n_pairs,
                }
                by_group_rows.append(row)
                metric_rows.append(row)

            metric_df = pd.DataFrame(metric_rows)
            ranked = metric_df[metric_df["concordance_index"].notna()]
            summary_rows.append({
                "model": model, "target": target,
                "sample_pooled_within_rmse": pooled_reg["rmse"],
                "sample_pooled_within_mae": pooled_reg["mae"],
                "sample_pooled_within_pearson_r": pooled_reg["pearson_r"],
                "macro_group_within_rmse": float(metric_df["within_rmse"].mean()),
                "macro_group_within_mae": float(metric_df["within_mae"].mean()),
                "macro_group_within_pearson_r": float(metric_df["within_pearson_r"].mean()),
                "macro_group_spearman_rho": float(metric_df["within_spearman_rho"].mean()),
                "macro_group_concordance_index": float(metric_df["concordance_index"].mean()),
                "n_samples": int(metric_df["n"].sum()),
                "n_drug_dataset_fold_groups": int(len(metric_df)),
                "n_ranked_groups": int(len(ranked)),
                "n_rank_pairs": int(metric_df["n_rank_pairs"].sum()),
                "min_rank_samples": int(min_rank_samples),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(by_group_rows)
