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
