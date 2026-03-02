from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from gene_benchmark.config import Config, default_config
from gene_benchmark.features import get_feature, read_data, read_data_timecut
from gene_benchmark.models.diffusion import evaluate_disease


def run_diffusion(config: Config | None = None, feature_list: Sequence[str] | None = None) -> Path:
    """
    Run diffusion-based evaluation for all eligible diseases using the unified config.
    """
    cfg = config or default_config()
    feats = list(feature_list) if feature_list is not None else list(cfg.feature_list)

    out_path = cfg.results_dir / "diffusion"
    out_path_pred = out_path.with_name(out_path.name + "_pred")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_pred.mkdir(parents=True, exist_ok=True)

    # Prepare merged feature matrix
    merged_df: pd.DataFrame | None = None
    time_feature_list = feats

    for feature in time_feature_list:
        feature_df = get_feature(cfg, feature)
        if "diffusion" not in feature:
            feature_cols = [col for col in feature_df.columns if col.startswith("feature")]
            if feature_cols:
                scaler = MinMaxScaler()
                feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

        feature_df.rename(
            columns={col: f"{feature}_{col}" if col.startswith("feature") else col for col in feature_df.columns},
            inplace=True,
        )

        merged_df = feature_df if merged_df is None else pd.merge(merged_df, feature_df, on="string_id", how="inner")

    name_list = feats + ["string_id"]
    merged_df = merged_df[[col for col in merged_df.columns if any(item in col for item in name_list)]]

    # Load labels
    dga_path = cfg.data_dir / "disgent_2020/timecut/dga_time_uniport.csv"
    all_df = pd.read_csv(dga_path)
    all_df = all_df[all_df["string_id"].isin(merged_df["string_id"])]

    methods = ["random_negative"]
    selected_diseases: list[str] = []
    if cfg.time_split:
        for disease_id in all_df["disease_id"].unique():
            sub_df = all_df[all_df["disease_id"] == disease_id]
            if len(sub_df) < 15:
                continue
            if (
                sub_df["first_pub_year"].max() > cfg.time
                and sub_df["first_pub_year"].min() <= cfg.time
                and len(sub_df[sub_df["first_pub_year"] < cfg.time]) >= 5
            ):
                selected_diseases.append(disease_id)
    else:
        selected_diseases = (
            all_df.groupby("disease_id").filter(lambda x: len(x) > 15)["disease_id"].unique().tolist()
        )

    all_results = []
    for disease in selected_diseases:
        if cfg.time_split:
            df, y = read_data_timecut(disease, all_df, merged_df, cfg.time)
        else:
            df, y = read_data(disease, all_df, merged_df)
        result_df, prediction_collection = evaluate_disease(disease, cfg.time, feats, df, y, methods, cfg.time_split)

        with open(out_path_pred / f"{disease}_pred.pkl", "wb") as f:
            pickle.dump(prediction_collection, f)

        result_df.to_csv(out_path / f"{disease}.csv", index=False)
        mean_df = (
            result_df.groupby(["method"])[
                [
                    "top_recall_25",
                    "top_recall_300",
                    "top_recall_10%",
                    "top_precision_10%",
                    "max_precision_10%",
                    "top_recall_30%",
                    "top_precision_30%",
                    "max_precision_30%",
                    "pm_0.5%",
                    "pm_1%",
                    "pm_5%",
                    "pm_10%",
                    "pm_15%",
                    "pm_20%",
                    "pm_25%",
                    "pm_30%",
                    "auroc",
                    "rank_ratio",
                    "bedroc_1",
                    "bedroc_5",
                    "bedroc_10",
                    "bedroc_30",
                ]
            ]
            .mean()
            .reset_index()
        )
        mean_df["disease"] = disease
        all_results.append(mean_df)

    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
        final_result.to_csv(out_path / "all_disease.csv", index=False)

    return out_path
