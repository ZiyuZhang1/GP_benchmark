from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .config import Config


def read_data(disease: str, dga: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build labeled dataframe for a single disease without time split.
    """
    columns_to_keep = [col for col in features.columns if col.startswith("feature") or col.startswith("string")]
    df = features[columns_to_keep].copy()
    pos_genes_list = dga[dga["disease_id"] == disease]["string_id"]
    df["label"] = df["string_id"].isin(pos_genes_list).astype(int)
    y = df["label"].to_numpy()
    df.set_index("string_id", inplace=True)
    df.drop(columns="label", inplace=True)
    return df, y


def read_data_timecut(
    disease: str, dga: pd.DataFrame, features: pd.DataFrame, time: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build labeled dataframe with time-based split column 'test'.
    """
    columns_to_keep = [col for col in features.columns if "feature" in col or col.startswith("string")]
    df = features[columns_to_keep].copy()
    pos_genes_list = dga[dga["disease_id"] == disease]["string_id"]
    df["label"] = df["string_id"].isin(pos_genes_list).astype(int)
    df["test"] = df["string_id"].isin(
        dga[(dga["disease_id"] == disease) & (dga["first_pub_year"] > time)]["string_id"]
    ).astype(int)
    y = df["label"].to_numpy()
    df.set_index("string_id", inplace=True)
    df.drop(columns="label", inplace=True)
    return df, y


def get_feature(config: Config, feature_name: str) -> pd.DataFrame:
    """
    Load a feature dataframe by name using config.data_dir.
    """
    data = config.data_dir
    name = feature_name
    if name == "ppi_align":
        path = data / "ppi_full_emb_aligned.csv"
    elif name == "ppi":
        path = data / "ppi_full_emb.csv"
    elif name == "ppi_2017_700":
        path = data / "ppi_full_2016_700_emb.csv"
    elif name == "biograd":
        path = data / "biograd/biograd_full_emb.csv"
    elif name == "biograd_2019_n2v":
        path = data / "biograd/uniport_biogrid_emb_2019.csv"
    elif name == "biograd_2019_dw_40":
        path = data / "biograd/biograd_entrz_2019_dw_emb_40.txt"
    elif name == "biogrid_diffusion_2019":
        path = data / "biogrid_diffusion_2019.csv"
    elif name == "prose":
        path = data / "prose/data/prose_emb_full.csv"
    elif name == "ppi_2016":
        path = data / "ppi_full_2016_emb.csv"
    elif name == "ppi_2017_dw_10":
        path = data / "ppi_full_2016_dw_emb_10.csv"
    elif name == "ppi_2017_dw_40":
        path = data / "ppi_full_2016_dw_emb_40.csv"
    elif name == "ppi_2017_dw_80":
        path = data / "ppi_full_2016_dw_emb_80.csv"
    elif name == "ppi_2019_dw_10":
        path = data / "ppi_full_2019_dw_emb_10.csv"
    elif name == "ppi_2019_dw_40":
        path = data / "ppi_full_2019_dw_emb_40.csv"
    elif name == "ppi_2019_dw_80":
        path = data / "ppi_full_2019_dw_emb_80.csv"
    elif name == "ppi_2019":
        path = data / "ppi_full_2019_emb.csv"
    elif name == "uniport":
        path = data / "pre_processed_features/seq_emb/human_uniport_seqemb.csv"
    elif name == "gene2vec":
        path = data / "pre_processed_features/expression_emb/exp_emb.csv"
    elif name == "ppi_2019_short":
        path = data / "short/ppi_short.csv"
    elif name == "bioconcept_short":
        path = data / "short/bio_short.csv"
    elif name == "scgpt":
        path = data / "scgpt/scgpt_full.csv"
    elif name == "bioconcept":
        path = data / "bioconcept/bioconcept_full.csv"
    elif name == "esm2":
        path = data / "esmfold/esm2.csv"
    elif name == "uniport_ppi_2017":
        path = data / "stringdb/uniport_ppi_2017.csv"
    elif name == "uniport_ppi_2019":
        path = data / "stringdb/uniport_ppi_2019.csv"
    elif name == "uniport_esm":
        path = data / "esmfold/uniport_esm2.csv"
    elif name == "uniport_seq":
        path = data / "pre_processed_features/seq_emb/uniport_emb.csv"
    elif name == "uniport_exp":
        path = data / "pre_processed_features/expression_emb/gene2vec_emb.csv"
    elif name == "uniport_bio":
        path = data / "bioconcept/uniport_bio_emb.csv"
    elif name == "diffusion_2019":
        path = data / "diffusion_2019.csv"
    elif name == "diffusion_2019_2":
        path = data / "diffusion_2019_2.csv"
    elif name == "text":
        path = data / "pre_processed_features/text_mining_processed.csv"
    elif name == "text_2":
        path = data / "pre_processed_features/text_mining_processed_14300.csv"
    elif name == "text_3":
        path = data / "pre_processed_features/text_mining_processed_14164.csv"
    elif name == "df_early":
        path = data / "pre_processed_features/df_early.csv"
    elif name == "df_early_ppi":
        path = data / "pre_processed_features/df_early_ppi.csv"
    else:
        raise ValueError(f"Unknown feature '{feature_name}'")

    return pd.read_csv(path)
