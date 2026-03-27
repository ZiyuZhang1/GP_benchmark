import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
from multiprocessing import Pool

from features_reindex import get_feature
from model_diffusion import eval_bagging
from model_mf import neg_bag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_list = sys.argv[1].split(',')
out_path = os.path.join(ROOT, sys.argv[2])
out_path_pred = out_path + '_pred'
time = int(sys.argv[3])
b_para = sys.argv[4]
n_para = sys.argv[5]

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_pred, exist_ok=True)


def select_feature_block(merged_df: pd.DataFrame, feat: str) -> np.ndarray:
    """Return side-info matrix for ONE feature block in the same row order as merged_df's string_ids."""
    cols = ['string_id'] + [c for c in merged_df.columns if c.startswith(f"{feat}_")]
    if len(cols) <= 1:
        raise ValueError(f"No columns found for feature block: {feat}")
    block_df = merged_df[cols].drop_duplicates("string_id").set_index("string_id")
    return block_df.to_numpy()


if time == 2017:
    time_feature_list = ['uniport_ppi_2017', 'ppi_2017_dw_80', 'uniport_exp', 'uniport_seq', 'uniport_esm']
elif time == 2019:
    time_feature_list = ['uniport_ppi_2019', 'ppi_2019_dw_40', 'uniport_bio', 'uniport_seq', 'uniport_esm', 'diffusion_2019_pca']
else:
    raise ValueError(f"Unsupported time cutoff: {time}. Expected 2017 or 2019.")

merged_df = None
for feature in time_feature_list:
    feature_df = get_feature(ROOT, feature)

    if 'diffusion' not in feature:
        feature_cols = [col for col in feature_df.columns if col.startswith('feature')]
        if feature_cols:
            scaler = MinMaxScaler()
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

    feature_df.rename(columns={
        col: f"{feature}_{col}" if col.startswith('feature') else col
        for col in feature_df.columns
    }, inplace=True)

    if merged_df is None:
        merged_df = feature_df
    else:
        merged_df = pd.merge(merged_df, feature_df, on='string_id', how='inner')
    del feature_df

name_list = feature_list + ['string_id']
merged_df = merged_df[[col for col in merged_df.columns if any(item in col for item in name_list)]]

all_df = pd.read_csv(os.path.join(ROOT, 'data/disgent_2020/timecut/dga_time_uniport.csv'))
all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]

selected_diseases = []
for disease_id in all_df['disease_id'].unique():
    sub_df = all_df[all_df['disease_id'] == disease_id]
    if len(sub_df) < 15:
        continue
    if (sub_df['first_pub_year'].max() > time
            and sub_df['first_pub_year'].min() <= time
            and len(sub_df[sub_df['first_pub_year'] < time]) >= 5):
        selected_diseases.append(disease_id)

print(feature_list, len(selected_diseases), len(merged_df))

all_df = all_df[all_df['string_id'].isin(merged_df['string_id'].unique())]
all_df = all_df[all_df['disease_id'].isin(selected_diseases)]

all_df_train = all_df[all_df['first_pub_year'] <= time][['disease_id', 'string_id']]
all_df_test = all_df[all_df['first_pub_year'] > time][['disease_id', 'string_id']]

string_ids = merged_df["string_id"].unique()
string2idx = {s: i for i, s in enumerate(string_ids)}
string_list = list(string2idx.keys())

disease_ids = all_df_train["disease_id"].unique()
disease2idx = {d: i for i, d in enumerate(disease_ids)}

df = all_df_test[all_df_test["string_id"].isin(string2idx)].copy()
rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)
Y_test = coo_matrix((data, (rows, cols)), shape=(len(disease_ids), len(string_ids))).tocsr()

df = all_df_train[all_df_train["string_id"].isin(string2idx)].copy()
rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)
Y_train = coo_matrix((data, (rows, cols)), shape=(len(disease_ids), len(string_ids))).tocsr()

num_processes = 20
seed_list = [42 + i for i in range(num_processes)]
all_results = []

for feat in feature_list:
    print(f"\n==== Running MF for feature: {feat} ====")

    side_info_feat = select_feature_block(merged_df, feat)

    args_list = [
        (all_df_train, string2idx, disease2idx, side_info_feat, seed, b_para, n_para)
        for seed in seed_list
    ]

    with Pool(processes=num_processes) as pool:
        bag_S = pool.starmap(neg_bag, args_list)

    S_stack = np.stack(bag_S, axis=0)
    S_mean = np.nanmean(S_stack, axis=0)

    del bag_S, S_stack, side_info_feat

    for disease in selected_diseases[:1]:
        disease_idx = disease2idx[disease]

        train_genes_idx = Y_train[disease_idx, :].nonzero()[1]
        train_genes = np.array(string_list)[train_genes_idx]

        all_genes_idx = np.arange(len(string_ids))
        test_genes_idx = np.setdiff1d(all_genes_idx, train_genes_idx)
        test_genes = np.array(string_list)[test_genes_idx]

        final_y_score = S_mean[disease_idx, test_genes_idx]
        y_test = Y_test[disease_idx, test_genes_idx].toarray().ravel()

        pred_path = os.path.join(out_path_pred, f'{disease}_pred.pkl')

        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                prediction_collection = pickle.load(f)
        else:
            prediction_collection = {
                'true_label': y_test,
                'test_genes': test_genes,
                'train_pos_genes': train_genes,
            }

        prediction_collection[feat] = final_y_score

        with open(pred_path, 'wb') as f:
            pickle.dump(prediction_collection, f)

        ranked_predict_index, results = eval_bagging(final_y_score, y_test)

        disease_csv = os.path.join(out_path, f"{disease}.csv")
        columns = [
            'method', 'fold', 'para',
            'top_recall_25', 'top_recall_300', 'top_recall_10%',
            'top_precision_10%', 'max_precision_10%',
            'top_recall_30%', 'top_precision_30%', 'max_precision_30%',
            'pm_0.5%', 'pm_1%', 'pm_5%', 'pm_10%', 'pm_15%', 'pm_20%', 'pm_25%', 'pm_30%',
            'auroc', 'rank_ratio', 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30'
        ]

        if os.path.exists(disease_csv):
            result_df = pd.read_csv(disease_csv)
        else:
            result_df = pd.DataFrame(columns=columns)

        result_df.loc[len(result_df)] = ["random_negative", '0', f"{feat}-0-0-0", *results]
        result_df.to_csv(disease_csv, index=False)

        mean_df = (
            result_df.groupby(['method'])[columns[3:]]
            .mean()
            .reset_index()
        )
        mean_df['disease'] = disease
        mean_df['feature'] = feat
        all_results.append(mean_df)

final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(os.path.join(out_path, 'all_disease.csv'), index=False)
