import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

from features_reindex import get_feature, read_data_timecut
from model_diffusion import evaluate_disease

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_list = sys.argv[1].split(',')
out_path = os.path.join(ROOT, sys.argv[2])
out_path_pred = out_path + '_pred'
time = int(sys.argv[3])

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_pred, exist_ok=True)

if time == 2017:
    time_feature_list = ['uniport_ppi_2017', 'ppi_2017_dw_80', 'uniport_exp', 'uniport_seq', 'uniport_esm']
elif time == 2019:
    time_feature_list = ['uniport_ppi_2019', 'ppi_2019_dw_40', 'uniport_bio', 'uniport_seq', 'uniport_esm', 'diffusion_2019']
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

methods = ['random_negative']

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
all_results = []

for disease in selected_diseases:
    print(disease, len(all_df[all_df['disease_id'] == disease]))
    df, y = read_data_timecut(disease, all_df, merged_df, time)
    result_df, prediction_collection = evaluate_disease(disease, time, feature_list, df, y, methods, True)

    with open(os.path.join(out_path_pred, f'{disease}_pred.pkl'), 'wb') as f:
        pickle.dump(prediction_collection, f)

    result_df.to_csv(os.path.join(out_path, f'{disease}.csv'), index=False)
    mean_df = result_df.groupby(['method'])[[
        'top_recall_25', 'top_recall_300', 'top_recall_10%', 'top_precision_10%', 'max_precision_10%',
        'top_recall_30%', 'top_precision_30%', 'max_precision_30%',
        'pm_0.5%', 'pm_1%', 'pm_5%', 'pm_10%', 'pm_15%', 'pm_20%', 'pm_25%', 'pm_30%',
        'auroc', 'rank_ratio', 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30'
    ]].mean().reset_index()
    mean_df['disease'] = disease
    all_results.append(mean_df)

final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(os.path.join(out_path, 'all_disease.csv'), index=False)
