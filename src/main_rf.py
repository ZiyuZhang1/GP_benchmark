import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

from features_reindex import get_feature, read_data_timecut
from model_rf_uni_inductive import neg_bagging_early, neg_bagging_later, eval_bagging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one_fold_evaluate(disease, time, feature_list, df, y, train_idx, test_idx, methods, result_df, fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5 * len(train_pos_df)
    neg_df = df[y == 0]
    neg_df_add_test_pos = pd.concat([neg_df, test_pos_df])

    print('train test split')

    test_neg_df = neg_df
    test_df = pd.concat([test_pos_df, test_neg_df])
    test_index_loc = df.index.get_indexer(test_df.index)
    y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

    predcition_collection = dict()
    predcition_collection['true_label'] = y_test
    predcition_collection["test_genes"] = test_df.index
    predcition_collection["train_pos_genes"] = train_pos_df.index

    num_processes = 2
    base_seed = 42
    seed_list = [base_seed + i for i in range(num_processes)]

    args_list = [
        (neg_df_add_test_pos, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed)
        for seed in seed_list]

    if 'early_fusion' in methods:
        print('early fusion')

        with mp.Pool(processes=num_processes) as pool:
            bagging_y_scores = pool.map(neg_bagging_early, args_list)

        all_preds, all_aucs = zip(*bagging_y_scores)
        final_y_score = np.mean(all_preds, axis=0)
        mean_auc = np.mean(all_aucs)
        print('early fusion validation auc:', mean_auc)

        ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        result_df.loc[len(result_df.index)] = ["random_negative", fold, 'RF_early' + '-0-0-0', *results]
        predcition_collection['RF_early'] = final_y_score

    if 'later_fusion' in methods:
        print('later fusion')
        with mp.Pool(processes=num_processes) as pool:
            bagging_y_scores = pool.map(neg_bagging_later, args_list)

        feature_preds_collection = defaultdict(list)
        fused_preds_collection = []
        lf_mpl_preds_collection = []
        dict_list = []
        for feature_preds, fused_preds, auc_records, lf_mpl_preds in bagging_y_scores:
            dict_list.append(auc_records)
            for feature_name, preds_path in feature_preds.items():
                with open(preds_path, "rb") as f:
                    preds = pickle.load(f)
                feature_preds_collection[feature_name].append(preds)
            fused_preds_collection.append(fused_preds)
            lf_mpl_preds_collection.append(lf_mpl_preds)

        aggregated = defaultdict(list)
        for d in dict_list:
            for key, value in d.items():
                aggregated[key].append(value)

        for feature_name, preds_list in feature_preds_collection.items():
            final_y_score = np.mean(preds_list, axis=0)
            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            result_df.loc[len(result_df.index)] = ["random_negative", fold, 'RF_' + str(feature_name) + '-0-0-0', *results]
            predcition_collection['RF_' + str(feature_name)] = final_y_score

        valid_preds = [p for p in fused_preds_collection if p is not None]
        if valid_preds:
            final_y_score = np.mean(valid_preds, axis=0)
            y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            result_df.loc[len(result_df.index)] = ["random_negative", fold, 'RF_later_avg' + '-0-0-0', *results]
            predcition_collection['RF_later_avg'] = final_y_score

        valid_preds = [p for p in lf_mpl_preds_collection if p is not None]
        if valid_preds:
            final_y_score = np.mean(valid_preds, axis=0)
            y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            result_df.loc[len(result_df.index)] = ["random_negative", fold, 'RF_later_rf' + '-0-0-0', *results]
            predcition_collection['RF_later_rf'] = final_y_score

    return predcition_collection


def evaluate_disease(disease, time, feature_list, df, y, methods, time_spilt):
    result_df = pd.DataFrame(columns=[
        'method', "fold", "para",
        'top_recall_25', 'top_recall_300', 'top_recall_10%', 'top_precision_10%', 'max_precision_10%',
        'top_recall_30%', 'top_precision_30%', 'max_precision_30%',
        'pm_0.5%', 'pm_1%', 'pm_5%', 'pm_10%', 'pm_15%', 'pm_20%', 'pm_25%', 'pm_30%',
        'auroc', "rank_ratio", 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30'
    ])

    test_idx = df[df['test'] == 1].index
    train_idx = df[y == 1].index.difference(test_idx)
    df.drop(columns='test', inplace=True)
    predcition_collection = one_fold_evaluate(disease, time, feature_list, df, y, train_idx, test_idx, methods, result_df, 1)
    return result_df, predcition_collection


def main():
    feature_list = sys.argv[1].split(',')
    out_path = os.path.join(ROOT, sys.argv[2])
    out_path_pred = out_path + '_pred'
    time = int(sys.argv[3])

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_pred, exist_ok=True)

    merged_df = None
    for feature in feature_list:
        feature_df = get_feature(ROOT, feature)

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

    all_df = pd.read_csv(os.path.join(ROOT, 'data/disgent_2020/timecut/dga_time_uniport.csv'))
    all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]

    methods = ['early_fusion', 'later_fusion']

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
        result_df, predcition_collection = evaluate_disease(disease, time, feature_list, df, y, methods, True)
        result_df.to_csv(os.path.join(out_path, f"{disease}.csv"), index=False)
        with open(os.path.join(out_path_pred, f'{disease}_pred.pkl'), 'wb') as f:
            pickle.dump(predcition_collection, f)

        mean_df = result_df.groupby(['method'])[[
            'top_recall_25', 'top_recall_300', 'top_recall_10%', 'top_precision_10%', 'max_precision_10%',
            'top_recall_30%', 'top_precision_30%', 'max_precision_30%',
            'pm_0.5%', 'pm_1%', 'pm_5%', 'pm_10%', 'pm_15%', 'pm_20%', 'pm_25%', 'pm_30%',
            'auroc', "rank_ratio", 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30'
        ]].mean().reset_index()
        mean_df['disease'] = disease
        all_results.append(mean_df)

    final_result = pd.concat(all_results, ignore_index=True)
    final_result.to_csv(os.path.join(out_path, 'all_disease.csv'), index=False)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
