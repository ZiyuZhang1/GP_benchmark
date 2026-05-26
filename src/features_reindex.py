import pandas as pd
import os

def read_data(disease, dga, features):
    columns_to_keep = [col for col in features.columns if col.startswith('feature') or col.startswith('string')]
    df = features[columns_to_keep]

    pos_genes_list = dga[dga['disease_id']==disease]['string_id']
    df['label'] = df['string_id'].isin(pos_genes_list).astype(int)

    # X = df.loc[:, df.columns.str.startswith("feature_")].to_numpy()
    y = df['label'].to_numpy()
    df.set_index('string_id', inplace=True)
    df.drop(columns='label', inplace=True)
    return df, y

def read_data_timecut(disease, dga, features,time):
    pos_genes_list = dga[dga['disease_id']==disease]['string_id']
    columns_to_keep = [col for col in features.columns if 'feature' in col or col.startswith('string')]
    df = features[columns_to_keep]
    df['label'] = df['string_id'].isin(pos_genes_list).astype(int)
    df['test'] = df['string_id'].isin(dga[(dga['disease_id'] == disease) & (dga['first_pub_year'] > time)]['string_id']).astype(int)

    # X = df.loc[:, df.columns.str.startswith("feature_")].to_numpy()
    y = df['label'].to_numpy()
    df.set_index('string_id', inplace=True)
    df.drop(columns='label', inplace=True)
    return df, y


def get_feature(root, feature_name):
    if feature_name == 'ppi_2019_dw_10':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2019_dw_emb_10.csv'))
    elif feature_name == 'ppi_2019_dw_40':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2019_dw_emb_40.csv'))
    elif feature_name == 'ppi_2019_dw_80':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2019_dw_emb_80.csv'))
    elif feature_name == 'uniport_ppi_2019':
        feature_df = pd.read_csv(os.path.join(root,'data/stringdb/uniport_ppi_2019.csv'))
    elif feature_name == 'uniport_esm':
        feature_df = pd.read_csv(os.path.join(root,'data/esmfold/uniport_esm2.csv'))
    elif feature_name == 'uniport_seq':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/seq_emb/uniport_emb.csv'))
    elif feature_name == 'uniport_bio':
        feature_df = pd.read_csv(os.path.join(root,'data/bioconcept/uniport_bio_emb.csv'))
    elif feature_name == 'diffusion_2019':
        feature_df = pd.read_csv(os.path.join(root,'data/diffusion_2019.csv'))
    elif feature_name == 'diffusion_2019_2':
        feature_df = pd.read_csv(os.path.join(root,'data/diffusion_2019_2.csv'))
    elif feature_name == 'diffusion_2019_pca':
        feature_df = pd.read_csv(os.path.join(root,'data/diffusion_2019_pcs.csv'))
    elif feature_name == 'df_early':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/df_early.csv'))  
    elif feature_name == 'df_early_ppi':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/df_early_ppi.csv'))  
    return feature_df