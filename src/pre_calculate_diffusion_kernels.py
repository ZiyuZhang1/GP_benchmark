import networkx as nx
import numpy as np
from pathlib import Path
from scipy.linalg import expm
import pickle
import os
import pandas as pd
import random

_REPO_ROOT = Path(__file__).resolve().parent.parent

# def normalize_kernel(K):
#     diag = np.sqrt(np.diag(K))
#     diag[diag == 0] = 1e-8  # Avoid division by zero
#     for i in range(K.shape[0]):
#         K[i, :] /= diag[i]
#     for j in range(K.shape[1]):
#         K[:, j] /= diag[j]
#     return K

def normalize_kernel(K):
    diag = np.sqrt(np.diag(K))
    diag[diag == 0] = 1e-8  # Avoid division by zero
    return K / (diag[:, None] * diag[None, :])


def diffusion_kernel(G, beta, normalized=True):
    nodes = list(G.nodes())
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=nodes).todense()
    else:
        L = nx.laplacian_matrix(G, nodelist=nodes).todense()
    
    K = expm(-beta * L)

    return np.array(K)

def process_kernel(args):
    K= args

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)  # Avoid log(0)
    K_log = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T

    K_log = 0.5 * (K_log + K_log.T)

    return K_log

def merge_similarity_matrix(sim_matrix, sample_names, merge_groups, delete_list):

    # Step 1: Create a mapping from sample to index for quick lookup
    name_to_index = {name: i for i, name in enumerate(sample_names)}

    # Step 2: Build group name and sample-to-group map
    sample_to_group = {}
    new_names = []
    for group in merge_groups:
        new_name = '_'.join(sorted(group))
        new_names.append(new_name)
        for s in group:
            sample_to_group[s] = new_name

    all_samples = set(sample_names)
    merged_samples = set(sample_to_group.keys())
    kept_samples = sorted(all_samples - merged_samples - set(delete_list))

    # Step 3: Final sample list and group mapping
    final_samples = new_names + kept_samples
    group_map = {name: [name] for name in kept_samples}
    for group in merge_groups:
        group_name = '_'.join(sorted(group))
        group_map[group_name] = group

    # Step 4: Compute average similarities between groups (on the fly)
    new_matrix = pd.DataFrame(
        np.eye(len(final_samples)), 
        index=final_samples, 
        columns=final_samples
    )

    for i, group_i in enumerate(final_samples):
        for j in range(i + 1, len(final_samples)):
            group_j = final_samples[j]
            members_i = group_map[group_i]
            members_j = group_map[group_j]

            sims = []
            for a in members_i:
                for b in members_j:
                    if a == b:
                        continue
                    idx_a = name_to_index.get(a)
                    idx_b = name_to_index.get(b)
                    if idx_a is not None and idx_b is not None:
                        sims.append(sim_matrix[idx_a, idx_b])

            sim_val = np.mean(sims) if sims else 0.0
            new_matrix.loc[group_i, group_j] = sim_val
            new_matrix.loc[group_j, group_i] = sim_val

    return new_matrix, final_samples


def run(setting='2019', debug=False):
    """Compute and save diffusion kernels for the given time setting.

    Output kernels are written to:
        <repo_root>/results/dw_auc_norm/df/<setting>/

    Required inputs:
        <repo_root>/data/diffusion/ppi_full_<setting>.txt   — edge list
        <repo_root>/data/diffusion/<setting>_map.pkl        — [merge_groups, delete_ensp, map_dict_aligned]
    """
    file_path = str(_REPO_ROOT / 'data' / 'diffusion' / f'ppi_full_{setting}.txt')
    map_path  = str(_REPO_ROOT / 'data' / 'diffusion' / f'{setting}_map.pkl')
    save_dir  = str(_REPO_ROOT / 'results' / 'dw_auc_norm' / 'df' / setting)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"PPI edge list not found: {file_path}\n"
            "Place the STRING edge list for the requested year under data/diffusion/."
        )
    if not os.path.isfile(map_path):
        raise FileNotFoundError(
            f"Node-mapping file not found: {map_path}\n"
            "Place the ID-mapping pickle for the requested year under data/diffusion/."
        )

    if debug:
        G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())
        center_node = random.choice(list(G.nodes()))
        G = nx.ego_graph(G, center_node, radius=1)
    else:
        G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())

    nodes_order = list(G.nodes())
    os.makedirs(save_dir, exist_ok=True)

    with open(map_path, 'rb') as f:
        map_info = pickle.load(f)  # [merge_groups, delete_ensp, map_dict_aligned]

    for beta in [0.1, 0.2, 0.5, 0.8, 1, 2]:
        print(f'[pre_calculate_diffusion_kernels] beta={beta}: computing kernel...')
        K_full = diffusion_kernel(G, beta)
        K_full = 0.5 * (K_full + K_full.T)

        print(f'[pre_calculate_diffusion_kernels] beta={beta}: remapping to UniProt IDs...')
        new_matrix, final_samples = merge_similarity_matrix(K_full, nodes_order, map_info[0], map_info[1])
        uniport_id_order = pd.Series(final_samples).map(map_info[2]).tolist()

        print(f'[pre_calculate_diffusion_kernels] beta={beta}: normalising and saving...')
        K_full = new_matrix.to_numpy()
        K_full = normalize_kernel(K_full)
        K_full += np.eye(K_full.shape[0]) * 1e-6
        K_full = 0.5 * (K_full + K_full.T)

        with open(os.path.join(save_dir, f'uniport_ids_diffusion_K_{beta}.pkl'), 'wb') as f:
            pickle.dump(uniport_id_order, f)

        with open(os.path.join(save_dir, f'uniport_diffusion_K_{beta}.pkl'), 'wb') as f:
            pickle.dump(K_full, f)

        logm_k = process_kernel(K_full)
        with open(os.path.join(save_dir, f'uniport_diffusion_logK_{beta}.pkl'), 'wb') as f:
            pickle.dump(logm_k, f)

        print(f'[pre_calculate_diffusion_kernels] beta={beta}: done.')


if __name__ == '__main__':
    run(setting='2019')
