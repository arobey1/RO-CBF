import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import KDTree
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import os

# Idea: run Mr Peanut again on the safe states, and only enforce the safe
# constraint on the safe states identified by the second run on Mr Peanut,
# leaving a buffer zone in the middle.

def load_data_v2(args, output_map=None):

    # dictionary for storing meta data
    meta_data = {}

    # load data from file
    dfs = [pd.read_pickle(path) for path in args.data_path]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    if output_map is None:
        state_cols = ['cte', 'speed(m/s)', 'theta_e', 'd']
    else:
        state_cols = output_map.state_cols
        df = output_map.map(df)

    disturbance_cols, input_cols = ['dphi_t'], ['input']
    all_cols = state_cols + disturbance_cols + input_cols

    if args.normalize_state is True:
        max_cte = df['cte'].abs().max()
        max_speed = df['speed(m/s)'].abs().max()
        max_theta_e = df['theta_e'].abs().max()
        max_d = df['d'].abs().max()

        meta_data['normalizers'] = {
            'cte': max_cte, 'speed': max_speed, 'theta_e': max_theta_e, 'd': max_d
        }

        df['cte'] = df['cte'] / max_cte
        df['speed(m/s)'] = df['speed(m/s)'] / max_speed
        df['theta_e'] = df['theta_e'] / max_theta_e
        df['d'] = df['d'] / max_d

        # T_x should normalize the state
        # T_x = jnp.diag(jnp.array([
        #     1. / max_cte, 1. / max_speed, 1. / max_theta_e, 1. / max_d
        # ]))
        T_x = jnp.diag(jnp.array([
            max_cte, max_speed, max_theta_e, max_d
        ]))
    else:
        T_x = jnp.eye(4)

    df = df[all_cols]

    if args.data_augmentation is True:
        df_copy = df.copy()
        df_copy['cte'] = df_copy['cte'].multiply(-1)
        df_copy['theta_e'] = df_copy['theta_e'].multiply(-1)
        df_copy['dphi_t'] = df_copy['dphi_t'].multiply(-1)
        df = pd.concat([df, df_copy], ignore_index=True)

    n_all = len(df.index)
    get_bdy_states_v2(df, state_cols, args.nbr_thresh, args.min_n_nbrs)

    if args.n_samp_all != 0:
        df_copy = df.copy()
        df[state_cols] += 0.01 * np.random.randn(*df[state_cols].shape)
        df = pd.concat([df, df_copy])
    
    n_safe, n_unsafe = len(df[df.Safe == 1].index), len(df[df.Safe == 0].index)

    data_dict = {
        'safe': df[df.Safe == 1][state_cols].to_numpy(),
        'unsafe': df[df.Safe == 0][state_cols].to_numpy(),
        'all': df[state_cols].to_numpy(),
        'all_dists': df[disturbance_cols].to_numpy(),
        'all_inputs': df[input_cols].to_numpy()
    }

    create_tables(n_all, n_safe, n_unsafe, args, meta_data)
    _save_meta_data(meta_data, args)

    return data_dict, T_x

def load_data(args):

    if args.system == 'carla':
        state_cols = ['cte', 'speed(m/s)', 'theta_e', 'd']
        disturbance_cols = ['dphi_t']
        dfs = [pd.read_pickle(path) for path in args.data_path]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

        all_states = df[state_cols].to_numpy()  # all_states is [*, STATE_DIM]
        all_disturbances = df[disturbance_cols].to_numpy()
        n_all = all_states.shape[0]
    else:
        raise ValueError(f'System {args.system} is not supported.')

    # Separate safe and unsafe states using neighbor sampling
    safe_states, unsafe_states = get_bdy_states(all_states, args.nbr_thresh, args.min_n_nbrs)
    n_safe, n_unsafe = safe_states.shape[0], unsafe_states.shape[0]

    if args.n_samp_safe != 0:
        samp_safe_states = sample_extra_states(safe_states, args.n_samp_safe)
        safe_states = np.vstack((safe_states, samp_safe_states))

    if args.n_samp_unsafe != 0:
        samp_unsafe_states = sample_extra_states(unsafe_states, args.n_samp_unsafe)
        unsafe_states = np.vstack((unsafe_states, samp_unsafe_states))

    if args.n_samp_all != 0:
        samp_all_states = sample_extra_states(all_states, args.n_samp_all)
        all_states = np.vstack((all_states, samp_all_states))

    create_tables(n_all, n_safe, n_unsafe, args)

    return {
        'safe': jnp.asarray(safe_states), 
        'unsafe': jnp.asarray(unsafe_states), 
        'all': jnp.asarray(all_states)
    }

def get_bdy_states(state_matrix, thresh, min_num_nbrs):
    """Samples boundary/unsafe states from the state matrix.
    
    Args:
        state_matrix: [N, STATE_DIM] matrix containing expert states
            where N is the total number of expert states.
        thresh: Threshold distance that determines whether two states 
            are neighbors or not.
        min_num_neighbors: Minimum number of neighbors a state must have
            to be considered a boundary/unsafe state.

    Returns:
        [N_1, STATE_DIM] array containing N_1 safe states.
        [N_2, STATE_DIM] array containing N_2 boundary/unsafe states.

    Note:
        This procedure partitions the expert states, i.e. N_1 + N_2 = N.
    """

    tree = KDTree(state_matrix)
    dists = tree.query_radius(state_matrix, r=thresh, count_only=True)

    # [N, 1] array which has value 0 if state is safe and value 1 otherwise.
    outlier_mask = np.array(dists < min_num_nbrs)

    safe_states = state_matrix[~outlier_mask]
    unsafe_states = state_matrix[outlier_mask]

    return safe_states, unsafe_states

def get_bdy_states_v2(df, state_cols, thresh, min_num_nbrs):
    state_matrix = df[state_cols].to_numpy()
    tree = KDTree(state_matrix)
    dists = tree.query_radius(state_matrix, r=thresh, count_only=True)
    df['Safe'] = np.array(dists < min_num_nbrs)

def get_bdy_states_v3(df, state_cols, thresh, min_num_nbrs):
    state_matrix = df[state_cols].to_numpy()
    tree = KDTree(state_matrix)
    dists = tree.query_radius(state_matrix, r=thresh, count_only=True)
    df['Unsafe'] = np.array(dists >= min_num_nbrs)
    df['Safe'] = False

    safe_matrix = df[df.Unsafe == 0].to_numpy()
    tree = KDTree(safe_matrix)
    dists = tree.query_radius(safe_matrix, r=thresh / 2, count_only=True)
    df.loc[df.Unsafe == 0, ['Safe']] = np.array(dists < min_num_nbrs)


def sample_extra_states(states, num_samp):
    """Samples extra states in neighborhoods of given states.
    
    Args:
        states: [N, STATE_DIM] matrix containing states.
        num_samp: Number of additional states to sample for each state in {states}.

    Returns:
        [N * n_samp, STATE_DIM] matrix of states.
    """

    extra_states = [states + 0.01 * np.random.randn(*states.shape) for _ in range(num_samp)]
    return np.vstack(extra_states)

def create_tables(n_all, n_safe, n_unsafe, args, meta_data):

    meta_data['pct_safe'] = (n_safe * (args.n_samp_safe + 1)) / (n_all * (args.n_samp_all + 1)) * 100
    meta_data['pct_unsafe'] = (n_unsafe * (args.n_samp_unsafe + 1)) / (n_all * (args.n_samp_all + 1)) * 100
    meta_data['num-expert-states'] = {'all': n_all, 'safe': n_safe, 'unsafe': n_unsafe}
    meta_data['num-samp-states'] = {
        'all': n_all * args.n_samp_all, 'safe': n_safe * args.n_samp_safe, 'unsafe': n_unsafe * args.n_samp_unsafe
    }
    meta_data['num-total-states'] = {
        'all': n_all * (args.n_samp_all + 1), 'safe': n_safe * (args.n_samp_safe + 1), 'unsafe': n_unsafe * (args.n_samp_unsafe + 1)
    }

    expert_table = PrettyTable()
    expert_table.align = 'l'
    expert_table.field_names = ['State type', '# expert states', '# sampled states', 'Total', 'Percent of Total']
    expert_table.add_row(['All', n_all, n_all * args.n_samp_all, n_all * (args.n_samp_all + 1), '--'])
    expert_table.add_row(['Safe', n_safe, n_safe * args.n_samp_safe, n_safe * (args.n_samp_safe + 1), f'{meta_data["pct_safe"]:.1f} %'])
    expert_table.add_row(['Unsafe', n_unsafe, n_unsafe * args.n_samp_unsafe, n_unsafe * (args.n_samp_unsafe + 1), f'{meta_data["pct_unsafe"]:.1f} %'])
    print(expert_table)

def _save_meta_data(meta_data, args):
    """Saves meta data line arguments to JSON file.
    
    Args:
        d: Dictionary of items.
        args: Command line arguments
    """

    fname = os.path.join(args.results_path, 'meta_data.json')
    with open(fname, 'w') as f:
        json.dump(meta_data, f, indent=2)