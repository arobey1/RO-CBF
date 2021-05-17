import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import KDTree
from prettytable import PrettyTable

# Idea: run Mr Peanut again on the safe states, and only enforce the safe
# constraint on the safe states identified by the second run on Mr Peanut,
# leaving a buffer zone in the middle.

def load_data_v2(args, output_map=None):

    # load data from file
    dfs = [pd.read_pickle(path) for path in args.data_path]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    if output_map is None:
        state_cols = ['cte', 'speed(m/s)', 'theta_e', 'd']
    else:
        state_cols = output_map.state_cols
        df = output_map.map(df)

    disturbance_cols, input_cols = ['dtheta_t'], ['input']
    all_cols = state_cols + disturbance_cols + input_cols

    print(f"cte max: {df['cte'].abs().max()}")
    print(f"speed(m/s) max: {df['speed(m/s)'].abs().max()}")
    print(f"theta_e max: {df['theta_e'].abs().max()}")
    print(f"d max: {df['d'].abs().max()}")
    print(f"dtheta_t max: {df['dtheta_t'].abs().max()}")
    print(f"input max: {df['input'].abs().max()}")
    quit()

    # normalization
    df['cte'] = df['cte'] / df['cte'].abs().max()
    df['speed(m/s)'] = df['speed(m/s)'] / df['speed(m/s)'].abs().max()
    df['theta_e'] = df['theta_e'] / df['theta_e'].abs().max()
    df['d'] = df['d'] / df['d'].abs().max()
    df['dtheta_t'] = df['dtheta_t'] / df['dtheta_t'].abs().max()
    df['input'] = df['input'] / df['input'].abs().max()

    df = df[all_cols]

    if args.data_augmentation is True:
        df_copy = df.copy()
        df_copy['cte'] = df_copy['cte'].multiply(-1)
        df_copy['theta_e'] = df_copy['theta_e'].multiply(-1)
        df_copy['dtheta_t'] = df_copy['dtheta_t'].multiply(-1)
        df = pd.concat([df, df_copy], ignore_index=True)

    n_all = len(df.index)
    get_bdy_states_v2(df, state_cols, args.nbr_thresh, args.min_n_nbrs)
    n_safe, n_unsafe = len(df[df.Safe == 1].index), len(df[df.Safe == 0].index)

    data_dict = {
        'safe': df[df.Safe == 1][state_cols].to_numpy(),
        'unsafe': df[df.Safe == 0][state_cols].to_numpy(),
        'all': df[state_cols].to_numpy(),
        'all_dists': df[disturbance_cols].to_numpy(),
        'all_inputs': df[input_cols].to_numpy()
    }

    create_tables(n_all, n_safe, n_unsafe, args)
    # quit()
    return data_dict

def load_data(args):

    if args.system == 'carla':
        state_cols = ['cte', 'speed(m/s)', 'theta_e', 'd']
        disturbance_cols = ['dtheta_t']
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

def create_tables(n_all, n_safe, n_unsafe, args):

    pct_safe = f'{(n_safe * (args.n_samp_safe + 1)) / (n_all * (args.n_samp_all + 1)) * 100:.1f} %'
    pct_unsafe = f'{(n_unsafe * (args.n_samp_unsafe + 1)) / (n_all * (args.n_samp_all + 1)) * 100:.1f} %'

    expert_table = PrettyTable()
    expert_table.align = 'l'
    expert_table.field_names = ['State type', '# expert states', '# sampled states', 'Total', 'Percent of Total']
    expert_table.add_row(['All', n_all, n_all * args.n_samp_all, n_all * (args.n_samp_all + 1), '--'])
    expert_table.add_row(['Safe', n_safe, n_safe * args.n_samp_safe, n_safe * (args.n_samp_safe + 1), pct_safe])
    expert_table.add_row(['Unsafe', n_unsafe, n_unsafe * args.n_samp_unsafe, n_unsafe * (args.n_samp_unsafe + 1), pct_unsafe])
    print(expert_table)

