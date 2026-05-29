import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
### Paths

obs_labels_path = "/jukebox/falkner/Jorge/Dexter_results/all_obs_xpo_pred_dict_v2_102325.pkl"
attack_labels_path ="/jukebox/falkner/Jorge/PhotometryFiles/reviews/fully_labeled_traces_feats3_082925.pickle"
fig_path = "/usr/people/tt1131/projects/social_dojo_observer/fig"
fig_pub_path = os.path.join(fig_path, "pub")

### Data and functions
fps = 40

class_to_label_dict = {'grooming': 0, 'investigate': 1, 'rearing': 2, 'scratching': 3, 'sniffing': 4, 'still': 5, 'turning': 6}
label_to_class_dict = {v: k for k, v in class_to_label_dict.items()}

label_to_cond_dict = {
    '1185': "obs", '30R2': "obs", '29L': "obs",
    '1162B': "obs", '87L2': "obs", '933R': "obs",
    '86L': "obs", '927R': "obs", '927L': "obs",
    '4321L': "xpo", '7452R': "xpo", '7452L2': "xpo",
    '4321L2': "xpo", '4321R2': "xpo",
}

# 1: Attentive state, 2: self-directed, 3: Idle, 4: Explortory
categories_label_dict = {
    1: "Attention",
    2: "Self-directed",
    3: "Idle",
    4: "Exploratory"
}
label_to_categories_dict = {
    0: 2,
    1: 4,
    2: 4,
    3: 2,
    4: 1,
    5: 3,
    6: 4
}
label_to_categories = lambda x: np.vectorize(label_to_categories_dict.get)(x)

categories_cmap = ["dodgerblue", "gold", "lightgray", "darkorange"]

mouse_to_id_dict = {
    "mouse0": "30R2",
    "mouse1": "29L",
    "mouse2": "86L",
    "mouse3": "87L2",
    "mouse4": "927R",
    "mouse5": "927L",
    "mouse6": "933R",
    "mouse7": "1185",
    "mouse8": "1162B"
}

id_to_mouse_dict = {v: k for k, v in mouse_to_id_dict.items()}

region_labels = [
    'PrL (E)', 'PrL (I)', 'vLS (E)', 'vLS (I)', 
    'POA (E)', 'POA (I)','BNST (E)', 'BNST (I)', 
    'AH (E)', 'AH (I)', 'MeA (E)', 'MeA (I)',
    'VMH (E)', 'VMH (I)', 'PAG (E)', 'PAG (I)', 
    'PMv (E)', 'PMv (I)', 'LHb (E)', 'LHb (I)', 
    'PA (E)', 'PA (I)', 'NAc (DA)',
]

def count_state_lengths(state_array, state_lengths_dict=None):
    """
    Count the lengths of consecutive states in an array with discrete values.
    
    Parameters:
    state_array (array-like): Array containing discrete state values
    state_lengths_dict (dict, optional): Existing dictionary to append to. If None, creates new dict.
    
    Returns:
    dict: Dictionary where keys are state values and values are lists of consecutive lengths
    """
    if len(state_array) == 0:
        return state_lengths_dict if state_lengths_dict is not None else {}
    
    if state_lengths_dict is None:
        state_lengths = {}
    else:
        state_lengths = state_lengths_dict
    
    current_state = state_array[0]
    current_length = 1
    
    for i in range(1, len(state_array)):
        if state_array[i] == current_state:
            current_length += 1
        else:
            # State changed, record the length
            if current_state not in state_lengths:
                state_lengths[current_state] = []
            state_lengths[current_state].append(current_length)
            
            # Start counting new state
            current_state = state_array[i]
            current_length = 1
    
    # Don't forget the last state
    if current_state not in state_lengths:
        state_lengths[current_state] = []
    state_lengths[current_state].append(current_length)
    
    return state_lengths

def find_continuous_ones(arr):
    """
    Find all continuous runs of 1s in a numpy array.
    Returns list of (start_idx, end_idx) for each run (end_idx is exclusive).
    """
    arr = np.asarray(arr)
    is_one = arr == 1
    changes = np.diff(is_one.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if is_one[0]:
        starts = np.r_[0, starts]
    if is_one[-1]:
        ends = np.r_[ends, len(arr)]

    return list(zip(starts, ends))

def setup_plot_style():
    plt.rcParams['font.family'] = 'Nimbus Sans'
    plt.rcParams['font.size'] = 15
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['savefig.facecolor'] = 'none'
    plt.rcParams['savefig.transparent'] = True
    
    
from scipy.stats import mode
from numpy.lib.stride_tricks import sliding_window_view

def rolling_mode_fast(labels, window_size = 59):
    """Fast rolling mode using strided views."""
    # Pad to handle edges
    pad_width = window_size // 2
    padded = np.pad(labels, pad_width, mode='edge')
    
    # Create sliding windows and compute mode
    windows = sliding_window_view(padded, window_size)
    return mode(windows, axis=1, keepdims=False).mode



def generate_full_data_dict(obs_data_dict, 
                            neural_beh_data_dict, 
                            mouse_id_list, 
                            label_to_cond_dict,
                            days: np.array = np.arange(1, 9),
                            sessions: np.array = np.arange(1, 4),
                            smoothing_func = rolling_mode_fast):
    full_data_dict = {} ## Contains neural, behavioral and observer predictions

    for idx, mouse_id in enumerate(mouse_id_list):
        pred_d_s_pd = obs_data_dict[mouse_id]
        pred_d_s_pd["smoothed_prediction"] = 0
        mouse_full_dict = {}
        for day_idx in days:
            for session_idx in sessions:
                pred_d_s = pred_d_s_pd[(pred_d_s_pd["day_id"] == day_idx) & (pred_d_s_pd["session_id"] == session_idx)]
                neural_beh_dict_label = f'{mouse_id}_d{day_idx}_{label_to_cond_dict[mouse_id]}_t{session_idx}'
                neural_beh_session = neural_beh_data_dict[neural_beh_dict_label]
                if len(pred_d_s) == 0:
                    print(f"No data for {mouse_id} day {day_idx} session {session_idx}")
                    continue
                pred_smpl = label_to_categories(pred_d_s["prediction"].values)
                smoothed_pred = smoothing_func(pred_smpl)
                length_diff = len(pred_d_s) - len(neural_beh_session)
                if length_diff < 0:
                    print(f"{mouse_id}, Day {day_idx}, Session {session_idx}, skipped due to missing neural data (diff = {length_diff})")
                else:
                    neural_beh_session["smooth_obs_pred"] = smoothed_pred[length_diff: ] ## we crop the first part out
                    neural_beh_session["obs_pred"] = pred_d_s["prediction"].values[length_diff: ]
                mouse_full_dict[f"d{day_idx}_s{session_idx}"] = neural_beh_session
        full_data_dict[mouse_id] = mouse_full_dict
    return full_data_dict

def extract_bouts_with_sessions(
    full_data_dict,
    obs_mouse_id_list,
    region_labels,
    bout_duration=21,
    intvl=10,
    major_beh_prop=0.7
):
    """
    Extract neural bouts with session labels for proper cross-validation.
    
    Returns:
        decoder_data_dict: {mice_id: (X, y, session_labels, bout_info)}
            X: (n_samples, n_timepoints, n_regions)
            y: (n_samples,) behavior labels
            session_labels: (n_samples,) session identity for each bout
            bout_info: dict with metadata about extraction
    """
    decoder_data_dict = {}
    
    for mice_id in obs_mouse_id_list:
        x_features = []
        y_labels = []
        session_ids = []
        
        # Track statistics for reporting
        total_windows = 0
        accepted_windows = 0
        
        for session_idx, (day_sesh_label, dat_pd) in enumerate(full_data_dict[mice_id].items()):
            if "smooth_obs_pred" not in dat_pd.columns:
                continue
            
            # Pre-extract arrays
            obs_pred = dat_pd['smooth_obs_pred'].values
            region_data = dat_pd[region_labels].values
            n_samples = len(dat_pd)
            
            for start_idx in range(0, n_samples - bout_duration, intvl):
                total_windows += 1
                end_idx = start_idx + bout_duration
                window_labels = obs_pred[start_idx:end_idx]
                
                unique, counts = np.unique(window_labels, return_counts=True)
                max_idx = counts.argmax()
                
                if counts[max_idx] / bout_duration >= major_beh_prop:
                    accepted_windows += 1
                    x_features.append(region_data[start_idx:end_idx])
                    y_labels.append(int(unique[max_idx]))
                    session_ids.append(session_idx)
        
        X = np.array(x_features)  # (n_samples, n_timepoints, n_regions)
        y = np.array(y_labels)
        session_labels = np.array(session_ids)
        
        bout_info = {
            'total_windows': total_windows,
            'accepted_windows': accepted_windows,
            'acceptance_rate': accepted_windows / total_windows if total_windows > 0 else 0,
            'n_sessions': len(np.unique(session_labels)),
            'class_distribution': {int(u): int(c) for u, c in zip(*np.unique(y, return_counts=True))}
        }
        
        decoder_data_dict[mice_id] = (X, y, session_labels, bout_info)
        
        print(f"Mouse {mice_id}: {accepted_windows}/{total_windows} bouts accepted "
              f"({bout_info['acceptance_rate']:.1%}), {bout_info['n_sessions']} sessions")
        print(f"  Class distribution: {bout_info['class_distribution']}")
    
    return decoder_data_dict