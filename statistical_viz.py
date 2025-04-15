# statistical_observation.py (Interactive Display Version)

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Ensure this is imported
import yaml
import os
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from scipy import stats # For confidence intervals, distributions

# --- Constants and Configuration ---

# File Paths (Update if necessary)
DATA_DIR = "data"
EXP1_CSV = os.path.join(DATA_DIR, "waypoints_1.csv")
EXP1_YAML = os.path.join(DATA_DIR, "environment_config_1.yaml")
EXP2_CSV = os.path.join(DATA_DIR, "waypoints_2.csv")
EXP2_YAML = os.path.join(DATA_DIR, "environment_config_2.yaml")
# PLOT_OUTPUT_DIR = "interface_plots" # No longer saving automatically

# Interface Names (in expected order of segmentation)
INTERFACE_NAMES = [
    "button_2", "button_3", "button_4", "button_5_and_6", "switch_0",
    "switch_1", "knob_2_and_3", "lever_4_7", "lever_8_11", "knob_12",
    "switch_13", "knob_14_18"
]

# Filtering Parameter
MIN_WAYPOINTS_PER_INTERFACE = 3
CONFIDENCE_LEVEL = 0.95 # For ellipsoids

# DH Parameters & TCP Transform (copied from previous script)
DH_PARAMS = [
    [0.1625,  0,        math.pi/2], [0, -0.425, 0], [0, -0.3922, 0],
    [0.1333,  0,        math.pi/2], [0.0997, 0, -math.pi/2], [0.0996, 0, 0]
]
T_6_TCP = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1565], [0, 0, 0, 1]
])

# --- Helper Functions (FK, YAML read, transforms, ellipsoid - unchanged) ---

def dh_transformation_matrix(theta, d, a, alpha):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
        [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def calculate_fk_position(joint_angles, dh_parameters, flange_to_tcp_transform):
    T_0_6 = np.identity(4)
    for i in range(len(dh_parameters)):
        T_0_6 = T_0_6 @ dh_transformation_matrix(joint_angles[i], *dh_parameters[i])
    T_0_TCP = T_0_6 @ flange_to_tcp_transform
    return T_0_TCP[0:3, 3]

def read_aruco_transform_from_yaml(filepath):
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r') as f: data = yaml.safe_load(f)
        if 'aruco_device' in data:
            if 'matrix' in data['aruco_device']:
                matrix = np.array(data['aruco_device']['matrix'])
                if matrix.shape == (4, 4): return matrix
            if 'pose' in data['aruco_device']:
                 pose = data['aruco_device']['pose']
                 pos_list = pose['position']
                 quat_list = pose['quaternion']
                 if all(isinstance(sub, list) and len(sub) == 1 for sub in pos_list):
                     pos = np.array([p[0] for p in pos_list])
                 else: pos = np.array(pos_list)
                 if pos.shape == (3,) and len(quat_list) == 4:
                     rot = R.from_quat(quat_list).as_matrix()
                     T = np.identity(4); T[:3, :3] = rot; T[:3, 3] = pos
                     return T
    except Exception as e: print(f"Error reading YAML {filepath}: {e}")
    print(f"Error: Could not extract valid 4x4 transform from {filepath}")
    return None

def invert_transform(T):
    R = T[0:3, 0:3]; t = T[0:3, 3]
    R_inv = R.T; t_inv = -R_inv @ t
    T_inv = np.identity(4); T_inv[0:3, 0:3] = R_inv; T_inv[0:3, 3] = t_inv
    return T_inv

def transform_point(T, p):
    p_homog = np.append(p, 1.0)
    p_transformed_homog = T @ p_homog
    return p_transformed_homog[0:3]

def get_confidence_ellipsoid(points, confidence=0.95):
    if points.shape[0] < 4: return None, None, None
    center = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    if not np.all(np.linalg.eigvalsh(cov) >= -1e-9): return None, None, None
    scale = np.sqrt(stats.chi2.ppf(confidence, df=3))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues[eigenvalues < 0] = 0
    radii = scale * np.sqrt(eigenvalues)
    u = np.linspace(0, 2 * np.pi, 50); v = np.linspace(0, np.pi, 50)
    x_sph = np.outer(np.cos(u), np.sin(v)); y_sph = np.outer(np.sin(u), np.sin(v))
    z_sph = np.outer(np.ones_like(u), np.cos(v))
    ellipsoid_points = np.stack((x_sph.ravel(), y_sph.ravel(), z_sph.ravel()), axis=0)
    ellipsoid_transformed = eigenvectors @ np.diag(radii) @ ellipsoid_points
    ellipsoid_final = ellipsoid_transformed + center[:, np.newaxis]
    ell_x = ellipsoid_final[0, :].reshape(x_sph.shape)
    ell_y = ellipsoid_final[1, :].reshape(x_sph.shape)
    ell_z = ellipsoid_final[2, :].reshape(x_sph.shape)
    return ell_x, ell_y, ell_z

# --- Main Data Processing Function (unchanged) ---

def process_experiment_data(csv_filepath, yaml_filepath, dh_params, t_6_tcp):
    # (Same implementation as the previous 'saving' version)
    print(f"\n--- Processing Experiment: {os.path.basename(csv_filepath)} ---")
    if not os.path.exists(csv_filepath): return None
    T_base_aruco = read_aruco_transform_from_yaml(yaml_filepath)
    if T_base_aruco is None: return None
    try: T_aruco_base = invert_transform(T_base_aruco)
    except Exception as e: print(f"Error inverting transform: {e}"); return None
    try: data = pd.read_csv(csv_filepath)
    except Exception as e: print(f"Error reading CSV {csv_filepath}: {e}"); return None
    required_cols = ['joint1pose', 'joint2pose', 'joint3pose', 'joint4pose', 'joint5pose', 'joint6pose', 'GripperState']
    if not all(col in data.columns for col in required_cols): print("Error: Missing columns"); return None

    processed_interfaces = {}
    current_block_points_aruco = []
    interface_index_counter = 0
    is_gripper_closed = False
    ignore_next_point = False
    last_gripper_state = -1

    for index, row in data.iterrows():
        try:
            current_gripper_state = int(float(row['GripperState']))
            if current_gripper_state not in [0, 1]: current_gripper_state = last_gripper_state; continue

            if not is_gripper_closed and current_gripper_state == 1:
                is_gripper_closed = True; current_block_points_aruco = []; ignore_next_point = True
            elif is_gripper_closed and current_gripper_state == 0:
                is_gripper_closed = False; ignore_next_point = False
                if len(current_block_points_aruco) >= MIN_WAYPOINTS_PER_INTERFACE:
                    print(f"  -> Storing Interface Block {interface_index_counter} ({len(current_block_points_aruco)} pts).")
                    processed_interfaces[interface_index_counter] = current_block_points_aruco
                    interface_index_counter += 1
                elif len(current_block_points_aruco) > 0: print(f"  -> Discarding block ({len(current_block_points_aruco)} pts).")
                current_block_points_aruco = []
            elif is_gripper_closed and current_gripper_state == 1:
                if ignore_next_point: ignore_next_point = False
                else:
                    joint_angles = row[['joint1pose', 'joint2pose', 'joint3pose', 'joint4pose', 'joint5pose', 'joint6pose']].values.astype(float)
                    P_base_tcp = calculate_fk_position(joint_angles, dh_params, t_6_tcp)
                    P_aruco_tcp = transform_point(T_aruco_base, P_base_tcp)
                    current_block_points_aruco.append(P_aruco_tcp)
            last_gripper_state = current_gripper_state
        except Exception as e: print(f"Error processing row {index}: {e}. Skipping.")

    if is_gripper_closed and len(current_block_points_aruco) >= MIN_WAYPOINTS_PER_INTERFACE:
         print(f"  -> Storing final Block {interface_index_counter} ({len(current_block_points_aruco)} pts).")
         processed_interfaces[interface_index_counter] = current_block_points_aruco
         interface_index_counter += 1
    elif is_gripper_closed and len(current_block_points_aruco) > 0: print(f"  -> Discarding final block ({len(current_block_points_aruco)} pts).")

    print(f"--- Finished Processing. Found {len(processed_interfaces)} valid interface blocks. ---")
    return processed_interfaces


# --- Plotting Function for a Single Interface (MODIFIED) ---

def plot_interface_analysis(interface_name, data_exp1, data_exp2): # Removed output_dir
    """Generates and displays a detailed analysis plot for one interface."""

    print(f"\nDisplaying analysis plot for: {interface_name}")
    # No filename or saving logic needed here

    all_data = np.vstack((data_exp1, data_exp2))
    n1, n2 = data_exp1.shape[0], data_exp2.shape[0]
    mean1 = np.mean(data_exp1, axis=0); mean2 = np.mean(data_exp2, axis=0)
    mean_all = np.mean(all_data, axis=0)

    # --- Create Figure (same layout) ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1), height_ratios=(2, 1))
    fig.suptitle(f"Interface Analysis: {interface_name}\n(Exp1: {n1} pts, Exp2: {n2} pts)", fontsize=16, y=0.98)

    # --- Panel 1: 3D Scatter Plot with Ellipsoids (same plotting logic) ---
    ax_3d = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax_3d.scatter(data_exp1[:, 0], data_exp1[:, 1], data_exp1[:, 2], c='blue', marker='.', alpha=0.6, label='Exp 1 Data')
    ax_3d.scatter(data_exp2[:, 0], data_exp2[:, 1], data_exp2[:, 2], c='orange', marker='.', alpha=0.6, label='Exp 2 Data')
    ax_3d.scatter(mean1[0], mean1[1], mean1[2], c='darkblue', marker='X', s=100, depthshade=False, label=f'Exp 1 Mean')
    ax_3d.scatter(mean2[0], mean2[1], mean2[2], c='darkorange', marker='X', s=100, depthshade=False, label=f'Exp 2 Mean')
    ax_3d.scatter(mean_all[0], mean_all[1], mean_all[2], c='red', marker='P', s=120, depthshade=False, label=f'Overall Mean')
    ell_x1, ell_y1, ell_z1 = get_confidence_ellipsoid(data_exp1, CONFIDENCE_LEVEL)
    if ell_x1 is not None: ax_3d.plot_wireframe(ell_x1, ell_y1, ell_z1, color='darkblue', alpha=0.2, linewidth=0.5, label=f'Exp 1 {int(CONFIDENCE_LEVEL*100)}% CI Ellipsoid')
    ell_x2, ell_y2, ell_z2 = get_confidence_ellipsoid(data_exp2, CONFIDENCE_LEVEL)
    if ell_x2 is not None: ax_3d.plot_wireframe(ell_x2, ell_y2, ell_z2, color='darkorange', alpha=0.2, linewidth=0.5, label=f'Exp 2 {int(CONFIDENCE_LEVEL*100)}% CI Ellipsoid')
    ax_3d.set_xlabel('X (ArUco Frame) [m]'); ax_3d.set_ylabel('Y (ArUco Frame) [m]'); ax_3d.set_zlabel('Z (ArUco Frame) [m]')
    ax_3d.set_title(f'3D Position & {int(CONFIDENCE_LEVEL*100)}% Confidence Ellipsoids')
    all_points_for_scale = np.vstack((data_exp1, data_exp2, mean_all.reshape(1,-1)))
    max_range = np.ptp(all_points_for_scale, axis=0).max()
    mid = np.mean(all_points_for_scale, axis=0)
    ax_3d.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax_3d.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax_3d.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    handles, labels = ax_3d.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax_3d.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))

    # --- Panels 2, 3, 4: Violin Plots (same plotting logic) ---
    df_list = []
    coords = ['X', 'Y', 'Z']
    for i, coord in enumerate(coords):
        for val in data_exp1[:, i]: df_list.append({'Value': val, 'Coordinate': coord, 'Experiment': 'Exp 1'})
        for val in data_exp2[:, i]: df_list.append({'Value': val, 'Coordinate': coord, 'Experiment': 'Exp 2'})
    plot_df = pd.DataFrame(df_list)
    violin_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, coord in enumerate(coords):
        ax_v = violin_axes[i]
        sns.violinplot(x='Experiment', y='Value', data=plot_df[plot_df['Coordinate'] == coord], hue='Experiment', palette={'Exp 1': 'lightblue', 'Exp 2': 'navajowhite'}, inner='quartile', ax=ax_v, legend=False)
        sns.stripplot(x='Experiment', y='Value', data=plot_df[plot_df['Coordinate'] == coord], hue='Experiment', palette={'Exp 1': 'blue', 'Exp 2': 'orange'}, size=3, alpha=0.7, jitter=0.1, dodge=True, ax=ax_v, legend=False)
        ax_v.set_title(f'{coord} Distribution')
        ax_v.set_xlabel(""); ax_v.set_ylabel(f'{coord} Value [m]')
        ax_v.grid(True, axis='y', linestyle=':')

    # --- Final Adjustments & SHOW ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(filename, dpi=150) # REMOVED saving
    # plt.close(fig) # REMOVED closing
    plt.show() # ADDED show - this will block until the window is closed


# --- Main Execution (MODIFIED LOOP) ---

if __name__ == "__main__":

    exp1_data = process_experiment_data(EXP1_CSV, EXP1_YAML, DH_PARAMS, T_6_TCP)
    exp2_data = process_experiment_data(EXP2_CSV, EXP2_YAML, DH_PARAMS, T_6_TCP)

    if exp1_data is None or exp2_data is None:
        print("\nAborting due to errors during data processing.")
        exit()

    combined_data = {}
    num_interfaces = min(len(exp1_data), len(exp2_data), len(INTERFACE_NAMES))
    if len(exp1_data) != len(exp2_data): print(f"\nWarning: Mismatched block counts ({len(exp1_data)} vs {len(exp2_data)}). Plotting {num_interfaces}.")
    if num_interfaces < len(INTERFACE_NAMES): print(f"\nWarning: Found {num_interfaces} blocks, but {len(INTERFACE_NAMES)} names provided. Using first {num_interfaces}.")

    valid_interface_names = []
    for i in range(num_interfaces):
        name = INTERFACE_NAMES[i]
        if i in exp1_data and i in exp2_data:
             valid_interface_names.append(name)
             combined_data[name] = {
                 'exp1': np.array(exp1_data[i]),
                 'exp2': np.array(exp2_data[i])
             }

    if not combined_data:
        print("\nNo valid interface data found or combined. Cannot generate plots.")
        exit()

    # --- Generate and SHOW plot for each interface sequentially ---
    # Removed creation of output directory
    print(f"\nGenerating and showing {len(combined_data)} interface analysis plots interactively...")
    print("Close each plot window to view the next one.")

    for name, data in combined_data.items():
        # Call the plotting function - it will now block until closed
        plot_interface_analysis(name, data['exp1'], data['exp2']) # Removed output_dir argument

    print("\nAll plots displayed. Script finished.")