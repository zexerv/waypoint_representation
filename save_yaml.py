# save_yaml.py (Flow Style Matrix Output)

import pandas as pd
import numpy as np
import math
import yaml
import os
from scipy.spatial.transform import Rotation as R

# --- Constants and Configuration ---

# File Paths (Update if necessary)
DATA_DIR = "data"
EXP1_CSV = os.path.join(DATA_DIR, "waypoints_1.csv")
EXP1_YAML = os.path.join(DATA_DIR, "environment_config_1.yaml")
EXP2_CSV = os.path.join(DATA_DIR, "waypoints_2.csv")
EXP2_YAML = os.path.join(DATA_DIR, "environment_config_2.yaml")
OUTPUT_YAML_FILE = "interface_relative_poses_flow.yaml" # Changed output name slightly

# Interface Names (in expected order of segmentation)
INTERFACE_NAMES = [
    "button_2", "button_3", "button_4", "button_5_and_6", "switch_0",
    "switch_1", "knob_2_and_3", "lever_4_7", "lever_8_11", "knob_12",
    "switch_13", "knob_14_18"
]

# Filtering Parameter
MIN_WAYPOINTS_PER_INTERFACE = 3

# DH Parameters & TCP Transform (copied from previous script)
DH_PARAMS = [
    [0.1625,  0,        math.pi/2], [0, -0.425, 0], [0, -0.3922, 0],
    [0.1333,  0,        math.pi/2], [0.0997, 0, -math.pi/2], [0.0996, 0, 0]
]
T_6_TCP = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1565], [0, 0, 0, 1]
])

# --- Helper Functions (FK, YAML read, transforms - identical) ---

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


# --- Main Data Processing Function (identical) ---

def process_experiment_data(csv_filepath, yaml_filepath, dh_params, t_6_tcp):
    # (Same implementation as the previous script)
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


# --- Main Execution ---

if __name__ == "__main__":

    exp1_data = process_experiment_data(EXP1_CSV, EXP1_YAML, DH_PARAMS, T_6_TCP)
    exp2_data = process_experiment_data(EXP2_CSV, EXP2_YAML, DH_PARAMS, T_6_TCP)

    if exp1_data is None or exp2_data is None:
        print("\nAborting due to errors during data processing.")
        exit()

    all_points_by_interface_name = {}
    num_interfaces = min(len(exp1_data), len(exp2_data), len(INTERFACE_NAMES))
    if len(exp1_data) != len(exp2_data): print(f"\nWarning: Mismatched block counts ({len(exp1_data)} vs {len(exp2_data)}). Processing {num_interfaces}.")
    if num_interfaces < len(INTERFACE_NAMES): print(f"\nWarning: Found {num_interfaces} blocks, but {len(INTERFACE_NAMES)} names provided. Using first {num_interfaces}.")

    for i in range(num_interfaces):
        name = INTERFACE_NAMES[i]
        if i in exp1_data and i in exp2_data:
            points1 = np.array(exp1_data[i])
            points2 = np.array(exp2_data[i])
            all_points_by_interface_name[name] = np.vstack((points1, points2))
        else:
            print(f"Skipping interface index {i} as it wasn't found in both experiments' processed data.")

    if not all_points_by_interface_name:
        print("\nNo valid interface data found after combining experiments. Cannot save YAML.")
        exit()

    relative_poses_numpy = {} # Store numpy arrays first
    print("\nCalculating mean positions and creating relative transforms (T_aruco_interface)...")
    for name, all_points in all_points_by_interface_name.items():
        if all_points.shape[0] > 0:
            P_aruco_interface_mean = np.mean(all_points, axis=0)
            T_aruco_interface = np.identity(4)
            T_aruco_interface[0:3, 3] = P_aruco_interface_mean
            relative_poses_numpy[name] = T_aruco_interface # Store numpy array
            print(f"  - {name}: Mean Rel Pos = [{P_aruco_interface_mean[0]:.4f}, {P_aruco_interface_mean[1]:.4f}, {P_aruco_interface_mean[2]:.4f}]")
        else:
            print(f"  - Warning: No points found for {name} after combining. Skipping.")

    # Prepare Data for YAML Output: Convert numpy arrays to list of lists
    yaml_output_data = {}
    for name, T_aruco_interface in relative_poses_numpy.items():
        yaml_output_data[name] = {
            'T_aruco_interface': T_aruco_interface.tolist()
        }


    # Save the calculated relative poses to a YAML file
    if yaml_output_data:
        print(f"\nSaving relative interface poses to '{OUTPUT_YAML_FILE}'...")
        try:
            with open(OUTPUT_YAML_FILE, 'w') as f:
                # <<< MODIFIED HERE >>>
                # default_flow_style=True forces the compact [[...], [...]] format
                yaml.dump(yaml_output_data, f, default_flow_style=True, sort_keys=False, indent=None) # Indent is ignored in flow style
            print("Successfully saved YAML file (using flow style for matrices).")
        except Exception as e:
            print(f"Error saving YAML file: {e}")
    else:
        print("\nNo relative poses were calculated. YAML file not saved.")

    print("\nScript finished.")