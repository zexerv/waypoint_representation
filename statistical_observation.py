# statistical_observation.py

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import os
from scipy.spatial.transform import Rotation as R # For quaternion/matrix conversion

# --- Constants and Configuration ---

# File Paths (Update if necessary)
DATA_DIR = "data"
EXP1_CSV = os.path.join(DATA_DIR, "waypoints_1.csv")
EXP1_YAML = os.path.join(DATA_DIR, "environment_config_1.yaml")
EXP2_CSV = os.path.join(DATA_DIR, "waypoints_2.csv")
EXP2_YAML = os.path.join(DATA_DIR, "environment_config_2.yaml")

# Interface Names (in expected order of segmentation)
INTERFACE_NAMES = [
    "button_2",
    "button_3",
    "button_4",
    "button_5_and_6",
    "switch_0",
    "switch_1",
    "knob_2_and_3",
    "lever_4_7",
    "lever_8_11",
    "knob_12",
    "switch_13",
    "knob_14_18"
]

# Filtering Parameter
MIN_WAYPOINTS_PER_INTERFACE = 3

# DH Parameters for UR5e (copied from previous script)
DH_PARAMS = [
    # d [m],   a [m],    alpha [rad]
    [0.1625,  0,        math.pi/2],   # Joint 1 -> Link 1
    [0,       -0.425,   0],           # Joint 2 -> Link 2
    [0,       -0.3922,  0],           # Joint 3 -> Link 3
    [0.1333,  0,        math.pi/2],   # Joint 4 -> Link 4
    [0.0997,  0,       -math.pi/2],   # Joint 5 -> Link 5
    [0.0996,  0,        0]            # Joint 6 -> Link 6
]

# Flange to TCP Transformation (copied from previous script)
# IMPORTANT: Assumes this TCP is accurate for the probing point!
T_6_TCP = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  0],
    [0,  0,  1,  0.1565], # Z translation (m) to the TCP
    [0,  0,  0,  1]
])

# --- Helper Functions ---

def dh_transformation_matrix(theta, d, a, alpha):
    """Calculates the homogeneous transformation matrix A_i for a joint."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    A = np.array([
        [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
        [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0,         sin_alpha,             cos_alpha,            d            ],
        [0,         0,                     0,                    1            ]
    ])
    return A

def calculate_fk_position(joint_angles, dh_parameters, flange_to_tcp_transform):
    """Calculates the TCP position in the base frame."""
    if len(joint_angles) != len(dh_parameters):
        raise ValueError("Number of joint angles must match number of DH parameters.")
    T_0_6 = np.identity(4)
    for i in range(len(dh_parameters)):
        theta = joint_angles[i]
        d, a, alpha = dh_parameters[i]
        A_i = dh_transformation_matrix(theta, d, a, alpha)
        T_0_6 = T_0_6 @ A_i
    T_0_TCP = T_0_6 @ flange_to_tcp_transform
    tcp_position = T_0_TCP[0:3, 3] # Extract position vector
    return tcp_position

def read_aruco_transform_from_yaml(filepath):
    """Reads the 4x4 transformation matrix T_base_aruco from the YAML file."""
    if not os.path.exists(filepath):
        print(f"Error: ArUco config file not found at '{filepath}'")
        return None
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        # Extract matrix directly if available
        if 'aruco_device' in data and 'matrix' in data['aruco_device']:
            matrix_list = data['aruco_device']['matrix']
            T_base_aruco = np.array(matrix_list)
            if T_base_aruco.shape == (4, 4):
                 print(f"Read T_base_aruco matrix from '{os.path.basename(filepath)}'")
                 return T_base_aruco
            else:
                 print(f"Error: Matrix in '{filepath}' does not have shape (4, 4). Found {T_base_aruco.shape}")
                 return None
        # Fallback: construct from position/quaternion if matrix is missing
        elif 'aruco_device' in data and 'pose' in data['aruco_device']:
            pose = data['aruco_device']['pose']
            pos_list = pose['position']
            quat_list = pose['quaternion'] # Assuming [x, y, z, w] order based on common ROS/scipy conventions
             # Handle potential list wrapping [[x],[y],[z]] vs [x,y,z]
            if all(isinstance(sub, list) and len(sub) == 1 for sub in pos_list):
                position = np.array([p[0] for p in pos_list])
            else:
                 position = np.array(pos_list)

            if position.shape != (3,) or len(quat_list) != 4:
                 print(f"Error: Invalid position/quaternion format in '{filepath}'")
                 return None

            # Convert quaternion [x, y, z, w] to rotation matrix
            # Note: Scipy expects [x, y, z, w]
            rotation = R.from_quat(quat_list)
            rotation_matrix = rotation.as_matrix()

            T_base_aruco = np.identity(4)
            T_base_aruco[0:3, 0:3] = rotation_matrix
            T_base_aruco[0:3, 3] = position
            print(f"Constructed T_base_aruco from pose in '{os.path.basename(filepath)}'")
            return T_base_aruco
        else:
            print(f"Error: Could not find 'matrix' or 'pose' under 'aruco_device' in '{filepath}'")
            return None
    except Exception as e:
        print(f"Error reading or parsing YAML file '{filepath}': {e}")
        return None

def invert_transform(T):
    """Computes the inverse of a 4x4 homogeneous transformation matrix."""
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.identity(4)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    return T_inv

def transform_point(T, p):
    """Applies a 4x4 transformation matrix T to a 3D point p."""
    p_homog = np.append(p, 1.0) # Convert to homogeneous coordinates
    p_transformed_homog = T @ p_homog
    return p_transformed_homog[0:3] # Return non-homogeneous 3D point


# --- Main Data Processing Function ---

def process_experiment_data(csv_filepath, yaml_filepath, dh_params, t_6_tcp):
    """
    Processes one experiment's data: reads CSV, segments by gripper state,
    calculates FK, transforms points to ArUco frame.

    Returns:
        dict: {interface_index (int): [list of points in ArUco frame (np.array)]}
              Returns None if critical errors occur.
    """
    print(f"\n--- Processing Experiment: {os.path.basename(csv_filepath)} ---")
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at '{csv_filepath}'")
        return None

    # 1. Read ArUco Pose and calculate inverse transform
    T_base_aruco = read_aruco_transform_from_yaml(yaml_filepath)
    if T_base_aruco is None:
        return None # Error already printed
    try:
        T_aruco_base = invert_transform(T_base_aruco)
        print("Calculated T_aruco_base (inverse transform)")
    except Exception as e:
        print(f"Error inverting ArUco transform: {e}")
        return None

    # 2. Read CSV data
    try:
        data = pd.read_csv(csv_filepath)
        # Check required columns
        required_cols = ['joint1pose', 'joint2pose', 'joint3pose', 'joint4pose', 'joint5pose', 'joint6pose', 'GripperState']
        if not all(col in data.columns for col in required_cols):
            print(f"Error: Missing one or more required columns in {csv_filepath}")
            print(f"Required: {required_cols}")
            print(f"Found: {list(data.columns)}")
            return None
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return None

    # 3. Iterate, Segment, Calculate FK, and Transform
    processed_interfaces = {}
    current_block_points_aruco = []
    interface_index_counter = 0
    is_gripper_closed = False # Start assuming gripper is open or state is unknown
    ignore_next_point = False # Flag to skip the first point after closing
    last_gripper_state = -1 # Initialize with invalid state

    print("Segmenting data based on GripperState (0=Open, 1=Closed)...")
    for index, row in data.iterrows():
        try:
            current_gripper_state = int(float(row['GripperState'])) # Read state (0 or 1)
            if current_gripper_state not in [0, 1]:
                 print(f"Warning: Row {index} has unexpected GripperState {current_gripper_state}. Treating as previous state.")
                 current_gripper_state = last_gripper_state # Keep previous state logic
                 if last_gripper_state == -1: continue # Skip if very first state is invalid

            # --- State Machine Logic ---
            # Transition: Open -> Closed (0 -> 1)
            if not is_gripper_closed and current_gripper_state == 1:
                # print(f"Row {index}: Gripper Closed (1). Starting potential new block.")
                is_gripper_closed = True
                current_block_points_aruco = [] # Reset block points
                ignore_next_point = True # Skip the next point (movement phase)

            # Transition: Closed -> Open (1 -> 0)
            elif is_gripper_closed and current_gripper_state == 0:
                # print(f"Row {index}: Gripper Opened (0). Finishing block.")
                is_gripper_closed = False
                ignore_next_point = False
                # Check if the completed block is valid
                if len(current_block_points_aruco) >= MIN_WAYPOINTS_PER_INTERFACE:
                    print(f"  -> Storing Interface Block {interface_index_counter} with {len(current_block_points_aruco)} points.")
                    processed_interfaces[interface_index_counter] = current_block_points_aruco
                    interface_index_counter += 1
                elif len(current_block_points_aruco) > 0: # Discard if too short but not empty
                    print(f"  -> Discarding block with {len(current_block_points_aruco)} points (min required: {MIN_WAYPOINTS_PER_INTERFACE}).")
                # Reset block (already done when closing, but good practice)
                current_block_points_aruco = []

            # State: Gripper is Closed (1) - Collecting Data
            elif is_gripper_closed and current_gripper_state == 1:
                if ignore_next_point:
                    # print(f"Row {index}: Ignoring first point after closing gripper.")
                    ignore_next_point = False # Only ignore the very first one
                else:
                    # This is a valid measurement point for the current block
                    joint_angles = [
                        row['joint1pose'], row['joint2pose'], row['joint3pose'],
                        row['joint4pose'], row['joint5pose'], row['joint6pose']
                    ]
                    # Calculate FK
                    P_base_tcp = calculate_fk_position(joint_angles, dh_params, t_6_tcp)
                    # Transform to ArUco frame
                    P_aruco_tcp = transform_point(T_aruco_base, P_base_tcp)
                    # Add to current block
                    current_block_points_aruco.append(P_aruco_tcp)
                    # print(f"Row {index}: Added point to current block. N points = {len(current_block_points_aruco)}")

            # State: Gripper is Open (0) - Waiting
            elif not is_gripper_closed and current_gripper_state == 0:
                # Do nothing, just waiting for gripper to close
                pass

            last_gripper_state = current_gripper_state

        except KeyError as e: print(f"Error processing row {index} in '{os.path.basename(csv_filepath)}': Missing column {e}. Skipping row.")
        except Exception as e: print(f"Error processing row {index}: {e}. Skipping row.")

    # Handle end of file: Check if the last block was valid
    if is_gripper_closed and len(current_block_points_aruco) >= MIN_WAYPOINTS_PER_INTERFACE:
         print(f"  -> Storing final Interface Block {interface_index_counter} with {len(current_block_points_aruco)} points (end of file).")
         processed_interfaces[interface_index_counter] = current_block_points_aruco
         interface_index_counter += 1
    elif is_gripper_closed and len(current_block_points_aruco) > 0:
         print(f"  -> Discarding final block with {len(current_block_points_aruco)} points (min required: {MIN_WAYPOINTS_PER_INTERFACE}, end of file).")


    print(f"--- Finished Processing. Found {len(processed_interfaces)} valid interface blocks. ---")
    return processed_interfaces


# --- Main Execution ---

if __name__ == "__main__":

    # Process both experiments
    exp1_data = process_experiment_data(EXP1_CSV, EXP1_YAML, DH_PARAMS, T_6_TCP)
    exp2_data = process_experiment_data(EXP2_CSV, EXP2_YAML, DH_PARAMS, T_6_TCP)

    if exp1_data is None or exp2_data is None:
        print("\nAborting due to errors during data processing.")
        exit()

    # Combine data by interface name
    combined_data = {}
    num_interfaces = min(len(exp1_data), len(exp2_data), len(INTERFACE_NAMES))
    if len(exp1_data) != len(exp2_data):
         print(f"\nWarning: Experiments resulted in different numbers of valid interface blocks ({len(exp1_data)} vs {len(exp2_data)}).")
         print(f"         Will only plot the first {num_interfaces} interfaces using names from INTERFACE_NAMES.")
    elif num_interfaces < len(INTERFACE_NAMES):
         print(f"\nWarning: Found {num_interfaces} valid interface blocks, but {len(INTERFACE_NAMES)} names were provided.")
         print(f"         Using the first {num_interfaces} names.")


    valid_interface_names = []
    for i in range(num_interfaces):
        name = INTERFACE_NAMES[i]
        valid_interface_names.append(name)
        combined_data[name] = {
            'exp1': np.array(exp1_data[i]), # Convert list of points to numpy array (N x 3)
            'exp2': np.array(exp2_data[i])
        }
        # print(f"Combined data for '{name}': Exp1={len(combined_data[name]['exp1'])}, Exp2={len(combined_data[name]['exp2'])}")


    if not combined_data:
        print("\nNo valid interface data found or combined. Cannot generate plots.")
        exit()

    # --- Plotting ---
    print("\nGenerating plots...")
    num_plot_interfaces = len(valid_interface_names)
    interface_indices = np.arange(num_plot_interfaces) # X-axis positions for categories

    coords = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(3, 1, figsize=(max(10, num_plot_interfaces * 1.5), 12), sharex=True) # Adjust width based on #interfaces

    colors = {'exp1': 'blue', 'exp2': 'orange', 'mean1': 'darkblue', 'mean2': 'darkorange', 'mean_all': 'red'}
    markers = {'exp1': '.', 'exp2': '.'}
    mean_markers = {'mean1': '_', 'mean2': '_', 'mean_all': '_'}
    mean_linewidth = 2
    jitter = 0.1 # Offset points within a category to avoid overlap

    for i, coord_name in enumerate(coords):
        ax = axes[i]
        all_means1, all_means2, all_means_overall = [], [], []

        for j, name in enumerate(valid_interface_names):
            data1 = combined_data[name]['exp1'][:, i] # Select X, Y or Z coordinate for exp1
            data2 = combined_data[name]['exp2'][:, i] # Select X, Y or Z coordinate for exp2
            data_all = np.concatenate((data1, data2))

            mean1 = np.mean(data1)
            mean2 = np.mean(data2)
            mean_all = np.mean(data_all)
            all_means1.append(mean1)
            all_means2.append(mean2)
            all_means_overall.append(mean_all)

            # Plot individual points with jitter
            ax.scatter(np.full(data1.shape, j - jitter), data1, color=colors['exp1'], marker=markers['exp1'], alpha=0.6, label='Exp 1 Data' if j == 0 else "")
            ax.scatter(np.full(data2.shape, j + jitter), data2, color=colors['exp2'], marker=markers['exp2'], alpha=0.6, label='Exp 2 Data' if j == 0 else "")

            # Plot means as horizontal lines across the jittered category width
            ax.plot([j - jitter*2, j - jitter], [mean1, mean1], color=colors['mean1'], marker=mean_markers['mean1'], linestyle='-', linewidth=mean_linewidth, label='Exp 1 Mean' if j == 0 else "")
            ax.plot([j + jitter, j + jitter*2], [mean2, mean2], color=colors['mean2'], marker=mean_markers['mean2'], linestyle='-', linewidth=mean_linewidth, label='Exp 2 Mean' if j == 0 else "")
            ax.plot([j - jitter*2, j + jitter*2], [mean_all, mean_all], color=colors['mean_all'], marker=mean_markers['mean_all'], linestyle='-', linewidth=mean_linewidth+1, label='Overall Mean' if j == 0 else "") # Thicker line for overall mean

        ax.set_ylabel(f'{coord_name} coordinate (ArUco Frame) [m]')
        ax.grid(True, axis='y', linestyle=':')
        ax.legend()


    # Configure overall plot
    axes[-1].set_xticks(interface_indices)
    axes[-1].set_xticklabels(valid_interface_names, rotation=45, ha='right')
    axes[-1].set_xlabel("Interface Element")
    fig.suptitle("Interface Positions Relative to ArUco Frame (Experiments 1 & 2)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

    print("Showing plot...")
    plt.show()

    print("\nScript finished.")