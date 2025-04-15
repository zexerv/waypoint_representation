# visualize_tfs.py

import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Important for 3D projection
import os

# --- Configuration ---

# Input YAML files
RELATIVE_POSES_YAML = "interface_relative_poses_flow.yaml" # Or the block-style YAML name
ARUCO_POSE_YAML_EXP1 = os.path.join("data", "environment_config_1.yaml")
ARUCO_POSE_YAML_EXP2 = os.path.join("data", "environment_config_2.yaml")

# Plotting Parameters
AXIS_LENGTH = 0.05  # Length of plotted coordinate axes in meters
LABEL_OFFSET = AXIS_LENGTH * 0.1 # Offset for text labels from origin

COLORS = {
    'exp1': {'aruco': 'blue', 'interface': 'cornflowerblue'},
    'exp2': {'aruco': 'darkorange', 'interface': 'sandybrown'}
}
LINESTYLES = { # Optional: differentiate line styles
    'aruco': '-',
    'interface': '--'
}
LINEWIDTHS = {
    'aruco': 1.5,
    'interface': 1.0
}

# --- Helper Functions ---

def read_yaml_file(filepath):
    """Loads data from a YAML file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"Error reading or parsing YAML file '{filepath}': {e}")
        return None

def plot_frame(ax, T, label, length=0.1, colors=None, linestyle='-', linewidth=1.0, label_offset=0.0):
    """
    Plots a 3D coordinate frame represented by a 4x4 transformation matrix T.

    Args:
        ax: Matplotlib 3D axes object.
        T: 4x4 homogeneous transformation matrix (numpy array).
        label (str): Text label for the frame.
        length (float): Length of the axes to draw.
        colors (list/tuple, optional): Colors for X, Y, Z axes (e.g., ['r', 'g', 'b']).
                                      If None, defaults to RGB.
        linestyle (str): Linestyle for axes.
        linewidth (float): Linewidth for axes.
        label_offset (float): Small offset for the text label from origin.
    """
    if colors is None:
        colors = ['r', 'g', 'b'] # Default: X=Red, Y=Green, Z=Blue
    if len(colors) != 3:
        print("Warning: plot_frame requires 3 colors for XYZ. Using defaults.")
        colors = ['r', 'g', 'b']

    # Origin
    origin = T[0:3, 3]

    # Axis endpoints in base frame
    x_axis = T @ np.array([length, 0, 0, 1])
    y_axis = T @ np.array([0, length, 0, 1])
    z_axis = T @ np.array([0, 0, length, 1])

    # Plot axes
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
            color=colors[0], linestyle=linestyle, linewidth=linewidth) # X
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
            color=colors[1], linestyle=linestyle, linewidth=linewidth) # Y
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
            color=colors[2], linestyle=linestyle, linewidth=linewidth) # Z

    # Plot label slightly offset from origin
    ax.text(origin[0] + label_offset, origin[1] + label_offset, origin[2] + label_offset,
            label, color='k', fontsize=8) # Black text label


def get_transform_from_yaml_dict(data_dict, key='T_aruco_interface'):
    """Safely extracts and converts matrix list from loaded YAML data."""
    if key in data_dict:
        matrix_list = data_dict[key]
        try:
            matrix = np.array(matrix_list)
            if matrix.shape == (4, 4):
                return matrix
            else:
                print(f"Warning: Matrix under key '{key}' has incorrect shape {matrix.shape}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not convert data under key '{key}' to numpy array: {e}. Skipping.")
    else:
        print(f"Warning: Key '{key}' not found in dictionary. Skipping.")
    return None

# --- Main Execution ---
if __name__ == "__main__":

    # 1. Load Relative Interface Poses
    print(f"Loading relative interface poses from: {RELATIVE_POSES_YAML}")
    relative_poses_data = read_yaml_file(RELATIVE_POSES_YAML)
    if relative_poses_data is None:
        exit()

    # 2. Load ArUco Poses for both experiments
    print(f"Loading ArUco pose for Exp 1 from: {ARUCO_POSE_YAML_EXP1}")
    aruco_data_exp1 = read_yaml_file(ARUCO_POSE_YAML_EXP1)
    if aruco_data_exp1 is None:
        exit()
    # Extract T_base_aruco for Exp1 (reuse helper from save_yaml or extract manually)
    if 'aruco_device' in aruco_data_exp1 and 'matrix' in aruco_data_exp1['aruco_device']:
         T_base_aruco_1 = np.array(aruco_data_exp1['aruco_device']['matrix'])
         if T_base_aruco_1.shape != (4,4):
             print("Error: ArUco matrix for Exp 1 has wrong shape.")
             exit()
    else:
         print("Error: Could not find ArUco matrix in Exp 1 YAML.")
         exit()


    print(f"Loading ArUco pose for Exp 2 from: {ARUCO_POSE_YAML_EXP2}")
    aruco_data_exp2 = read_yaml_file(ARUCO_POSE_YAML_EXP2)
    if aruco_data_exp2 is None:
        exit()
    # Extract T_base_aruco for Exp2
    if 'aruco_device' in aruco_data_exp2 and 'matrix' in aruco_data_exp2['aruco_device']:
         T_base_aruco_2 = np.array(aruco_data_exp2['aruco_device']['matrix'])
         if T_base_aruco_2.shape != (4,4):
             print("Error: ArUco matrix for Exp 2 has wrong shape.")
             exit()
    else:
         print("Error: Could not find ArUco matrix in Exp 2 YAML.")
         exit()


    # 3. Setup 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    print("\nCalculating absolute poses and plotting frames...")

    # Keep track of min/max coordinates for setting plot limits
    all_origins = []

    # 4. Plot ArUco Frames
    print("Plotting ArUco Frames...")
    plot_frame(ax, T_base_aruco_1, "ArUco Exp1", length=AXIS_LENGTH, colors=[COLORS['exp1']['aruco']]*3, linestyle=LINESTYLES['aruco'], linewidth=LINEWIDTHS['aruco'], label_offset=LABEL_OFFSET)
    plot_frame(ax, T_base_aruco_2, "ArUco Exp2", length=AXIS_LENGTH, colors=[COLORS['exp2']['aruco']]*3, linestyle=LINESTYLES['aruco'], linewidth=LINEWIDTHS['aruco'], label_offset=LABEL_OFFSET)
    all_origins.append(T_base_aruco_1[0:3, 3])
    all_origins.append(T_base_aruco_2[0:3, 3])


    # 5. Calculate and Plot Interface Frames for both Experiments
    print("Plotting Interface Frames...")
    for interface_name, data_dict in relative_poses_data.items():
        T_aruco_interface = get_transform_from_yaml_dict(data_dict, key='T_aruco_interface')

        if T_aruco_interface is not None:
            # Experiment 1
            T_base_interface_1 = T_base_aruco_1 @ T_aruco_interface
            plot_frame(ax, T_base_interface_1, f"{interface_name} Exp1", length=AXIS_LENGTH*0.8, colors=[COLORS['exp1']['interface']]*3, linestyle=LINESTYLES['interface'], linewidth=LINEWIDTHS['interface'], label_offset=LABEL_OFFSET*0.8)
            all_origins.append(T_base_interface_1[0:3, 3])

            # Experiment 2
            T_base_interface_2 = T_base_aruco_2 @ T_aruco_interface
            plot_frame(ax, T_base_interface_2, f"{interface_name} Exp2", length=AXIS_LENGTH*0.8, colors=[COLORS['exp2']['interface']]*3, linestyle=LINESTYLES['interface'], linewidth=LINEWIDTHS['interface'], label_offset=LABEL_OFFSET*0.8)
            all_origins.append(T_base_interface_2[0:3, 3])
        else:
            print(f"Skipping interface {interface_name} due to error reading relative transform.")

    # 6. Configure and Show Plot
    ax.set_xlabel("X (Base Frame) [m]")
    ax.set_ylabel("Y (Base Frame) [m]")
    ax.set_zlabel("Z (Base Frame) [m]")
    ax.set_title("ArUco and Interface Poses in Robot Base Frame (Exp1 vs Exp2)")

    # Set plot limits based on plotted origins - ensures visibility
    if all_origins:
        origins_array = np.array(all_origins)
        max_vals = np.max(origins_array, axis=0)
        min_vals = np.min(origins_array, axis=0)
        mid_point = (max_vals + min_vals) / 2
        max_range = np.max(max_vals - min_vals) + 2 * AXIS_LENGTH # Add axis length buffer
        if max_range < 0.1: max_range = 0.5 # Ensure a minimum visible range

        ax.set_xlim(mid_point[0] - max_range / 2, mid_point[0] + max_range / 2)
        ax.set_ylim(mid_point[1] - max_range / 2, mid_point[1] + max_range / 2)
        ax.set_zlim(mid_point[2] - max_range / 2, mid_point[2] + max_range / 2)
        print(f"Setting axis limits around data range: {max_range:.2f}m")
    else:
        # Default limits if nothing was plotted
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.2, 0.8)

    # Improve layout and show
    ax.grid(True, linestyle=':')
    # Legend might be too crowded, relying on text labels from plot_frame
    # fig.legend() # Uncomment if you want to try a legend
    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()

    print("\nScript finished.")