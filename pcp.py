import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import io
import os # To check if files exist

# --- Denavit-Hartenberg (DH) Parameters for UR5e (Unchanged) ---
dh_params = [
    # d [m],   a [m],    alpha [rad]
    [0.1625,  0,        math.pi/2],   # Joint 1 -> Link 1
    [0,       -0.425,   0],           # Joint 2 -> Link 2
    [0,       -0.3922,  0],           # Joint 3 -> Link 3
    [0.1333,  0,        math.pi/2],   # Joint 4 -> Link 4
    [0.0997,  0,       -math.pi/2],   # Joint 5 -> Link 5
    [0.0996,  0,        0]            # Joint 6 -> Link 6
]

T_6_TCP = np.array([
    [1,  0,  0,  0],       # No change in rotation or X translation
    [0,  1,  0,  0],       # No change in rotation or Y translation
    [0,  0,  1,  0.1565],  # Z translation (m) to the TCP
    [0,  0,  0,  1]
])
# --- Function to calculate the Transformation Matrix using DH parameters (Unchanged) ---
def dh_transformation_matrix(theta, d, a, alpha):
    """
    Calculates the homogeneous transformation matrix A_i for a joint
    based on its DH parameters (standard DH convention).
    """
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

# --- Function to calculate UR5e positions from CSV file (Unchanged) ---
def calculate_ur5e_positions_from_file(filepath, dh_parameters):
    """
    Reads a CSV file containing UR5e joint poses, calculates forward kinematics
    for each row, and returns a list of end-effector positions.

    Args:
        filepath (str): Path to the CSV file.
        dh_parameters (list): A list of lists containing DH parameters [d, a, alpha] for each joint.

    Returns:
        list: A list of numpy arrays, where each array is the [X, Y, Z] position
              for a waypoint. Returns None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: UR5e CSV file not found at '{filepath}'")
        return None

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading UR5e CSV file '{filepath}': {e}")
        return None

    print(f"Calculating UR5e End-Effector Positions from '{os.path.basename(filepath)}':")
    print("-" * 40)

    positions_list = [] # Store resulting position vectors

    for index, row in data.iterrows():
        try:
            joint_angles = [
                row['joint1pose'], row['joint2pose'], row['joint3pose'],
                row['joint4pose'], row['joint5pose'], row['joint6pose']
            ]
            T_0_6 = np.identity(4)
            for i in range(len(dh_parameters)):
                theta = joint_angles[i]
                d, a, alpha = dh_parameters[i]
                A_i = dh_transformation_matrix(theta, d, a, alpha)
                T_0_6 = T_0_6 @ A_i
                T_0_TCP = T_0_6 @ T_6_TCP  
            position = T_0_TCP[0:3, 3]
            positions_list.append(position)
        except KeyError as e:
            print(f"Error processing row {index} in '{os.path.basename(filepath)}': Missing column {e}. Skipping row.")
        except Exception as e:
            print(f"Error processing row {index} (Waypoint {row.get('waypoint_id', 'N/A')}) in '{os.path.basename(filepath)}': {e}. Skipping row.")

    print(f"Finished processing {len(positions_list)} waypoints.")
    print("-" * 40)
    return positions_list

# --- Function to read device position from matrix file (Unchanged) ---
def read_device_position_from_file(filepath):
    """
    Reads a text file containing a 4x4 transformation matrix and extracts
    the position (X, Y, Z) from the last column.

    Args:
        filepath (str): Path to the text file.

    Returns:
        numpy.ndarray: The [X, Y, Z] position vector, or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: Device position file not found at '{filepath}'")
        return None

    try:
        matrix_data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 4:
                 print(f"Error: File '{filepath}' does not contain enough lines for a 4x4 matrix.")
                 return None
            for i in range(4):
                row = [float(val) for val in lines[i].strip().split() if val]
                if len(row) != 4:
                    print(f"Error: Line {i+1} in '{filepath}' does not contain 4 numeric values.")
                    return None
                matrix_data.append(row)
        matrix = np.array(matrix_data)
        position = matrix[0:3, 3]
        print(f"Device Position read from '{os.path.basename(filepath)}': [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
        return position
    except ValueError as e:
        print(f"Error parsing numeric value in '{filepath}': {e}")
        return None
    except Exception as e:
        print(f"Error reading device position file '{filepath}': {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # --- Define file paths ---
    ur5e_csv_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/data/2025-03-13_15-03-56/waypoints.csv"
    device_matrix_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/data/device_position_2025-03-13_15-39-32.txt"

    # --- Check if plotting library is available ---
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        plotting_available = True
        print(f"Using Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("\nWarning: Matplotlib not found. Cannot generate plot.")
        print("Install it using: pip install matplotlib")
        plotting_available = False

    # --- Calculate UR5e Waypoint Positions ---
    ur5e_positions_list = calculate_ur5e_positions_from_file(ur5e_csv_filepath, dh_params)

    # --- Read Device Position ---
    device_position = read_device_position_from_file(device_matrix_filepath)

    # --- Plotting (if data is valid and library available) ---
    if plotting_available and ur5e_positions_list and device_position is not None:
        print("\nGenerating 3D Scatter Plot...")

        # Extract X, Y, Z for UR5e waypoints
        if len(ur5e_positions_list) > 0:
            ur5e_x, ur5e_y, ur5e_z = zip(*ur5e_positions_list)
        else:
            ur5e_x, ur5e_y, ur5e_z = [], [], []

        # Device position coordinates
        dev_x, dev_y, dev_z = device_position

        # --- Create 3D Plot ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot UR5e waypoints as scatter points (no lines connecting them)
        if ur5e_x:
            ax.scatter(ur5e_x, ur5e_y, ur5e_z, marker='o', s=50, color='blue', label='UR5e Waypoints', alpha=0.7, depthshade=True)
        else:
            print("No UR5e waypoints to plot.")

        # Plot device position as a distinct scatter point
        ax.scatter(dev_x, dev_y, dev_z, marker='^', s=150, color='magenta', label='Device Position', depthshade=False, edgecolor='black')

        # --- Configure Plot Appearance ---
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")
        ax.set_zlabel("Z coordinate (m)")
        ax.set_title("UR5e Waypoints and Device Position (Scatter Plot)")

        # --- Set Equal Axis Scaling ---
        # Combine all points to find the maximum range
        all_x = list(ur5e_x) + [dev_x]
        all_y = list(ur5e_y) + [dev_y]
        all_z = list(ur5e_z) + [dev_z]

        if all_x: # Ensure there's data
            max_range = np.array([max(all_x) - min(all_x),
                                  max(all_y) - min(all_y),
                                  max(all_z) - min(all_z)]).max()
            # Avoid division by zero or tiny ranges if all points are coincident
            if max_range < 1e-6:
                max_range = 1.0 # Default range if points are too close

            mid_x = (max(all_x) + min(all_x)) * 0.5
            mid_y = (max(all_y) + min(all_y)) * 0.5
            mid_z = (max(all_z) + min(all_z)) * 0.5

            # Set the limits for each axis to be centered around the midpoint
            # and span the maximum range found across all axes.
            ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
            ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
            print(f"Set axis limits to achieve equal scaling with range: {max_range:.3f}")
        else:
             print("No data points to determine axis scaling.")


        ax.legend()
        ax.grid(True)

        # Show plot
        plt.show()
        print("Plot window opened.")

    elif not plotting_available:
        print("Plotting skipped as Matplotlib is not installed.")
    else:
        print("\nCould not generate plot due to errors in reading files or calculating positions.")
