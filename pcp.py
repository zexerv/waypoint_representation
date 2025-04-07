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

# Transformation from Robot Flange (Frame 6) to Hand-E TCP
T_6_TCP = np.array([
    [1,  0,  0,  0],       # No change in rotation or X translation
    [0,  1,  0,  0],       # No change in rotation or Y translation
    [0,  0,  1,  0.1565],  # Z translation (m) to the TCP
    [0,  0,  0,  1]
])

# --- Function to calculate the Transformation Matrix using DH parameters (Unchanged) ---
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

# --- Function to calculate UR5e TCP positions from CSV file (Unchanged from previous correct version) ---
def calculate_ur5e_positions_from_file(filepath, dh_parameters, flange_to_tcp_transform):
    """Reads CSV, calculates FK, returns list of TCP positions."""
    if not os.path.exists(filepath):
        print(f"Error: UR5e CSV file not found at '{filepath}'")
        return None
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading UR5e CSV file '{filepath}': {e}")
        return None
    print(f"Calculating UR5e TCP Positions from '{os.path.basename(filepath)}':")
    print("-" * 40)
    positions_list = []
    for index, row in data.iterrows():
        try:
            joint_angles = [
                row['joint1pose'], row['joint2pose'], row['joint3pose'],
                row['joint4pose'], row['joint5pose'], row['joint6pose']
            ]
            T_0_6 = np.identity(4)
            for i in range(len(dh_parameters)):
                theta = joint_angles[i]; d, a, alpha = dh_parameters[i]
                A_i = dh_transformation_matrix(theta, d, a, alpha)
                T_0_6 = T_0_6 @ A_i
            T_0_TCP = T_0_6 @ flange_to_tcp_transform
            tcp_position = T_0_TCP[0:3, 3]
            positions_list.append(tcp_position)
        except KeyError as e: print(f"Error processing row {index} in '{os.path.basename(filepath)}': Missing column {e}. Skipping row.")
        except Exception as e: print(f"Error processing row {index} (Waypoint {row.get('waypoint_id', 'N/A')}) in '{os.path.basename(filepath)}': {e}. Skipping row.")
    print(f"Finished processing {len(positions_list)} waypoints.")
    print("-" * 40)
    return positions_list

# --- Function to read device position from a 4x4 matrix file ---
# (Reinstated from previous version)
def read_device_position_from_matrix_file(filepath):
    """
    Reads a text file containing a 4x4 transformation matrix and extracts
    the position (X, Y, Z) from the last column.
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
            # Read only the first 4 lines that contain enough numbers
            valid_lines_read = 0
            for line in lines:
                 if valid_lines_read >= 4: break
                 # Split by whitespace and convert to float, filter out empty strings
                 row = [float(val) for val in line.strip().split() if val]
                 if len(row) == 4: # Ensure exactly 4 values found
                     matrix_data.append(row)
                     valid_lines_read += 1
                 elif len(row) > 0 : # If line has content but not 4 numbers
                     print(f"Warning: Ignoring line in '{filepath}' with unexpected number of values: {line.strip()}")

            if valid_lines_read < 4:
                 print(f"Error: Could not find 4 valid rows with 4 values each in '{filepath}'.")
                 return None

        matrix = np.array(matrix_data)
        # Extract position (first 3 elements of the 4th column)
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
    # !!! UPDATE THESE PATHS TO MATCH YOUR SYSTEM !!!
    ur5e_csv_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/data/2025-03-13_15-03-56/waypoints.csv" # Example Linux path
    # Reinstate the path to the file containing the 4x4 matrix
    device_matrix_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/data/device_position_2025-03-13_15-39-32.txt" # Example Linux path

    # --- Check if plotting library is available ---
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        plotting_available = True
        print(f"Using Matplotlib version: {matplotlib.__version__}")
        # Make sure required libraries are installed: pip install matplotlib pandas numpy
    except ImportError:
        print("\nWarning: Matplotlib not found. Cannot generate plot.")
        print("Install it using: pip install matplotlib pandas numpy")
        plotting_available = False
        ur5e_tcp_positions_list = None
        device_position = None

    if plotting_available:
        # --- Calculate UR5e Waypoint Positions ---
        ur5e_tcp_positions_list = calculate_ur5e_positions_from_file(ur5e_csv_filepath, dh_params, T_6_TCP)

        # --- Read Device Position from its dedicated file ---
        device_position = read_device_position_from_matrix_file(device_matrix_filepath)

        # --- Plotting (if data is valid) ---
        # Check if both calculations/readings were successful
        if ur5e_tcp_positions_list is not None and device_position is not None:
            print("\nGenerating Interactive 3D Scatter Plot...")

            # Extract X, Y, Z for UR5e TCP waypoints
            ur5e_x, ur5e_y, ur5e_z = zip(*ur5e_tcp_positions_list)

            # Device position coordinates (already extracted)
            dev_x, dev_y, dev_z = device_position

            # --- Create 3D Plot ---
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot *all* UR5e waypoints as standard scatter points (make them pickable)
            ax.scatter(ur5e_x, ur5e_y, ur5e_z, marker='o', s=50, color='blue', label='UR5e Waypoint', alpha=0.7, depthshade=True, picker=True) # Added picker=True

            # Plot the single device position point with a different style (make it pickable too)
            ax.scatter(dev_x, dev_y, dev_z, marker='^', s=150, color='magenta', label='Device Position', depthshade=False, edgecolor='black', picker=True) # Added picker=True

            # --- Configure Plot Appearance ---
            ax.set_xlabel("X coordinate (m)")
            ax.set_ylabel("Y coordinate (m)")
            ax.set_zlabel("Z coordinate (m)")
            ax.set_title("UR5e Waypoints and Device Position (Click Points for Coords)")

            # --- Set Equal Axis Scaling (using all points) ---
            all_x = list(ur5e_x) + [dev_x]
            all_y = list(ur5e_y) + [dev_y]
            all_z = list(ur5e_z) + [dev_z]

            if all_x: # Check if there is data
                max_range = np.array([max(all_x) - min(all_x),
                                      max(all_y) - min(all_y),
                                      max(all_z) - min(all_z)]).max()
                if max_range < 1e-6: max_range = 1.0
                mid_x = (max(all_x) + min(all_x)) * 0.5
                mid_y = (max(all_y) + min(all_y)) * 0.5
                mid_z = (max(all_z) + min(all_z)) * 0.5
                ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
                ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
                ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
                print(f"Set axis limits to achieve equal scaling with range: {max_range:.3f}")
            else:
                 print("No data points to determine axis scaling.")

            # --- Define the Callback Function for Clicking (Revised) ---
            def on_pick(event):
                """Callback function to handle pick events on scatter points."""
                ind = event.ind
                if ind.size == 0: return True # No points picked

                index = ind[0] # Use first picked point if multiple overlap
                artist = event.artist
                artist_label = artist.get_label()

                try:
                    if artist_label == 'UR5e Waypoint':
                        # Retrieve coords from the main UR5e data lists using index
                        x, y, z = ur5e_x[index], ur5e_y[index], ur5e_z[index]
                        print(f"Clicked {artist_label} (Index {index}): X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
                    elif artist_label == 'Device Position':
                        # Retrieve coords from the specific device position variables
                        x, y, z = dev_x, dev_y, dev_z
                        # Index might be [0] here, referring to the only point in *this* scatter call
                        print(f"Clicked {artist_label}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
                    else:
                        # Handle clicks on potential other artists if needed
                        pass
                except IndexError:
                     print(f"Clicked {artist_label}, but index {index} is out of bounds for data.")
                except NameError:
                    print(f"Clicked {artist_label}, but coordinate data not found in scope.")

                return True # Indicate event was handled

            # --- Connect Callback and Add Legend/Grid ---
            ax.legend()
            ax.grid(True)
            fig.canvas.mpl_connect('pick_event', on_pick) # Connect the event handler

            # --- Show Plot ---
            print("\nPlot window opened. Click on points to print their coordinates to the console.")
            plt.show()

        # Handle cases where data reading failed
        elif ur5e_tcp_positions_list is None:
             print("\nPlotting skipped: Failed to calculate UR5e positions.")
        elif device_position is None:
             print("\nPlotting skipped: Failed to read device position.")

    # Fallback if plotting is not available
    elif not plotting_available:
        print("Plotting skipped as Matplotlib is not installed.")