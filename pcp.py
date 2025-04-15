import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import io
import os # To check if files exist
import yaml # For YAML parsing

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

# Transformation from Robot Flange (Frame 6) to Hand-E TCP (Unchanged)
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

# --- Function to calculate UR5e TCP positions AND Gripper State (as int) from CSV file ---
# <<< MODIFIED SECTION START >>>
def calculate_ur5e_data_from_file(filepath, dh_parameters, flange_to_tcp_transform):
    """
    Reads CSV, calculates FK, and extracts GripperState (expected 0 or 1).
    Returns a list of tuples: [(tcp_position, gripper_state_int), ...].
    Returns None on critical error (file not found, missing columns).
    Handles non-integer GripperState values by assigning -1.
    """
    if not os.path.exists(filepath):
        print(f"Error: UR5e CSV file not found at '{filepath}'")
        return None
    try:
        # Consider specifying dtype={'GripperState': 'object'} if mixed types are possible
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading UR5e CSV file '{filepath}': {e}")
        return None

    # Check for required columns
    required_cols = ['joint1pose', 'joint2pose', 'joint3pose', 'joint4pose', 'joint5pose', 'joint6pose', 'GripperState']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing required columns in '{os.path.basename(filepath)}': {', '.join(missing_cols)}")
        return None

    print(f"Calculating UR5e TCP Positions and reading Gripper State from '{os.path.basename(filepath)}':")
    print("-" * (40+20))
    positions_data_list = []
    for index, row in data.iterrows():
        try:
            joint_angles = [
                row['joint1pose'], row['joint2pose'], row['joint3pose'],
                row['joint4pose'], row['joint5pose'], row['joint6pose']
            ]

            # Get gripper state, attempt conversion to integer (expecting 0 or 1)
            raw_state = row['GripperState']
            try:
                # Pandas might already read it as int/float if column is purely numeric
                gripper_state = int(float(raw_state)) # Convert to float first, then int to handle "1.0" etc.
                if gripper_state not in [0, 1]:
                    print(f"Warning: Row {index} has unexpected numeric GripperState '{raw_state}'. Treating as 'Other'.")
                    gripper_state = -1 # Assign a distinct value for 'other' numeric state
            except (ValueError, TypeError):
                # Handle cases where conversion fails (e.g., empty string, text)
                print(f"Warning: Row {index} has non-numeric GripperState '{raw_state}'. Treating as 'Other'.")
                gripper_state = -1 # Assign a distinct value for 'other' non-numeric state

            T_0_6 = np.identity(4)
            for i in range(len(dh_parameters)):
                theta = joint_angles[i]; d, a, alpha = dh_parameters[i]
                A_i = dh_transformation_matrix(theta, d, a, alpha)
                T_0_6 = T_0_6 @ A_i
            T_0_TCP = T_0_6 @ flange_to_tcp_transform
            tcp_position = T_0_TCP[0:3, 3]

            # Append tuple of (position_array, gripper_state_integer)
            positions_data_list.append((tcp_position, gripper_state))

        except KeyError as e: print(f"Error processing row {index} in '{os.path.basename(filepath)}': Missing column {e}. Skipping row.")
        except Exception as e: print(f"Error processing row {index} (Waypoint {row.get('waypoint_id', 'N/A')}) in '{os.path.basename(filepath)}': {e}. Skipping row.")

    print(f"Finished processing {len(positions_data_list)} waypoints.")
    print("-" * (40+20))
    return positions_data_list
# <<< MODIFIED SECTION END >>>


# --- Function to read device position from a YAML file (Unchanged) ---
def read_device_position_from_yaml_file(filepath):
    """
    Reads a YAML file containing device pose information and extracts
    the position (X, Y, Z) from the 'aruco_device.pose.position' key.
    """
    if not os.path.exists(filepath):
        print(f"Error: Device position YAML file not found at '{filepath}'")
        return None
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f) # Use safe_load for security

        # Navigate the dictionary structure to get the position list
        position_list = data['aruco_device']['pose']['position']

        # Validate the position data
        if not isinstance(position_list, list) or len(position_list) != 3:
            print(f"Error: Position data in '{filepath}' is not a list of 3 elements.")
            print(f"Found: {position_list}")
            return None

        # Convert to numpy array
        # Handle potential single-element lists within the main list like [[x], [y], [z]]
        if all(isinstance(sublist, list) and len(sublist) == 1 for sublist in position_list):
             position = np.array([item[0] for item in position_list], dtype=float)
        # Handle the standard list like [x, y, z]
        elif all(isinstance(item, (int, float)) for item in position_list):
             position = np.array(position_list, dtype=float)
        else:
             print(f"Error: Unexpected format for position data in '{filepath}'. Expected [x, y, z] or [[x], [y], [z]].")
             print(f"Found: {position_list}")
             return None

        # Ensure we have exactly 3 position values
        if position.shape != (3,):
             print(f"Error: Extracted position from '{filepath}' does not have 3 elements.")
             print(f"Shape: {position.shape}, Data: {position}")
             return None

        print(f"Device Position read from '{os.path.basename(filepath)}': [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
        return position

    except FileNotFoundError:
        print(f"Error: Device position YAML file not found at '{filepath}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{filepath}': {e}")
        return None
    except KeyError as e:
        print(f"Error accessing key {e} in YAML file '{filepath}'. Check the file structure.")
        return None
    except TypeError as e:
        print(f"Error: Unexpected data type encountered while processing YAML file '{filepath}': {e}")
        return None
    except ValueError as e:
        print(f"Error converting position value to number in '{filepath}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred reading device position file '{filepath}': {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # --- Define file paths ---

    # !!! UPDATE THESE PATHS TO MATCH YOUR SYSTEM !!!
    ur5e_csv_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/config/waypoints_2.csv" # Example Linux path
    # Update the path to the YAML file containing the device position
    device_yaml_filepath = r"/home/kadi/projects/merlin_ws/src/airbus_project/demo_manager/config/environment_config.yaml" # <<< EXAMPLE YAML FILE PATH - CHANGE THIS

    # --- Check if plotting library is available ---
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        plotting_available = True
        print(f"Using Matplotlib version: {matplotlib.__version__}")
        # Make sure required libraries are installed: pip install matplotlib pandas numpy PyYAML
    except ImportError:
        print("\nWarning: Matplotlib or dependencies not found. Cannot generate plot.")
        print("Install required libraries using: pip install matplotlib pandas numpy PyYAML")
        plotting_available = False
        ur5e_data_list = None # Changed variable name
        device_position = None

    if plotting_available:
        # --- Calculate UR5e Waypoint Positions and Gripper States ---
        ur5e_data_list = calculate_ur5e_data_from_file(ur5e_csv_filepath, dh_params, T_6_TCP)

        # --- Read Device Position from its dedicated YAML file ---
        device_position = read_device_position_from_yaml_file(device_yaml_filepath)

        # --- Plotting (if data is valid) ---
        if ur5e_data_list is not None and device_position is not None:
            print("\nGenerating Interactive 3D Scatter Plot with Gripper State...")

            # --- Separate UR5e data based on gripper state (0 or 1) ---
            # <<< MODIFIED SECTION START >>>
            ur5e_open_points = []   # State == 1
            ur5e_closed_points = [] # State == 0
            ur5e_other_points = []  # State != 0 and State != 1

            for position, state in ur5e_data_list: # state is now integer (0, 1, or -1/other)
                if state == 1: # Gripper Open
                    ur5e_open_points.append(position)
                elif state == 0: # Gripper Closed
                    ur5e_closed_points.append(position)
                else: # Handle -1 or any other unexpected state
                    ur5e_other_points.append(position)
            # <<< MODIFIED SECTION END >>>

            # --- Extract coordinates for each state ---
            open_x, open_y, open_z = zip(*ur5e_open_points) if ur5e_open_points else ([], [], [])
            closed_x, closed_y, closed_z = zip(*ur5e_closed_points) if ur5e_closed_points else ([], [], [])
            other_x, other_y, other_z = zip(*ur5e_other_points) if ur5e_other_points else ([], [], [])

            # Device position coordinates (already extracted)
            dev_x, dev_y, dev_z = device_position

            # --- Create 3D Plot ---
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            # --- Plot points for each gripper state with different colors/labels ---
            scatter_artists = {} # Dictionary to store scatter plot objects for the pick handler

            if open_x:
                # Green 'o' for Gripper Open (State 1)
                sc_open = ax.scatter(open_x, open_y, open_z, marker='o', s=50, color='green', label='UR5e Waypoint (Gripper Open, State=1)', alpha=0.7, depthshade=True, picker=True)
                scatter_artists[sc_open] = {'label': 'UR5e Waypoint (Gripper Open, State=1)', 'data': ur5e_open_points}
            if closed_x:
                 # Red 'o' for Gripper Closed (State 0)
                sc_closed = ax.scatter(closed_x, closed_y, closed_z, marker='o', s=50, color='red', label='UR5e Waypoint (Gripper Closed, State=0)', alpha=0.7, depthshade=True, picker=True)
                scatter_artists[sc_closed] = {'label': 'UR5e Waypoint (Gripper Closed, State=0)', 'data': ur5e_closed_points}
            if other_x:
                # Gray 'x' for Other/Unknown states
                sc_other = ax.scatter(other_x, other_y, other_z, marker='x', s=40, color='gray', label='UR5e Waypoint (Gripper Other/Unknown)', alpha=0.6, depthshade=True, picker=True)
                scatter_artists[sc_other] = {'label': 'UR5e Waypoint (Gripper Other/Unknown)', 'data': ur5e_other_points}

            # Plot the single device position point
            sc_dev = ax.scatter(dev_x, dev_y, dev_z, marker='^', s=150, color='magenta', label='Device Position', depthshade=False, edgecolor='black', picker=True)
            scatter_artists[sc_dev] = {'label': 'Device Position', 'data': [device_position]} # Store as list of one

            # --- Configure Plot Appearance ---
            ax.set_xlabel("X coordinate (m)")
            ax.set_ylabel("Y coordinate (m)")
            ax.set_zlabel("Z coordinate (m)")
            ax.set_title("UR5e Waypoints (by Gripper State) and Device Position")

            # --- Set Equal Axis Scaling (using all points) ---
            all_x = list(open_x) + list(closed_x) + list(other_x) + [dev_x]
            all_y = list(open_y) + list(closed_y) + list(other_y) + [dev_y]
            all_z = list(open_z) + list(closed_z) + list(other_z) + [dev_z]

            if all_x:
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

            # --- Define the Callback Function for Clicking (Unchanged logic, relies on labels) ---
            def on_pick(event):
                """Callback function to handle pick events on scatter points."""
                ind = event.ind
                if not ind: return True

                point_index_in_artist = ind[0]
                artist = event.artist
                artist_info = scatter_artists.get(artist)

                if artist_info:
                    label = artist_info['label']
                    data_list = artist_info['data']
                    try:
                        position = data_list[point_index_in_artist]
                        x, y, z = position
                        print(f"Clicked {label} (Point index within category: {point_index_in_artist}): X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
                    except IndexError:
                        print(f"Index error for clicked point on artist '{label}'. Index: {point_index_in_artist}, Data size: {len(data_list)}")
                    except Exception as e:
                        print(f"Error processing click on '{label}': {e}")
                else:
                    print(f"Clicked an unrecognized plot element: {artist.get_label()}")
                return True

            # --- Connect Callback and Add Legend/Grid ---
            ax.legend() # Legend now includes the (State=0/1) clarification
            ax.grid(True)
            fig.canvas.mpl_connect('pick_event', on_pick)

            # --- Show Plot ---
            print("\nPlot window opened. Click on points to print their coordinates to the console.")
            plt.show()

        # Handle cases where data reading failed
        elif ur5e_data_list is None:
             print("\nPlotting skipped: Failed to calculate UR5e positions/state.")
        elif device_position is None:
             print("\nPlotting skipped: Failed to read device position from YAML.")

    # Fallback if plotting is not available
    elif not plotting_available:
        print("Plotting skipped as required libraries are not installed.")
