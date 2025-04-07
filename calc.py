import yaml
import numpy as np
import os
import sys
# Import Rotation from scipy for quaternion calculation
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("Error: SciPy library not found. Please install it: pip install scipy")
    sys.exit(1)

# --- Configuration ---
yaml_filename = 'device_to_waypoint_transform.yaml' # Assuming the same filename
# Construct the full path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
yaml_filepath = os.path.join(script_dir, yaml_filename)

# --- Main Logic ---
try:
    # 1. Read YAML file
    if not os.path.exists(yaml_filepath):
        print(f"Error: File not found: {yaml_filepath}")
        sys.exit(1)

    with open(yaml_filepath, 'r') as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        print("Error: YAML content is not a dictionary.")
        sys.exit(1)

    # 2. Get input matrices
    required_keys = ['device', 'Transformation_1', 'Transformation_2']
    if not all(key in data for key in required_keys):
        print(f"Error: YAML must contain {required_keys} keys.")
        sys.exit(1)

    matrices = {}
    for key in required_keys:
        matrix_list = data[key]
        if not (isinstance(matrix_list, list) and all(isinstance(row, list) for row in matrix_list)):
            print(f"Error: '{key}' must be a list of lists.")
            sys.exit(1)
        matrices[key] = np.array(matrix_list)
        if matrices[key].shape != (4, 4):
            print(f"Error: Matrix '{key}' must be 4x4. Shape is {matrices[key].shape}")
            sys.exit(1)

    m_device = matrices['device']
    m_trans1 = matrices['Transformation_1']
    m_trans2 = matrices['Transformation_2']

    # 3. Perform multiplication: result = (device * trans1) * trans2
    print("--- Calculating: intermediate = device * Transformation_1")
    intermediate_result = np.matmul(m_device, m_trans1)
    print("--- Calculating: final_result = intermediate * Transformation_2")
    final_result_matrix = np.matmul(intermediate_result, m_trans2)

    # 4. Store final result matrix
    data['result'] = final_result_matrix.tolist()
    print(f"--- Final result matrix calculated.")

    # 5. Extract Position (x, y, z from the last column)
    position = final_result_matrix[:3, 3].tolist()
    data['result_position'] = position
    print(f"--- Extracted Position: {position}")

    # 6. Extract Rotation Matrix and calculate Quaternion
    rotation_matrix = final_result_matrix[:3, :3]
    try:
        r = Rotation.from_matrix(rotation_matrix)
        # Get quaternion in [x, y, z, w] format (SciPy default)
        quaternion = r.as_quat().tolist()
        data['result_quaternion'] = quaternion
        print(f"--- Calculated Quaternion (x,y,z,w): {quaternion}")
    except ValueError as e:
        print(f"Error calculating quaternion: {e}. The rotation matrix might be invalid.")
        # Decide if you want to exit or continue without quaternion
        data['result_quaternion'] = None # Or some error indicator
        print("--- Skipping quaternion calculation.")


    # 7. Write updated data back to YAML
    with open(yaml_filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=None, sort_keys=False, allow_unicode=True)

    print(f"Successfully updated YAML file: '{yaml_filepath}' with result, position, and quaternion.")

except FileNotFoundError:
    print(f"Error: File not found during read/write: {yaml_filepath}")
    sys.exit(1)
except (yaml.YAMLError, TypeError, ValueError, KeyError) as e:
    print(f"An error occurred during YAML processing or calculation: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)