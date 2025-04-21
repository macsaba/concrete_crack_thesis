import pandas as pd
import os
import numpy as np
import torch

def log_training_result(csv_path, new_row):
    """
    Logs a training result to a CSV file using pandas.

    Args:
        csv_path (str): Path to the CSV file.
        new_row (dict): Dictionary of training values to log. 
                        Keys are column names; values are the record values.
    """
    # Load existing CSV or create new DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # Add missing columns from new_row
    for key in new_row:
        if key not in df.columns:
            df[key] = None  # or pd.NA

    # Ensure all columns are included in the new row (missing cols = None)
    for col in df.columns:
        if col not in new_row:
            new_row[col] = None

    # Append the new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save it back to CSV
    df.to_csv(csv_path, index=False)

def save_model_files(files_path, model_weights, np_arrays):
    if os.path.exists(files_path):
        raise FileExistsError('Files path already exists.')
    else:
        # Create the directory
        os.mkdir(files_path)
        for key, value in model_weights.items():
            torch.save(value, files_path + key + '.pth')
        for key, value in np_arrays.items():
            value_np = np.array(value)
            np.save(files_path + key + '.npy', value_np)
