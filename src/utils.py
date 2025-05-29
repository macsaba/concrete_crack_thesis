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

# save the results of the training
def save_model_files(files_path, model_weights, np_arrays, override = False):
    if os.path.exists(files_path) and not override:
        raise FileExistsError('Files path already exists.')
    else:
        # Create the directory
        os.makedirs(files_path, exist_ok=True)
        for key, value in model_weights.items():
            torch.save(value, files_path + key + '.pth')
        for key, value in np_arrays.items():
            value_np = np.array(value)
            np.save(files_path + key + '.npy', value_np)

# generates latex table from pandas dataset            
# e.g. generate_latex_table(results_table_log_reg, './tables/table1.tex')
def generate_latex_table(dataset, name):
    latex_output = dataset.to_latex(index=False, column_format='|' + 'c|'*dataset.shape[1], float_format="%.3f")
    latex_output = latex_output.replace("\\toprule", "\\hline\n\\rowcolor{gray!50}")
    latex_output = latex_output.replace("\\midrule", "\\hline")
    latex_output = latex_output.replace("\\bottomrule", "\\hline")
    latex_output = '\\resizebox{\\textwidth}{!}{\n' + latex_output + '}'
    #print(latex_output)
    # Save tables to LaTeX files
    with open(name, "w") as f:
        f.write(latex_output)

# generates latex table from pandas dataset            
# e.g. generate_latex_table(results_table_log_reg, './tables/table1.tex')
def generate_latex_table_thesis(dataset, name):
    latex_output = dataset.to_latex(index=False, column_format='|' + 'c|'*dataset.shape[1], float_format="%.3f")
    latex_output = latex_output.replace("\\toprule", "\\hline\n\\rowcolor{gray!50}")
    latex_output = latex_output.replace("\\midrule", "\\hline")
    latex_output = latex_output.replace("\\bottomrule", "\\hline")
    #latex_output = '\\resizebox{\\textwidth}{!}{\n' + latex_output + '}'
    #print(latex_output)
    # Save tables to LaTeX files
    with open(name, "w") as f:
        f.write(latex_output)

def generate_latex_table_thesis2(dataset, name):
    # Copy to avoid modifying original
    dataset = dataset.copy()

    # Escape underscores in column names
    dataset.columns = [str(col).replace('_', '\\_') for col in dataset.columns]

    # Escape underscores in all string values in the dataset
    for col in dataset.columns:
        dataset[col] = dataset[col].apply(
            lambda x: x.replace('_', '\\_') if isinstance(x, str) else x
        )

    # Generate LaTeX table
    latex_output = dataset.to_latex(
        index=False,
        column_format='|' + 'c|' * dataset.shape[1],
        float_format="%.3f",
        escape=False  # Disable default escaping since we manually handle it
    )

    # Beautify table for thesis style
    latex_output = latex_output.replace("\\toprule", "\\hline\n\\rowcolor{gray!50}")
    latex_output = latex_output.replace("\\midrule", "\\hline")
    latex_output = latex_output.replace("\\bottomrule", "\\hline")

    # Write LaTeX code to file
    with open(name, "w") as f:
        f.write(latex_output)
