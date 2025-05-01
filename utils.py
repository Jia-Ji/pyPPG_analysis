from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
import os
import pandas as pd
from pyPPG import PPG, Fiducials
import pyPPG.ppg_sqi as SQI
import os
import re

def plot_ppg_data(signal, fs):

    # setup figure
    fig, ax = plt.subplots(figsize=(15, 5))

    # create time vector
    t = np.arange(0, len(signal))/fs

    # plot raw PPG signal
    ax.plot(t, signal, color = 'blue')
    ax.set(xlabel = 'Time (s)', ylabel = 'raw PPG')

    # show plot
    plt.show()

def plot_derived_signal(signal):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex = True, sharey = False)

    # create time vector
    t = np.arange(0, len(signal.ppg))/signal.fs

    # plot filtered PPG signal
    ax1.plot(t, signal.ppg)
    ax1.set(xlabel = '', ylabel = 'PPG')

    # plot first derivative
    ax2.plot(t, signal.vpg)
    ax2.set(xlabel = '', ylabel = 'PPG\'')

    # plot second derivative
    ax3.plot(t, signal.apg)
    ax3.set(xlabel = '', ylabel = 'PPG\'\'')

    # plot third derivative
    ax4.plot(t, signal.jpg)
    ax4.set(xlabel = 'Time (s)', ylabel = 'PPG\'\'\'')

    # show plot
    plt.show()


def estimate_HR(s: PPG, fp: Fiducials):
    num_beats=len(fp.sp)  # number of the beats
    duration_seconds=len(s.ppg)/s.fs  # duration in seconds
    HR = (num_beats / duration_seconds) * 60 # heart rate
    print('Estimated HR: ',HR,' bpm' )
    return HR

def calculate_SQI(s:PPG, fp: Fiducials):
    annotations = fp.sp.copy()
    # Convert to numpy if it’s pandas
    if isinstance(annotations, pd.Series):
        annotations = annotations.dropna().values

    # Remove invalid annotations (e.g., less than 1 or larger than PPG length)
    annotations = annotations[(annotations > 0) & (annotations < len(s.ppg))]

    ppgSQI = round(np.mean(SQI.get_ppgSQI(s.ppg, s.fs, annotations)) * 100, 2)
    print('Mean PPG SQI: ', ppgSQI, '%')
    return ppgSQI


def convert_npy_to_mat(s: np.array, pad:bool, tile:bool, tile_reps:int, pad_width: int, save_path:str, signal_index:int):
    if tile or pad:
        filename = f"temp_segment_{signal_index}.mat"
    
    if tile:
        s = np.tile(s, tile_reps)
    
    if pad:
        s = np.pad(s, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        filename = f"segment_{signal_index}.mat"
    signal_column = s.reshape(-1, 1)  # Convert to (1920, 1)
    np.expand_dims(signal_column, axis=1)
    # signal_column = signal_column.tolist()
    # print(signal_column.shape)
    mat_data = {
        'Data': signal_column,
        'Fs': 50
    }
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    savemat(save_path+'/'+filename, mat_data)

    return mat_data

def delete_empty_dirs(root_dir):
    """
    Recursively delete empty subdirectories under the given root directory.
    
    Args:
        root_dir (str): Path to the root directory to check.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Check if directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty directory: {dirpath}")
            except OSError as e:
                print(f"Error deleting {dirpath}: {e}")

# def merge_ppg_segment_csvs(savingpath, temp_dir, csv_num):
#     """
#     Merge all PPG segment CSV files into a single CSV.

#     Args:
#         directory (str): Path to the directory containing the CSV files.
#         output_path (str): Path to save the merged CSV.
#     """
#     dir_path = os.path.join(savingpath, temp_dir)
#     merged_data = {}

#     print(csv_num)


#     for filename in os.listdir(dir_path):
#         match = re.match(r'segment_(\d+)[_ppg_sig]*[_ratios]*_btwn_(\d+)-(\d+)\.csv', filename)
#         if match:
#             index = int(match.group(1))
#             filepath = os.path.join(dir_path, filename)
#             df = pd.read_csv(filepath, index_col=0)
#             merged_data[index] = df

#     if not merged_data:
#         print("No matching files found.")
#         return

#     # Merge into a multi-index column DataFrame
#     merged_df = pd.concat(merged_data, axis=1)
#     merged_df.columns.names = ['Segment_Index', 'Statistics']

#     # Save each feature to its own file
#     for feature in merged_df.columns.levels[1]:
#         feature_df = merged_df.xs(feature, level='Statistics', axis=1).T

#         feature_df.index.name = 'Segment_Index'
#         feature_df = feature_df.sort_index()

#         output_path = os.path.join(savingpath, f"{feature}.csv")
#         feature_df.to_csv(output_path)
#         print(f"Saved: {output_path}")

def merge_ppg_segment_csvs(savingpath, temp_dir, csv_num):
    """
    Merge all PPG segment CSV files into a single CSV.

    Args:
        directory (str): Path to the directory containing the CSV files.
        output_path (str): Path to save the merged CSV.
    """
    dir_path = os.path.join(savingpath, temp_dir)
    data_rows = []


    for filename in os.listdir(dir_path):
        match = re.match(r'segment_(\d+)[_ppg_sig]*[_ratios]*_btwn_(\d+)-(\d+)\.csv', filename)
        if match:
            index = int(match.group(1))
            filepath = os.path.join(dir_path, filename)
            df = pd.read_csv(filepath, index_col=0)  

            flat_row = {}
            for col in df.columns:
                for stat in df.index:
                    flat_row[f"{col}_{stat}"] = df.loc[stat, col]
            flat_row['Segment'] = index
            data_rows.append(flat_row)  

    if not data_rows:
        print("No matching files found.")
        return
    
    final_df = pd.DataFrame(data_rows).set_index('Segment').sort_index().groupby('Segment').first()
    output_path = os.path.join(savingpath, "biomarkers_stats.csv")
    final_df.to_csv(output_path)
    print(f"Merged CSV saved to: {output_path}")