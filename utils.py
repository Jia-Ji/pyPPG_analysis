from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
import os
import pandas as pd
from pyPPG import PPG, Fiducials
import pyPPG.ppg_sqi as SQI

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