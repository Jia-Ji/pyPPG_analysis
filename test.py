from pyPPG.example import ppg_example
import numpy as np
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
import sys
import json
import pyPPG
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.signal import resample
import pandas as pd

# ppg_example(check_ppg_len=False)

'''
Load PPG data, set parameters
'''
train_data = np.load('x_test.npy')
train_data = train_data.reshape(train_data.shape[0], -1)
fs = 240
start_sig = 0
end_sig = -1
# savingfolder = 'biomarkers'
# savingformat = 'csv'


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


'''
save PPG segmnet as .mat
'''

pad_width = 960
for i, segment in enumerate(train_data[0:5]):
    # segment = resample(segment, int(len(segment) *  100 / 240))
    # segment = np.tile(segment, 2)
    # print(segment.shape)
    segment = np.pad(segment, pad_width=pad_width, mode='constant', constant_values=0)
    signal_column = segment.reshape(-1, 1)  # Convert to (1920, 1)
    np.expand_dims(signal_column, axis=1)
    # signal_column = signal_column.tolist()
    # print(signal_column.shape)
    mat_data = {
        'Data': signal_column,
        'Fs': 240
    }

    filename = f"segment_{i}.mat"
    savemat(filename, mat_data)


'''
load an example PPG
'''

data_path = "segment_4.mat"
start_sig = 0 # the first sample of the signal to be analysed
end_sig = -1 # the last sample of the signal to be analysed (here a value of '-1' indicates the last sample)
savingfolder = 'temp_dir'
savingformat = 'csv'

signal0 = load_data(data_path=data_path, start_sig=start_sig, end_sig=end_sig, use_tk=False)
signal0.v = signal0.v

# PPG signal processing
signal0.filtering = True # whether or not to filter the PPG signal
signal0.fL=0.5000001 # Lower cutoff frequency (Hz)
signal0.fH=12 # Upper cutoff frequency (Hz)
signal0.order=4 # Filter order
signal0.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

prep = PP.Preprocess(fL=signal0.fL, fH=signal0.fH, order=signal0.order, sm_wins=signal0.sm_wins)
signal0.ppg, signal0.vpg, signal0.apg, signal0.jpg = prep.get_signals(s=signal0)
# plot_derived_signal(signal=signal0)


'''
Store the derived signals in a class
'''
# Initialise the correction for fiducial points
corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
correction=pd.DataFrame()
correction.loc[0, corr_on] = True
signal0.correction=correction

# Create a PPG class
s = PPG(s=signal0, check_ppg_len=False)

fpex = FP.FpCollection(s)
temp_fiducials = fpex.get_fiducials(s)

def filter_fiducials(df: pd.DataFrame, s_length: int, pad_width: int) -> pd.DataFrame:
    df_copy = df.copy()

    for col in df_copy.columns:
        if col != "Index of pulse":
            df_copy[col] = df_copy[col].apply(lambda x: x - pad_width if pd.notna(x) else x)

    # Remove rows with any value < 0 or > segment_length
    def is_valid_row(row):
        return all((pd.isna(val) or (0 <= val <= s_length)) for val in row)

    df_filtered = df_copy[df_copy.apply(is_valid_row, axis=1)].reset_index(drop=True)

    # Reset row index and keep "Index of pulse" as a column
    df_filtered.index.name = "Index of pulse"

    # Optional: convert entire DataFrame (except "Index of pulse") to int if all values are valid ints
    for column in df_filtered.columns:
        df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce').astype("Int64")

    return df_filtered

fiducials = filter_fiducials(temp_fiducials, 1920, pad_width)

print(fiducials)
# print(temp_fiducials.dtypes)
# print(fiducials.dtypes)

# print(type(fiducials['dp']))


# print("Fiducial points:\n",fiducials + s.start_sig)
# Create a fiducials class
fp = Fiducials(fiducials)




# Plot fiducial points
plot_fiducials(s, fp, savingfolder, legend_fontsize=12)
