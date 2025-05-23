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
from scipy.signal import resample
import pandas as pd
from tqdm import tqdm


from utils import convert_npy_to_mat, plot_ppg_data, estimate_HR
from utils import calculate_SQI, delete_empty_dirs, merge_ppg_segment_csvs


'''
Parameters
'''
ppg_data_path = 'data/ppg_data/x_test.npy'
mat_save_path = 'data/segments'
fig_save_path = 'output/test'
savingfolder = 'output/test'
temp_mat_save_path = 'data/temp_segments'
biomarker_file_name = 'tests.csv'
fs = 240
start_sig = 0
end_sig = -1
use_tk = False
pad_width = 960
tile_reps = 2
savingformat = 'csv'


'''
Functions
'''

def create_ppg(s_path, start_sig=0, end_sig=-1, use_tk=False):
    signal0 = load_data(data_path=s_path, start_sig=start_sig, end_sig=end_sig, use_tk=use_tk)
    signal0.v = signal0.v

    # PPG signal processing
    signal0.filtering = True # whether or not to filter the PPG signal
    signal0.fL=0.5000001 # Lower cutoff frequency (Hz)
    signal0.fH=12 # Upper cutoff frequency (Hz)
    signal0.order=4 # Filter order
    signal0.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

    prep = PP.Preprocess(fL=signal0.fL, fH=signal0.fH, order=signal0.order, sm_wins=signal0.sm_wins)
    signal0.ppg, signal0.vpg, signal0.apg, signal0.jpg = prep.get_signals(s=signal0)

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction=pd.DataFrame()
    correction.loc[0, corr_on] = True
    signal0.correction=correction

    # Create a PPG class
    s = PPG(s=signal0, check_ppg_len=False)

    return s

def filter_temp_fiducials(df: pd.DataFrame, s_length: int, pad_width: int) -> pd.DataFrame:
    df_copy = df.copy()

    for col in df_copy.columns:
        # if col != "Index of pulse":
        #     df_copy[col] = df_copy[col].apply(lambda x: x - pad_width if pd.notna(x) else x)
        def adjust(val):
                if pd.isna(val):
                    return pd.NA
                new_val = val - pad_width
                return new_val if 0 <= new_val <= s_length-1 else pd.NA
        df_copy[col] = df_copy[col].apply(adjust)

    # # Remove rows with any value < 0 or > segment_length
    # def is_valid_row(row):
    #     return all((pd.isna(val) or (0 <= val <= s_length)) for val in row)

    # df_filtered = df_copy[df_copy.apply(is_valid_row, axis=1)].reset_index(drop=True)

    # # Reset row index and keep "Index of pulse" as a column
    # df_filtered.index.name = "Index of pulse"
    fiducial_cols = df_copy.columns
    df_filtered = df_copy.dropna(subset=fiducial_cols, how="all").reset_index(drop=True)
    df_filtered.index.name = "Index of pulse"


    # Optional: convert entire DataFrame (except "Index of pulse") to int if all values are valid ints
    for column in df_filtered.columns:
        df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce').astype("Int64")

    return df_filtered

def calculate_hr(s, fp):
    HR=len(s.ppg)/len(fp.sp)*s.fs 
    return HR


if __name__ == '__main__':
    data = np.load(ppg_data_path)
    data = data.reshape(data.shape[0], -1)

    for i, segment in tqdm(enumerate(data), desc='Analyse ppg'):

        signal = convert_npy_to_mat(segment, pad =False, pad_width=pad_width, tile=False, tile_reps=tile_reps,save_path=mat_save_path, signal_index=i)  
        temp_signal = convert_npy_to_mat(segment, pad=True, pad_width=pad_width, tile=True, tile_reps=tile_reps, save_path=temp_mat_save_path, signal_index=i)

        # plot_ppg_data(signal['Data'], fs =fs)



        signal_path = mat_save_path+'/'+f"segment_{i}.mat"
        temp_signal_path = temp_mat_save_path+'/'+f"temp_segment_{i}.mat"
        s = create_ppg(s_path=signal_path, start_sig=start_sig, end_sig=end_sig, use_tk=use_tk)
        temp_s = create_ppg(s_path=temp_signal_path, start_sig=start_sig, end_sig=end_sig, use_tk=use_tk)

        fpex = FP.FpCollection(temp_s)
        temp_fiducials = fpex.get_fiducials(temp_s)

        # print(len(s.ppg))
        fiducials = filter_temp_fiducials(temp_fiducials, len(s.ppg) , pad_width)

        # Create a fiducials class
        fp = Fiducials(fiducials)

        # Plot fiducial points
        # plot_fiducials(s, fp, savingfolder=fig_save_path, legend_fontsize=6) 

        # Estimate HR
        hr = estimate_HR(s, fp)

        # calculate signal quality index
        # sqi = calculate_SQI(s, fp)

        # Extract biomarkers
        bmex = BM.BmCollection(s, fp)
        
        selected_biomarkers = ["Tpi", "Tpp", "IPR"] 
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        tmp_keys=bm_stats.keys()
        print('Statistics of the biomarkers:')
        for i in tmp_keys: print(i,'\n',bm_stats[i])

        bm = Biomarkers(bm_defs, bm_vals, bm_stats)

        # save ppg data
        fp_new = Fiducials(fp.get_fp() + s.start_sig) # here the starting sample is added so that the results are relative to the start of the original signal (rather than the start of the analysed segment)
        save_data(savingformat=savingformat, savingfolder=savingfolder, print_flag=False,s=s, fp =fp, bm=bm )
    merge_ppg_segment_csvs(savingfolder, 'Biomarker_stats', output_file=biomarker_file_name)
    delete_empty_dirs(savingfolder)

                

        

