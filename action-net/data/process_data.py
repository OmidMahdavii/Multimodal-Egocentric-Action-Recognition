import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def low_pass_filter(data, cutoff=5, fs=200):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data


def segment_data(df, segment_size=100):
    # List to store the segmented data
    segmented_data = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # Segment the left readings and timestamps
        left_readings = np.array(row['myo_left_readings'])
        left_timestamps = np.array(row['myo_left_timestamps'])
        
        # Segment the right readings and timestamps
        right_readings = np.array(row['myo_right_readings'])
        right_timestamps = np.array(row['myo_right_timestamps'])

        # Handle the case where the number of the readings is different for the sensors
        if left_timestamps.shape[0] < right_timestamps.shape[0]:
            min_len = left_timestamps.shape[0]
            final_timestamps = left_timestamps
        else:
            min_len = right_timestamps.shape[0]
            final_timestamps = right_timestamps
        
        # Determine how many segments can be created
        num_segments = min_len // segment_size
        
        for i in range(num_segments):
            segmented_data.append({
                'idx': idx,
                'start': final_timestamps[i * segment_size],
                'stop': final_timestamps[(i + 1) * segment_size - 1],
                'myo_left_readings': left_readings[i * segment_size: (i + 1) * segment_size],
                'myo_left_timestamps': left_timestamps[i * segment_size: (i + 1) * segment_size],
                'myo_right_readings': right_readings[i * segment_size: (i + 1) * segment_size],
                'myo_right_timestamps': right_timestamps[i * segment_size: (i + 1) * segment_size],
            })

    # segmented_data = []
    # for i, row in df.iterrows():
    #     if len(row['myo_left_timestamps']) == 0:
    #         continue
    #     start, stop = row['start'], row['stop']
    #     duration = stop - start
    #     num_segments = int(np.floor(duration / segment_length))
        
    #     for j in range(num_segments):
    #         seg_start = start + j * segment_length
    #         seg_stop = seg_start + segment_length
            
    #         segment = row.copy()
    #         segment['idx'] = i
    #         segment['start'] = seg_start
    #         segment['stop'] = seg_stop
            
    #         for arm in ['myo_left', 'myo_right']:
    #             readings = np.array(row[f'{arm}_readings'])
    #             timestamps = np.array(row[f'{arm}_timestamps'])
                
    #             # Segment the readings and timestamps
    #             mask = (timestamps >= seg_start) & (timestamps < seg_stop)
    #             segment[f'{arm}_readings'] = readings[mask]
    #             segment[f'{arm}_timestamps'] = timestamps[mask]
            
    #         # Handle the case where the number of the readings is different for the sensors
    #         min_len = min(segment['myo_left_readings'].shape[0], segment['myo_right_readings'].shape[0])
    #         segment['myo_left_timestamps'] = segment['myo_left_timestamps'][:min_len]
    #         segment['myo_right_timestamps'] = segment['myo_right_timestamps'][:min_len]
    #         segment['myo_left_readings'] = segment['myo_left_readings'][:min_len]
    #         segment['myo_right_readings'] = segment['myo_right_readings'][:min_len]
            # segmented_data.append(segment)
    
    return pd.DataFrame(segmented_data, index=range(len(segmented_data)))


labels_dict = { 'Spread': 0,
                'Get/Put': 1,
                'Clear': 2,
                'Slice': 3,
                'Clean': 4,
                'Pour': 5,
                'Load': 6,
                'Peel': 7,
                'Open/Close': 8,
                'Set': 9,
                'Stack': 10,
                'Unload': 11
}

emg_data = {'S00_2': None,
            'S01_1': None,
            'S02_2': None,
            'S02_3': None,
            'S02_4': None,
            'S03_1': None,
            'S03_2': None,
            'S04_1': None,
            'S05_2': None,
            'S06_1': None,
            'S06_2': None,
            'S07_1': None,
            'S08_1': None,
            'S09_2': None
            }

for s in emg_data:
    df =  pd.DataFrame(pd.read_pickle(f'emg\{s}.pkl'))
    emg_data[s] = segment_data(df)
    for i, row in emg_data[s].iterrows():
        # Rectification
        myo_left_abs = np.abs(row['myo_left_readings'])
        myo_right_abs = np.abs(row['myo_right_readings'])

        # Low-pass filtering
        myo_left_filtered = low_pass_filter(myo_left_abs)
        myo_right_filtered = low_pass_filter(myo_right_abs)

        # Normalization
        myo_left_normalized = normalize(myo_left_filtered)
        myo_right_normalized = normalize(myo_right_filtered)

        # # Rectification
        # myo_left_normalized_abs = np.abs(myo_left_normalized)
        # myo_right_normalized_abs = np.abs(myo_right_normalized)

        # # Sum across channels for forearm activation
        # forearm_activation_left = np.sum(myo_left_normalized_abs, axis=1)
        # forearm_activation_right = np.sum(myo_right_normalized_abs, axis=1)
        
        # # Smooth the summed signals
        # forearm_activation_left_smooth = low_pass_filter(forearm_activation_left)
        # forearm_activation_right_smooth = low_pass_filter(forearm_activation_right)

        # Store processed data
        emg_data[s].at[i, 'myo_left_readings'] = myo_left_normalized
        emg_data[s].at[i, 'myo_right_readings'] = myo_right_normalized
        # emg_data[s].at[i, 'myo_left_readings'] = forearm_activation_left_smooth
        # emg_data[s].at[i, 'myo_right_readings'] = forearm_activation_right_smooth



train_df =  pd.DataFrame(pd.read_pickle('.\ActionNet_train.pkl'))
test_df =  pd.DataFrame(pd.read_pickle('.\ActionNet_test.pkl'))

train_set = pd.DataFrame()
test_set = pd.DataFrame()

for i, row in train_df.iterrows():
    subject = row['file'][:-4]
    df = emg_data[subject]

    data = df[df['idx'] == row['index']].copy()
    data['label'] = labels_dict[row['labels']]
    train_set = pd.concat([train_set, data], ignore_index=True)

for i, row in test_df.iterrows():
    subject = row['file'][:-4]
    df = emg_data[subject]

    data = df[df['idx'] == row['index']].copy()
    data['label'] = labels_dict[row['labels']]
    test_set = pd.concat([test_set, data], ignore_index=True)


train_set.to_pickle('train_set.pkl')
test_set.to_pickle('test_set.pkl')


