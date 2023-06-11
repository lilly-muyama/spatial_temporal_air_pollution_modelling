import pandas as pd
import numpy as np

def get_device_data(full_df, device_id, cols):
    device_df = full_df[full_df.device_number==device_id]
    return device_df[cols]

def preprocessing(device_df):
    device_df.dropna(inplace=True)
    device_df = device_df.rename({'timestamp':'time', 'pm2_5_calibrated_value':'pm2_5'}, axis=1)
    device_df['time'] = [time.timestamp()/3600 for time in device_df['time']]
    return device_df

