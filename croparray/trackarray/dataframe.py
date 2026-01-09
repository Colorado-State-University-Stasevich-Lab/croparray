import numpy as np
import pandas as pd


def create_tracks_df(my_ta):
    # Initialize empty lists to store data
    track_dfs = []

    # Iterate over each track_id
    for track_id in my_ta.track_id.values:
        # Filter the xarray dataset by the specified track_id
        track_data = my_ta.sel(track_id=track_id)

        # Filter out rows where xc or yc are both [0, 0]
        valid_indices_xc = np.where(track_data.xc != 0)[0]
        valid_indices_yc = np.where(track_data.yc != 0)[0]
        valid_indices = np.intersect1d(valid_indices_xc, valid_indices_yc)

        # Filter the data based on the valid indices
        filtered_data = track_data.isel(t=valid_indices)

        # Create DataFrame
        track_df = pd.DataFrame({
            'track_id': [track_id] * len(filtered_data.t),
            't': filtered_data.t.values.tolist(),
            'y': filtered_data.yc[:, 0].values.tolist(),
            'x': filtered_data.xc[:, 0].values.tolist()
        })

        # Append the DataFrame to the list
        track_dfs.append(track_df)
    # track_dfs['y'] -= len(my_ta.y.values)
    # track_dfs['x'] -= len(my_ta.x.values)

    # Concatenate all DataFrames in the list
    result_df = pd.concat(track_dfs, ignore_index=True)

    result_df['y'] -= ((len(my_ta.y.values)-1)/2)
    result_df['x'] -= ((len(my_ta.x.values)-1)/2)
    return result_df


def track_signals_to_df(my_ta):
    """
    Combine signal and signal_raw data from each channel into a single DataFrame.

    Parameters:
    - my_ta: xarray.Dataset containing 'signal' and 'signal_raw' data for each channel.

    Returns:
    - DataFrame: Combined DataFrame with columns for each signal and signal_raw.
    """
    # Initialize an empty list to hold individual DataFrames for each channel
    combined_data = []

    # Get the number of channels
    num_channels = my_ta.ch.size

    # Loop through each channel to extract and combine the data
    for ch in range(num_channels):
        # Convert signal_raw to DataFrame
        df_signal_raw = my_ta.sel(ch=ch)['signal_raw'].to_dataframe().reset_index()
        
        # Convert signal to DataFrame
        df_signal = my_ta.sel(ch=ch)['signal'].to_dataframe().reset_index()

        # Merge both DataFrames on track_id, t, and ch
        df_combined = pd.merge(df_signal_raw, df_signal, on=['track_id', 't', 'ch'], suffixes=('', '_signal'))

        # Rename columns to distinguish between signal and signal_raw
        df_combined.rename(columns={'signal': f'signal_ch_{ch}', 'signal_raw': f'signal_raw_ch_{ch}'}, inplace=True)

        # Append the combined DataFrame to the list
        combined_data.append(df_combined)

    # Concatenate all DataFrames along the columns
    final_df = pd.concat(combined_data, axis=1)

    # Drop duplicate columns if necessary (like 'ch' appearing multiple times)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Remove unwanted columns
    columns_to_remove = ['ch', 'fov', 'fov_signal']
    final_df = final_df.drop(columns=columns_to_remove, errors='ignore')

    return final_df