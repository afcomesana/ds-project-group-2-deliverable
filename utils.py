# Author: Alberto Fernández Comesaña
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_meteo_data(meteo_dir:str, interfaces:str|list) -> pd.DataFrame:
    
    df = []
    
    month_col = None # store month values once
    
    for filename in os.listdir(meteo_dir):

        # Filter filenames corresponding to the provided interfaces
        if re.match(r'^(%s)[0-9]+\_[0-9]+\.csv' % "|".join(interfaces), filename) is None:
            continue
        
        # Read zone data and define name for dataframe
        filepath = os.path.join(meteo_dir, filename)
        zone_df = pd.read_csv(filepath)
        zone_name = filename.replace('.csv', '')
        
        # Add month feature
        if month_col is None:
            month_col = pd.Series(
                [int(dt.split('-')[1]) for dt in zone_df['date']],
                name='month'
            )
        
        # Add column to dataframe
        colnames = [col for col in zone_df.columns if col != 'date']
        renamed = zone_df[colnames].add_prefix(f'{zone_name}_')
        df.append(renamed)
    
    df = pd.concat([month_col] + df, axis=1)
    return df
    
def load_river_data(river_dir:str, flow_colname:str = 'flow', groups:dict = {}) -> pd.DataFrame:
    # TODO: Compare rivers inputs separately vs Adding up north/west flows
    
    df = pd.DataFrame()
    
    for river_file in os.listdir(river_dir):
        
        # Extract river name from filename
        river_name = re.findall(r'^Inflow\_file\_([a-z]+)\.csv$', river_file)[0]
        
        river_df = pd.read_csv(os.path.join(river_dir, river_file))
        
        if len(groups.keys()):
            for group_name in groups.keys():
                
                if river_name not in groups[group_name]:
                    continue
                
                if group_name in df.columns:
                    df[group_name] += river_df[flow_colname]
                    
                else:
                    df[group_name] = river_df[flow_colname]
                    
        else:
            df[river_name] = river_df[flow_colname]
        
    return df

def load_discharge(discharge_dir:str, interfaces:str|list = ["A", "B", "C"]) -> pd.DataFrame:
    df = pd.DataFrame()

    for filename in os.listdir(discharge_dir):
        
        zone_name = re.findall(r'^discharge_([%s][0-9]+)\.csv$' % "|".join(interfaces), filename)
        
        if len(zone_name) == 0:
            continue

        zone_name = zone_name[0]
        df[zone_name] = pd.read_csv(os.path.join(discharge_dir, filename))['flow']
        
    return df
    
def load_data(discharge_dir:str = None, meteo_dir:str = None, river_dir:str = None, interfaces:list = ["A", "B", "C"], split_ratio = 0, flow_colname = 'flow', groups:dict = {}) -> pd.DataFrame:

    if meteo_dir is None and river_dir is None:
        raise Exception("At least 1 directory containing the input data (rivers or meteorology) must be provided.")

    # --------------------
    # Meteorological data:
    # --------------------
    meteo_df = pd.DataFrame()
    if meteo_dir is not None:
        
        if not os.path.exists(meteo_dir):
            raise Exception(f"Provided directory for meteorological data '{meteo_dir}' does not exists.")
        
        if len(os.listdir(meteo_dir)) == 0:
            raise Exception(f"Provided directory for meteorological data '{meteo_dir}' is empty."   )
        
        meteo_df = load_meteo_data(meteo_dir, interfaces)
        
    
    # ----------------
    # River flow data:
    # ----------------
    river_df = pd.DataFrame()
    if river_dir is not None:
        
        if not os.path.exists(river_dir):
            raise Exception(f"Provided directory for rivers flow data '{river_dir}' does not exists.")
        
        if len(os.listdir(river_dir)) == 0:
            raise Exception(f"Provided directory for rivers data '{river_dir}' is empty.")
        
        river_df = load_river_data(river_dir, flow_colname, groups=groups)
        
    # Prepare input data:
    X = pd.concat((meteo_df, river_df), axis=1)
        
    
    # ------------------------
    # Discharge data (target):
    # ------------------------
    Y = pd.DataFrame()
    if discharge_dir is not None:
        
        if not os.path.exists(discharge_dir):
            raise Exception(f"Provided directory for discharge data '{discharge_dir}' does not exists.")
        
        if len(os.listdir(discharge_dir)) == 0:
            raise Exception(f"Provided directory for discharge data '{discharge_dir}' is empty.")
    
        Y = load_discharge(discharge_dir, interfaces)
    
    # -----------------------
    # Train-test split
    # (ordered, not shuffled)
    # -----------------------
    
    if len(X) != len(Y) and len(Y):
        raise Exception(f"Input and Target data must have the same number of samples/rows: "
                        f"Input data has {len(X)} rows, "
                        f"but Target data has {len(Y)} rows.")
    
    
    is_training = 0 < split_ratio <= 1
    
    if is_training and (discharge_dir is None or (river_dir is None and meteo_dir is None)):
        raise Exception("Discharge data and meteorological or river data or both must be provided when training.")
    
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_train = pd.DataFrame()
    Y_test = pd.DataFrame()
    
    if is_training:
        
        test_index = int(len(X)*split_ratio)

        X_train, X_test = X.iloc[:test_index].copy(), X.iloc[test_index:].copy()    
        Y_train, Y_test = Y.iloc[:test_index].copy(), Y.iloc[test_index:].copy()
        
    # -------------------------
    # Add lagged precipitation:
    # -------------------------
    if len(meteo_df):
        for col in X.columns:
            if 'precip' not in col:
                continue
            
            if is_training:
                X_train[f'{col}_lag'] = pd.concat([pd.Series([None]), X_train[col][1:]])
                X_test[f'{col}_lag'] = pd.concat([pd.Series([None]), X_test[col][1:]])
                
            else:
                X[f'{col}_lag'] = pd.concat([pd.Series([None]), X[col][1:]])
                
            
        if is_training:
            
            X_train = X_train.iloc[1:,:]
            X_test  = X_test.iloc[1:,:]
            Y_train = Y_train.iloc[1:,:]
            Y_test  = Y_test.iloc[1:,:]
            
        else:
            X = X.iloc[1:,:]
            Y = Y.iloc[1:,:]
            
    if is_training:
        return X_train, Y_train, X_test,  Y_test
    
    else:
        return X


def plot_predictions(Y_test, Y_pred):
        
    plt.figure(figsize=(10,8))
    
    for fig_idx in range(1,5):
        plt.subplot(4,1,fig_idx)
        plt.title(f"B{fig_idx}")
        plt.plot(Y_test.iloc[:,fig_idx-1].values, label="Target")
        plt.plot(Y_pred[:,fig_idx-1], label="Prediction")
    
    plt.legend()
