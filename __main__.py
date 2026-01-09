# Author: Alberto Fernández Comesaña

import re
import os
import sys
import joblib

from _model import train_model, evaluate_model, predict
import utils
from const import PREDS_DIR

if __name__ == "__main__":
    
    # TODO:
    # - Option for specifying pattern for river file names: default "Inflow_file_<river_name>.csv"
    # - Specify in the instructions that the date format must be in format YYYY-MM-DD
    # - Option for specifying date format in the CSV files (default YYYY-MM-DD)
    # - Option for specifying path for storing the comparison plots between predicted and real discharge values.
    # - Option for specifying which model wants to be used (default Linear Regression)
    # - Option for selecting which of the trained models wants to be used
    # - Apply standard scaler for linear regression to improve interpretability
    
    # --------------
    # Default values
    # --------------
    discharge_path = None
    river_path = None
    meteo_path = None
    split_ratio = 0
    river_groups = {}
    interfaces = ["A", "B", "C"]
    flow_colname = "flow"
    model_type = "linear_regression"
    
    # ---------------
    # Instructions
    # ---------------
    instructions = """
Usage: python model [options]
Options:
    -d, --discharge-path <path>     Default: None. Path to directory containing discharge CSV files
    -r, --river-path <path>         Default: None. Path to directory containing river flow CSV files
    -m, --meteo-path <path>         Default: None. Path to directory containing meteorological data CSV files
    -s, --split-ratio <float>       Default: 0. Ratio for splitting data into training and testing sets (between 0 and 1).
                                    A value of 0 means no training, all data is used for testing/prediction. A value of 1 means all data is used for training.
                                    Any value in between means that percentage of data is used for training and the rest for testing.
    -rg, --river-groups <str>       Default: None. Groups of rivers to add up their flows.
                                    Format: <group name 1>:<river name 1>,<river name 2>,...,<river name N>#<group name 2>:<river name 1>,<river name 2>,...,<river name N>#...
                                    Example: west:kopingsan,hedstromme#nnorth:fyrisan,orsundaan,savaan
                                    This will create two new columns in the river flow data named 'west' and 'north' containing the sum of the flows of the respective rivers and discard the flows of the individual rivers.
                                    River names should be formed only by lower case letter from the english alphabet and be present in the CSV file containing the river flow data.
                                    River group names can contain uppercase and lowercase letter from the english alphabet, as well as numbers, underscore '_' and hyphen '-'.
    -i, --interfaces <str>          Default: A,B,C. Interfaces to consider in the lake to consider for training or prediction.
                                    Format: An undetermined number of uppercase letters from the english alphabet separated by commas.
                                    Example: A,B,C
    -f, --flow-colname <str>        Default: flow. Name of the column containing the flow data in the river CSV files.
    
If the split-ratio is greater than 0, meteo-path parameter, river-path parameter or both must be provided, since those are the input data for the model.
In addition, discharge-path parameter must be provided since that is the target data for training.

If the split-ratio is 0, discharge-path parameter won't be used even though it is provided.
"""

    if len(sys.argv) == 1:
        print(instructions)
        sys.exit(0)

    # ---------------------
    # Read provided options
    # ---------------------
    for idx, arg in enumerate(sys.argv):
        
        if idx == 0:
            continue
        
        if arg in ('-d', '--discharge-path'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for discharge path.")
                sys.exit(1)
                
            discharge_path = sys.argv[idx+1]
            
            if not os.path.isdir(discharge_path):
                print("Provided discharge path does not exist or is not a directory.")
                sys.exit(1)
                
            if os.listdir(discharge_path) == 0:
                print("Discharge directory is empty.")
                sys.exit(1)
                
        if arg in ('-r', '--river-path'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for river path.")
                sys.exit(1)
                
            river_path = sys.argv[idx+1]
            
            if not os.path.isdir(river_path):
                print("Provided river path does not exist.")
                sys.exit(1)
                
            if os.listdir(river_path) == 0:
                print("River directory is empty.")
                sys.exit(1)
                
        if arg in ('-m', '--meteo-path'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for meteo path.")
                sys.exit(1)
                
            meteo_path = sys.argv[idx+1]
            
            if not os.path.isdir(meteo_path):
                print("Provided meteo path does not exist.")
                sys.exit(1)
                
            if os.listdir(meteo_path) == 0:
                print("Meteorological data directory is empty.")
                sys.exit(1)
                
        if arg in ('-s', '--split-ratio'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for split ratio.")
                sys.exit(1)
                
            split_ratio = float(sys.argv[idx+1])
            
            if split_ratio > 1 or split_ratio <= 0:
                print(f"Argument provided for split ratio '{split_ratio}' is not valid. Must be a number between 0 and 1 (e.g: 0.8)")
                sys.exit(1)
                
        if arg in ('-rg', '--river-groups'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for river groups.")
                sys.exit(1)
                
            river_groups_str = sys.argv[idx+1]

            if re.match(r'^(([a-zA-Z0-9\_]+:([a-z]+,?)*)#?)*$', river_groups_str) is None:
                print("Argument provided for river groups is not valid. River groups must be defined like")
                print("<group name 1>:<river name 1>,<river name 2>,...,<river name N>#<group name 2>:<river name 1>,<river name 2>,...,<river name N>#...")
                print("For example:")
                print("west:kopingsan,hedstromme#nnorth:fyrisan,orsundaan,savaan")
                print()
                print("River names should be formed only by lower case letter from the english alphabet and be present in the CSV file containing the river flow data.")
                print("River group names can contain uppercase and lowercase letter from the english alphabet, as well as numbers, underscore '_' and hyphen '-'.")
            
            for group in river_groups_str.split("#"):
                group_name, rivers = group.split(":")
                river_groups[group_name] = rivers.split(",")
                
        if arg in ('-i', '--interfaces'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for interfaces.")
                sys.exit(1)
            
            i_str = sys.argv[idx+1]
            if re.match(r'^([A-Z],?)+$', i_str) is None: 
                print("Argument provided for interfaces is not valid.")
                print("Interfaces argument should be formed by an undetermined number of uppercase letters from the english alphabet separated by commas")
                print("For example:")
                print("A")
                print("A,B")
                print("A,B,C")
                print()
                print("Do not introduce blank spaces between the letters, just commas.")
            
            interfaces = i_str.split(",")
            
        if arg in ('-f', '--flow-colname'):
            
            if len(sys.argv) < idx + 2:
                print("Argument was not provided for the flow column name.")
                sys.exit(1)
            
            flow_colname = sys.argv[idx + 1]
            
            
    # Validate split ratio and provided paths
    if split_ratio > 0:
        
        if meteo_path is None and river_path is None:
            print("For training, at least one of meteorological data path or river data path must be provided.")
            sys.exit(1)
        
        if discharge_path is None:
            print("For training, discharge path must be provided.")
            sys.exit(1)

    
    print("Provided options:")
    print(f"- Discharge path: {discharge_path}")
    print(f"- River path: {river_path}")
    print(f"- Meteorological data path: {meteo_path}")
    print(f"- Split ratio: {split_ratio}")
    print(f"- River groups: {river_groups}")
    print(f"- Interfaces: {interfaces}")
    print(f"- Flow column name: {flow_colname}")
    print()
    
    is_training = 0 < split_ratio <= 1
    
    if is_training:
        
        print("Loading data for model training...")
        X_train, Y_train, X_test, Y_test = utils.load_data(
            discharge_dir=discharge_path,
            meteo_dir=meteo_path,
            river_dir=river_path,
            split_ratio=split_ratio,
            interfaces=interfaces,
            groups=river_groups,
            flow_colname=flow_colname)
        
        print("Data loaded.")
        print("Starting model training...")
        
        train_model(X_train, Y_train, model_type=model_type)
        
        print("Model trained.")
        
        if split_ratio < 1:
            print("Starting model evaluation...")
            evaluate_model(X_test, Y_test, model_type=model_type)
            print("Model evaluation completed.")
        
        
    else:
        
        print("Loading data for prediction...")
        
        X = utils.load_data(
            meteo_dir=meteo_path,
            river_dir=river_path,
            split_ratio=split_ratio,
            interfaces=interfaces,
            groups=river_groups,
            flow_colname=flow_colname)
        
        print(f"Making predictions with {model_type} model...")
        predict(X, model_type=model_type)
