import os
import re
import sys
import utils
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from const import MODELS_DIR, FIGURES_DIR, COLNAMES_FILENAME, PREDS_DIR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train_model(X_train, Y_train, model_type="linear_regression"):
    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        raise Exception(f"Model type '{model_type}' is not supported.")
    
    model.fit(X_train, Y_train)
    
    # --------------------------------------
    # Store the model and the output columns
    # --------------------------------------
    model_dir = os.path.join(MODELS_DIR, model_type)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Store model
    joblib.dump(model, os.path.join(model_dir, f"{model_type}.joblib"))
    
    # Store column names
    with open(os.path.join(model_dir, COLNAMES_FILENAME), 'w') as f:
        f.write('\n'.join(Y_train.columns))
        
def load_model(model_type="linear_regression"):
    model_dir = os.path.join(MODELS_DIR, model_type)
    if not os.path.exists(model_dir):
        raise Exception(f"Model directory '{model_dir}' does not exist. Please train the model before predicting.")
    
    model = joblib.load(os.path.join(model_dir, f"{model_type}.joblib"))
    with open(os.path.join(model_dir, COLNAMES_FILENAME), 'r') as f:
        colnames = [line.strip() for line in f.readlines()]
        
    return model, colnames

def evaluate_model(X_test, Y_test, model_type="linear_regression"):
    model, _ = load_model(model_type=model_type)
    Y_pred = model.predict(X_test)
    print(f"R-squared: {r2_score(Y_test, Y_pred):.4f}")
    
    # Ensure the number of columns/features matches
    if Y_pred.shape[1] != Y_test.shape[1]:
        raise ValueError(
            f"The number of columns must match: "
            f"Predictions have {Y_pred.shape[1]} columns, "
            f"but Real Values have {Y_test.shape[1]} columns."
        )
    
    # Ensure the number of rows/samples matches
    if Y_pred.shape[0] != Y_test.shape[0]:
        raise ValueError(
            f"The number of rows/samples must match: "
            f"Predictions have {Y_pred.shape[0]} rows, "
            f"but Real Values have {Y_test.shape[0]} rows."
        )
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    n_cols = Y_test.shape[1]
    plt_cols = int(np.ceil(np.sqrt(n_cols)))
    plt_rows = int(np.ceil(n_cols / plt_cols))
    
    _, axes = plt.subplots(plt_rows, plt_cols, figsize=(5*plt_cols, 4*plt_rows))
    axes = axes.flatten()
    
    x_axis = np.arange(Y_test.shape[0])
    
    for i in range(n_cols):
        title = Y_test.columns[i]
        ax = axes[i]

        y_test = Y_test.iloc[:, i].values
        y_pred = Y_pred[:, i]
        
        ax.plot(x_axis, y_test, label="Target", color='blue')
        ax.plot(x_axis, y_pred, label="Prediction", color='orange')
        ax.set_title(title)
        ax.legend()
        
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_type}_evaluation.png"))
    plt.close()
    
def get_model_dir(model_type):
    return os.path.join(MODELS_DIR, model_type)

def predict(X:pd.Dataframe, model_type:str="linear_regression") -> pd.DataFrame:
    
    # Find model directory
    model, colnames = load_model(model_type=model_type)
    
    # Build dataframe for predictions
    Y_pred = model.predict(X)
    Y_pred = pd.DataFrame(Y_pred, columns=colnames)
    
    os.makedirs(PREDS_DIR, exist_ok=True)
    Y_pred.to_csv(os.path.join(PREDS_DIR, f"{model_type}_predictions.csv"), index=False)
    
    # Plot predictions
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    n_cols = Y_pred.shape[1]
    plt_cols = int(np.ceil(np.sqrt(n_cols)))
    plt_rows = int(np.ceil(n_cols / plt_cols))
    
    _, axes = plt.subplots(plt_rows, plt_cols, figsize=(5*plt_cols, 4*plt_rows))
    axes = axes.flatten()
    
    x_axis = np.arange(Y_pred.shape[0])
    
    for i in range(n_cols):
        title = Y_pred.columns[i]
        ax = axes[i]
        y_pred = Y_pred.iloc[:, i]
        
        ax.plot(x_axis, y_pred)
        ax.set_title(title)
        
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_type}_prediction.png"))
    plt.close()
