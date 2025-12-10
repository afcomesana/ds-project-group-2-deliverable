import os

# Models will be stored within the 'models' module
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Generated figure will be stored in the working directory
# (from where the script is executed)
FIGURES_DIR = os.path.join(os.getcwd(), "figures")

# CSV directory, for predictions made
PREDS_DIR = os.path.join(os.getcwd(), "predictions")

# Column names filename
COLNAMES_FILENAME = "colnames.txt"