import pandas as pd
import numpy as np

def load_data_csv(path):
    df = pd.read_csv(path)
    return df.values, df.index, df.columns

def save_features_csv(features, index, save_path, prefix="feature"):
    df = pd.DataFrame(features, index=index, columns=[f"{prefix}_{i}" for i in range(features.shape[1])])
    df.to_csv(save_path)
    print(f"Saved features to: {save_path}")
