import torch
import argparse
from model import Autoencoder
from extract import extract_features
from utils import load_data_csv, save_features_csv


def load_model(model_path, input_dim=768, latent_dim=10):
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./val_features.csv", help="768-dimensional feature CSV file")
    parser.add_argument("--model", type=str, default="./autoencoder_weights.pt", help="File path of the trained model weights (.pt)")
    parser.add_argument("--output", type=str, default="./val_compressed.csv", help="Output CSV path for saving 10-dimensional features")
    parser.add_argument("--latent_dim", type=int, default=10)
    args = parser.parse_args()

    print("Load Features...")
    data, index, _ = load_data_csv(args.input)

    print("Load Model...")
    model = load_model(args.model, input_dim=data.shape[1], latent_dim=args.latent_dim)

    print("Reduce Dimensions...")
    compressed = extract_features(model, data)

    print("Save the result...")
    save_features_csv(compressed, index, args.output)
    print("Finish")


if __name__ == "__main__":
    main()
