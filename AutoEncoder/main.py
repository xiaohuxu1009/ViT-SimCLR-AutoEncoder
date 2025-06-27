from model import Autoencoder
from train import train_autoencoder
from extract import extract_features
from utils import load_data_csv, save_features_csv
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./train_features.csv")
    parser.add_argument("--val_input", type=str, default="./val_features.csv")
    parser.add_argument("--output", type=str, default="./train_compressed.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--save_model", type=str, default="autoencoder_weights.pt", help="Model saving path")
    args = parser.parse_args()

    print("Loading train and val data...")
    train_data, index, _ = load_data_csv(args.input)
    val_data, _, _ = load_data_csv(args.val_input)

    print("Building model...")
    model = Autoencoder(input_dim=train_data.shape[1], latent_dim=args.latent_dim)

    print("Training...")
    model = train_autoencoder(model, train_data, val_data=val_data, epochs=args.epochs)

    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved: {args.save_model}")

    print("Extracting compressed features...")
    compressed = extract_features(model, train_data)

    print("Saving compressed features...")
    save_features_csv(compressed, index, args.output)
    print("Compressed features saved:", args.output)



if __name__ == "__main__":
    main()
