import os
import argparse
import torch

from models.regressor import SingleInputRegressor
from predict.predict_3D import predict_3D

def main(input_path, checkpoint_path, device, silhouettes_from):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)

    print("\nModel Loaded. ",'Weights from:', checkpoint_path)
    regressor.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    predict_3D(input_path, regressor, device, silhouettes_from=silhouettes_from)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--silh_from', choices=['densepose', 'pointrend'])
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    main(args.input, args.checkpoint, device, args.silh_from)
