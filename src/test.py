import os
from glob import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import segmentation_models as sm
sm.set_framework('tf.keras')

from src.models import build_unet
from src.data_loader import DataGenerator
from src.losses import dice_loss 

SEED = 42

def main(args):
    # Select needed constants. for more flexible control 
    # it is better to use .yaml or .ini train/test config 
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    df = pd.read_csv(args.dataframe_labels_path)
    # remove bug image
    # df = df[df['ImageId'] != '6384c3e78.jpg']
    df["image_path"] = args.test_dir+df["ImageId"]
    # if you downloaded a sample in the data directory: to prevent reading not dowloaded images
    # df = df[df["ImageId"].isin([Path(path).name for path in glob(f"{args.test_dir}/*")])]
    test_image_paths = glob(os.path.join(args.test_dir,"*"))
    test_generator = DataGenerator(test_image_paths,
                                dataframe_labels=df,
                                batch_size=BATCH_SIZE,
                                image_size=IMAGE_SIZE)
    if args.custom_model:
        model = build_unet((*IMAGE_SIZE,3))
    else:
        model = sm.Unet()
    model.load_weights(args.weights)
    model.compile('adam',loss=dice_loss)
    results = model.evaluate(test_generator)
    print(results)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='UNet Training Parameters')
    # Add the command line arguments
    parser.add_argument('--custom_model', type=bool, default=False, help='Use custom Unet')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--image_size', type=int, nargs=2, required=True, metavar=('width', 'height'),
                        help='Image size for an input layer in a network')
    parser.add_argument('--weights', type=str, required=True,help='Filepath to load weights')
    parser.add_argument('--dataframe_labels_path', type=str, 
                        default="./data/train_ship_segmentations_v2.csv",
                        help="Path to the dataframe with encoded masks")
    parser.add_argument('--test_dir',type=str,required=True,
                        help="Directory with image for test")
    #  Parse the arguments
    args = parser.parse_args()
    main(args)
