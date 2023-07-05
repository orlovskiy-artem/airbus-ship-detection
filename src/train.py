import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob

from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from src.models import build_unet
from src.data_loader import DataGenerator
from src.augmentations import get_training_augmentation
from src.losses import dice_loss, dice_coef, true_positive_rate, tversky,tversky_loss, focal_tversky_loss, focal_loss

SEED = 42

def main(args):
    # Select needed constants. for more flexible control 
    # it is better to use .yaml or .ini train/test config 
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = 1-TRAIN_RATIO
    TRAIN_EPOCHS = args.epochs
    SAMPLE = args.sample
    SAMPLE_SIZE = args.sample_size
    df_train = pd.read_csv(args.dataframe_labels_path)
    # remove bug image
    # df_train = df_train[df_train['ImageId'] != '6384c3e78.jpg']
    # if you downloaded a sample in the data directory: to prevent reading not dowloaded images
    # df_train = df_train[df_train["ImageId"].isin([Path(path).name for path in glob("./data/train_v2/*")])]
    df_train["image_path"] = "./data/train_v2/"+df_train["ImageId"]
    if SAMPLE:
        if SAMPLE_SIZE==-1:
            raise Exception("Not given sample size")
        images_paths = df_train[~df_train['EncodedPixels'].isna()].sample(SAMPLE,random_state=SEED)["image_path"].values
    else:
        images_paths = df_train[~df_train['EncodedPixels'].isna()]["image_path"].values
    train_image_paths, val_image_paths = train_test_split(images_paths,
                                                      train_size=TRAIN_RATIO,
                                                      test_size=VAL_RATIO,
                                                      random_state=SEED)
    train_generator = DataGenerator(train_image_paths,
                                    dataframe_labels=df_train,
                                    batch_size=BATCH_SIZE,
                                    image_size=IMAGE_SIZE,
                                    augmentations=get_training_augmentation())
    val_generator = DataGenerator(val_image_paths,
                                dataframe_labels=df_train,
                                batch_size=BATCH_SIZE,
                                image_size=IMAGE_SIZE)
    fname = os.path.sep.join([args.checkpoint_dir,"weights-{epoch:03d}-{val_loss:.4f}.h5"])
    checkpoint = ModelCheckpoint(fname,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.33,
                                    patience=1,
                                    verbose=1,
                                    mode='min',
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                        patience=20)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0,  
          write_graph=True, write_images=True)

    callbacks_list = [checkpoint, early, tensorboard, reduceLROnPlat]
    model = build_unet((*IMAGE_SIZE,3))

    model.compile(optimizer=keras.optimizers.Adam(1e-4, decay=1e-6), loss=dice_loss,
             metrics=[dice_coef, 'binary_accuracy', true_positive_rate, tversky,tversky_loss,
                      focal_tversky_loss,focal_loss])
    loss_history = model.fit(train_generator, 
                         epochs=TRAIN_EPOCHS, 
                         validation_data=val_generator,
                         callbacks=callbacks_list)
    Path(args.output_weights).parent.mkdir(exist_ok=True,parents=True)
    model.save(args.output_weights)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='UNet Training Parameters')
    # Add the command line arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, nargs=2, required=True, metavar=('width', 'height'),
                        help='Image size for an input layer in a network')
    parser.add_argument('--train_ratio', type=float, default=0.75, help='Training ratio for train-test split')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--sample', type=bool, default=False, help='Flag to enable sampling')
    parser.add_argument('--sample_size', type=int, default=-1, help='Sample size')
    parser.add_argument('--output_weights', type=str, required=True,help='Filepath for weights')
    parser.add_argument('--checkpoint_dir',type=str, required=True,help='Path for checkpoint to save or to train from')
    parser.add_argument('--dataframe_labels_path', type=str, 
                        default="./data/train_ship_segmentations_v2.csv",
                        help="Path to the dataframe with encoded masks")
    #  Parse the arguments
    args = parser.parse_args()
    print(args.image_size)
    main(args)
