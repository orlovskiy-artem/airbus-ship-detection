import argparse
import cv2
import numpy as np
from PIL import Image  
import segmentation_models as sm
sm.set_framework('tf.keras')

from src.models import build_unet


def load_image(image_path:str):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

def save_segmentation(mask:np.ndarray, output_path:str):
    mask = mask>0.5
    mask = mask.astype(np.uint8)*255
    if mask.shape==3 and mask.shape[-1]==1:
        mask = mask[:,:,0]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(output_path,mask)

def main(args):
    if args.custom_model:
        model = build_unet((*args.image_size,3))
    else:
        model = sm.Unet()
    model.load_weights(args.weights)
    image = load_image(args.image_path)
    # Perform inference. 
    # Image needs to be transformed to 4d tensor (batch of size 1)
    output = model.predict(image[None,...])
    # Save the segmentation mask
    # segmentation back to squeezed type (only data)
    save_segmentation(output[0], args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation Inference')
    parser.add_argument('--image_path', type=str, required=True, help='path to the input image')
    parser.add_argument('--custom_model', type=bool, default=False, help='Use custom Unet')
    parser.add_argument('--weights', type=str, required=True, help='path to the trained model weights')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the segmentation mask')
    args = parser.parse_args()
    main(args)