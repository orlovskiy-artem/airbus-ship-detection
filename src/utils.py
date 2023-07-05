import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_images(*images):
    "side tool to plot images in a row in jupyter"
    fig, axs = plt.subplots(1,len(images))
    if len(images)==1:
        axs.imshow(images[0])
        return
    for i, image in enumerate(images):
        axs[i].imshow(image)
        
def rle_to_mask(rle_code, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    code = rle_code.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (code[0:][::2], code[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 255
    return mask.reshape(shape).T  # Needed to align to RLE direction        

def show_segmented(img,mask):
    "show only segmented side with comparison to original"
    segmented = cv2.bitwise_and(img,img,mask=mask)
    plot_images(img,segmented)


def rle_codes_to_mask(rle_codes,image_size):
    mask = np.zeros(image_size,dtype=np.uint8)
    for rle_code in rle_codes:
        mask_ship = rle_to_mask(rle_code)
        mask = cv2.bitwise_or(mask,mask_ship)
    return mask