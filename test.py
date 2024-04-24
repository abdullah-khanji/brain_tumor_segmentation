import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import numpy as np, cv2, pandas as pd, tensorflow as tf
from glob import glob
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from metrics import dice_loss, dice_coef
from train import load_dataset
from unet import build_unet
from matplotlib import pyplot as plt


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def segment(image, mask, color):
    kernel= np.ones((3, 3), np.uint8)
    dilated_mask= cv2.dilate(mask, kernel, iterations=1)
    border= dilated_mask-mask
    #color
    mask_color= cv2.cvtColor(y_pred, cv2.COLOR_GRAY2BGR)
    mask_color[border>0]= color
    alpha=0.5
    overlay_image= cv2.addWeighted(image, 1-alpha, mask_color, alpha, 0)
    overlay_image= cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    return overlay_image

def save_results(image, mask, y_pred, save_image_path):
    # Create a border around the mask
    
    print(image.shape, mask.shape, y_pred.shape,'------------')
    kernel = np.ones((3,3), np.uint8)  # Adjust size for a thicker/thinner border
    dilated_mask = cv2.dilate(y_pred, kernel, iterations=1)
    border = dilated_mask - y_pred
    # Colorize the mask
    mask_color = cv2.cvtColor(y_pred, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR
    mask_color[border > 0] = [0, 255, 0]  # Green color for the border

    # Blend the colored mask with the original image
    alpha = 0.5  # Transparency factor (between 0 and 1)
    overlay_image = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

    # Convert image to RGB for Matplotlib
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(overlay_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    
    mask= np.expand_dims(mask, axis=-1)
    y_pred= np.expand_dims(y_pred, axis=-1)


    mask= np.concatenate([mask, mask, mask], axis=-1)
    y_pred=np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred= y_pred*255
    line=np.ones((128, 5, 3))*255

    result=np.concatenate([image, line, mask, line,y_pred], axis=1)
    cv2.imwrite(save_image_path, result)

if __name__=='__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results")

    with CustomObjectScope({"dice_coef":dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))
    
    model.summary()

    dataset_path="./data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y)= load_dataset(dataset_path)

    segmented_y_pred=[]
    segmented_actual=[]
    for j in range(12):
        img=cv2.imread(test_x[j])
        img= cv2.resize(img, (128, 128))
        x= img/255.0
        x= np.expand_dims(x, axis=0)

        # y_pred= model.predict(x, verbose=0)[0]
        #     y_pred= np.squeeze(y_pred, -1)
        #     y_pred= y_pred>=0.5
        #     y_pred= (y_pred * 255).astype(np.uint8)
        y_pred= model.predict(x, verbose=0)[0]
        y_pred= np.squeeze(y_pred, axis=-1)
        y_pred= y_pred>=0.5
        y_pred= (y_pred*255).astype(np.uint8)
        segmented_pred= segment(img, y_pred, [0, 255, 0]) 
        #actual mask
        mask= cv2.imread(test_y[j], cv2.IMREAD_GRAYSCALE)
        mask= cv2.resize(mask, (128, 128))
        mask= (mask).astype(np.uint8)
        segmented_truth= segment(img, mask, [255, 0, 0])
        #append for showing
        segmented_y_pred.append(segmented_pred)
        segmented_actual.append(segmented_truth)
    
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))  # Adjust size as needed

    # Plot the first 6 predicted images in row 1
    for i in range(6):
        axes[0, i].imshow(segmented_y_pred[i])
        axes[0, i].set_title("Predicted " + str(i+1))
        axes[0, i].axis("off")  # Hide the axes

    # Plot the first 6 actual images in row 2
    for i in range(6):
        axes[1, i].imshow(segmented_actual[i])
        axes[1, i].set_title("Actual " + str(i+1))
        axes[1, i].axis("off")  # Hide the axes

    # Plot the next 6 predicted images in row 3
    for i in range(6, 12):
        axes[2, i-6].imshow(segmented_y_pred[i])
        axes[2, i-6].set_title("Predicted " + str(i+1))
        axes[2, i-6].axis("off")  # Hide the axes

    # Plot the next 6 actual images in row 4
    for i in range(6, 12):
        axes[3, i-6].imshow(segmented_actual[i])
        axes[3, i-6].set_title("Actual " + str(i+1))
        axes[3, i-6].axis("off")  # Hide the axes

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


    # SCORE=[]

    # for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
    #     name = os.path.basename(x)

    #     print(name)
    #     image= cv2.imread(x, cv2.IMREAD_COLOR)
    #     image= cv2.resize(image, (128, 128))
    #     x1= image/255.0

    #     x= np.expand_dims(x1, axis=0)

    #     mask= cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    #     mask= cv2.resize(mask, (128, 128))
        
    #     y_pred= model.predict(x, verbose=0)[0]
    #     y_pred= np.squeeze(y_pred, -1)
    #     y_pred= y_pred>=0.5
    #     y_pred= (y_pred * 255).astype(np.uint8)
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust size as needed

    #     axes[0].imshow(x1)
    #     axes[0].set_title("Image 1")
    #     axes[0].axis("off")  # Hide the axes

    #     axes[1].imshow(mask)
    #     axes[1].set_title("Mask")
    #     axes[1].axis("off")  # Hide the axes

    #     axes[2].imshow(y_pred)
    #     axes[2].set_title("Predicted Mask")
    #     axes[2].axis("off")  # Hide the axes

    #     plt.show()

    #     saving_image_path= os.path.join("results", name)
    #     save_results(image, mask, y_pred, saving_image_path)
    #     break