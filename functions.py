import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing import image
from keras import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score



def load_data(dir_path: str = "brain_tumor_dataset",
              IMAGE_SHAPE: tuple = (224, 224),
              random_sample: int = 1,
              verbose: bool = True):
    
    """Load the image data from a directory, and select the relevant test data. 

    Args:
        dir_path (str): The name of the folder containing image dataset.
        IMAGE_SHAPE (tuple): Pixel size of images
        random_sample (int): Random sample used to ensure correct test data is used
        verbose (bool): Print information about the data. 

    Returns:
        files, labels and images of test data
    """

    files = []
    ground_truth = []
    labels = []
    images = []

    # Read the folders folders
    directories = os.listdir(dir_path)

    # Read files for each directory
    for folder in directories:
        
        fileList = glob.glob(f'{dir_path}/{folder}/*')
        ground_truth.extend([folder for _ in fileList])
        files.extend(fileList)

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(ground_truth)
    encoded_Y = encoder.transform(ground_truth)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = utils.to_categorical(encoded_Y,dtype ="uint8", num_classes = 2)

    ground_truth = np.array(dummy_y)

    # Split the data into testing, training and validation data
    # As using a random sample = 1, the testing set is the same as used for the model.
    x_train, x_tv, y_train, y_tv = train_test_split(files, ground_truth, test_size=.3, random_state = random_sample)
    x_test, x_val, y_test, y_val = train_test_split(x_tv, y_tv, test_size=.5, random_state = random_sample)

    for file, label in zip(x_test, y_test):

        # Prepare the image
        new_file = image.load_img(file, target_size=IMAGE_SHAPE)
        img_array = image.img_to_array(new_file)
        # Append to array
        images.append(img_array)
        labels.append(label)

    if verbose:
        print(f"Classes are {encoder.classes_}")

    print(labels[0:5])

    for img in images[0:5]:
        img = image.array_to_img(img)
        plt.imshow(img)
        plt.show()



    return files, labels, images



def make_predictions(images, model, temp_model):
    """Makes predictions for the class of an image

    Args:
        images (list): List containing array of image arrays 
        model: Full model to make predictions
        temp_model: Model tht returns final hidden layer activations

    Returns:
        final_layers (list): List of activations of the final hidden layer
        predictions (list): List of predictions
        pred_array (List): List of arg-maxed predictions
    """

    final_layers = temp_model.predict(np.stack(images))
    predictions = model.predict(np.stack(images))
    pred_array = [[0,1][np.argmax(individual_result)] for individual_result in predictions]

    print(predictions[0:5])
    print(pred_array[0:5])

    return final_layers, predictions, pred_array


def create_mixup_images(class_zero_df,
                        class_one_df,
                        images,
                        MUBA_ITERS = 60,
                        ):

    muba_df = pd.DataFrame(columns = [
                        "alpha_class_0",
                        "alpha_class_1",
                        "image_0",
                        "image_1",
                        "mixup_image",
                        "type",
                        "label",
        ])

    # Iterate through NO tumour images
    for index0, row0 in class_zero_df.iterrows():
        
        # Iterate through YES tumour images
        for index1, row1 in class_one_df.iterrows():

            for i in range(MUBA_ITERS):
                
                # Mixup images
                lam = (1/MUBA_ITERS) * np.random.rand() + ( (i) / MUBA_ITERS)
                new_img = lam * images[int(row0["image_index"])] + (1 - lam) * images[int(row1["image_index"])]

                # lam = proportion no tumour
                label = 1
                if lam > 0.5:
                    label = 0

                row = pd.DataFrame([({ "alpha_class_0": lam, # row 0 is multiplied by lam
                        "alpha_class_1": 1-lam,
                        "image_0": images[int(row0["image_index"])],
                        "image_1": images[int(row1["image_index"])],
                        "mixup_image": new_img,
                        "type":"mix",
                        "label":label
                        })])
                
                
                muba_df = pd.concat([muba_df, row],axis=0, ignore_index=True)

    return muba_df


def find_boundary_points(muba_df,
                         MUBA_ITERS: int = 60):
    """Generates new images with alpha values between those at which a model changes it's prediction

    Args:
        muba_df (DataFrame): DataFrame containing all mixed up images
        MUBA_ITERS (int, optional): _description_. Defaults to 60.

    Returns:
        boundary_points_df (DataFrame): Contains all boundary points
    """

    boundary_points_df = pd.DataFrame(columns = [
                    "alpha_class_0",
                    "alpha_class_1",
                    "image_0",
                    "image_1",
                    "mixup_image",
                    "type",
                    "label",
    ])

    for i in range(int((len(muba_df))/MUBA_ITERS)):

        # Create a mask to split df in to blocks of MUBA_ITERS
        mask = (muba_df.index >= MUBA_ITERS*i) & (muba_df.index < MUBA_ITERS*i + MUBA_ITERS)
        window_df = muba_df.loc[mask]

        # Find the index in which the prediction changes
        changing_pred_index = (window_df["argmax_pred"].diff()[window_df["argmax_pred"].diff() != 0].index.values)
        for index, row in window_df.iterrows():
            
            if index in changing_pred_index[1:]:
                
                row0 = window_df.loc[[index]]
                row1 = window_df.loc[[index-1]]


                alpha_class_0 = ( float(row0["alpha_class_0"]) + float(row1["alpha_class_0"])) / 2
                alpha_class_1 = 1 - alpha_class_0
                mixup_image = (alpha_class_0 * row0["image_0"] + alpha_class_1 * row0["image_1"])[index]
                
                label = 1
                if alpha_class_0 > 0.5:
                    label = 0

                row = pd.DataFrame([({ "alpha_class_0":alpha_class_0,
                        "alpha_class_1": alpha_class_1,
                        "image_0": row0["image_0"],
                        "image_1": row0["image_1"],
                        "mixup_image": mixup_image,
                        "type":"boundary",
                        "label":label
                        })])
                
                boundary_points_df = pd.concat([boundary_points_df, row],axis=0, ignore_index=True)
            
    return boundary_points_df


