#augment_data.py

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os

def augment_data_gen(dataset, augment_directory,  num_augment_image=3):
    """
    Params:
    dataset: Source Directory containing all image files to augment
    augment_directory: Target directory to hold all augmented images
    num_augment_image: int; specifies number of images to generate per input image, default is 3

    Returns:
         Nothing. But new directory created which holds all augmented images

    Note: this function saves to a particular directory. It is possible to just generate the augmented
    images and train on them directly without saving the images.
    """

    #Ensure the target directory exists
    os.makedirs(augment_directory, exist_ok=True)

    #Define the image generator
    imggen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    orig_files = os.listdir(dataset)

    for file in orig_files:
        img_path = os.path.join(dataset, file)
        img = image.load_img(img_path, target_size=(228, 228))
        x = img.image_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for _ in imggen.flow(x, batch_size=1, save_to_dir=augment_directory, save_prefix='aug',
                                      save_format='jpeg'):
            i = i + 1
            if i >= num_augment_image:
                break

def augment_dataframe(df, batch_size=10, target_size=(228, 228)):
    """
    Params:
    df: dataframe containing images and labels
    batch_size: size of each batch, default is 32
    target_size: image size, default value is (228,228)

    Returns:
    generator: a dataframe iterator
    """
    #Define the image generator
    imggen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    generator = imggen.flow_from_dataframe(dataframe=df, batch_size=batch_size, target_size=target_size, class_mode='raw', x_col='file_name',
                                           y_col='label')
    return generator