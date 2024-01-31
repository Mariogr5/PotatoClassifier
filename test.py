from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def test_model(nn_name, image_path):
    model = load_model(nn_name)
    test_image = image.load_img(image_path, target_size=(32, 32), color_mode='rgb')

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(
        './train_data/',
        target_size=(32, 32),
        color_mode="rgb",
        batch_size=32,
        class_mode='categorical')
    my_dict = training_set.class_indices
    pred = list(result[0])
    for i in range(len(pred)):
        if pred[i] != 0:
            print(get_key(i, my_dict))

    image_path = image_path
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test_model("potatoes_disease_detector_test5.h5", "C:/Users/mario/Desktop/kerassieci/Datasheet_base/Blackleg/11.jpg")
