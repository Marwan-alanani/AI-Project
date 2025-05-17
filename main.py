import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0


EPOCHS = 30
IMG_WIDTH = 224
IMG_HEIGHT = 224
TEST_SIZE = 0.4
NUM_CATEGORIES = 100
MAP = {}


def main():

    global MAP
    # Load  csv that contains the data into a dataframe object
    df = pd.read_csv('data/sports.csv')
    MAP = map_class_id_to_label(df)

    # Get image arrays and labels for all image files
    images, class_ids = load_data(df)
    class_ids = tf.keras.utils.to_categorical(class_ids)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(class_ids), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    model.save("model.h5")
    print(f"Model saved to {model}.")


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.expand_dims(image, axis=0)
    return image


def map_class_id_to_label(df: pd.DataFrame):
    mapper = {}
    for class_id, label in zip(df['class id'], df['labels']):
        mapper[class_id] = label
    return mapper


def load_data(df: pd.DataFrame):
    """
        loads data from a dataframe that has an image label 
        and the associated image file path
    """
    images = []
    class_ids = []
    for path, class_id in zip(df['filepaths'], df['class id']):
        image = cv2.imread("data/"+path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        images.append(image)
        class_ids.append(class_id)
    return (images, class_ids)


def predict(model, image):
    global MAP
    if len(MAP) == 0:
        MAP = map_class_id_to_label(pd.read_csv('data/sports.csv'))

    predicted_class = np.argmax(model.predict(image)[0])
    print(f"Predicted class is {MAP[predicted_class]}")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    base_model = EfficientNetB0(input_shape=(
        IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NUM_CATEGORIES, activation='softmax')

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def test(image_path, model_path='model.h5'):
    model = load_model(model_path)
    image = get_image(image_path)
    predict(model, image)


if __name__ == "__main__":
    main()
