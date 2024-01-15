import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import Model

from PIL import Image
import random
import requests
import cv2

np.random.seed(0)

num_of_samples = []
cols = 5
num_classes = 10
num_pixels = 784

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    model = leNet_model()
    analyze_model(model, X_test, y_test, X_train, y_train)


def analyze_model(model, X_test, y_test, X_train, y_train):
    print(model.summary())
    plot_loss(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    img = test_model_with_sample_image(model)
    visualize_cnn_output(model, img)


def visualize_cnn_output(model, img):
    layer1 = Model(inputs = model.layers[0].input, outputs = model.layers[0].output)
    layer2 = Model(inputs = model.layers[0].input, outputs = model.layers[2].output)
    visual_layer1 = layer1.predict(img)
    visual_layer2 = layer2.predict(img)
    plt.figure(figsize=(10, 6))
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
        plt.axis('off')
    plt.show()
    plt.figure(figsize=(10, 6))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap('jet'))
        plt.axis('off')
    plt.show()



def test_model_with_sample_image(model):
    url = 'https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png'
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    plt.imshow(img)
    plt.show()

    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28,28))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(gray_scale)
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()

    image = image/255
    image = image.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(image), axis=-1)
    print("Predicted digit: ", str(prediction))
    return image


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


def plot_loss(model, X_train, y_train):
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size = 200, verbose=1, shuffle=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'validation_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()


def preprocess_data(X_train, y_train, X_test, y_test):
    check_all_images_have_labels(X_train, y_train, X_test, y_test)
    check_image_dimensions(X_train, X_test)
    plot_five_images_from_each_class(X_train, y_train)
    plot_distribution_of_training_dataset()
    y_train, y_test = one_hot_encode_labels(y_train, y_test)
    X_train, X_test = scale_down_variance(X_train, X_test)
    X_train, X_test = reshape_images(X_train, X_test)
    return X_train, X_test, y_train, y_test


def reshape_images(X_train, X_test):
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    return X_train, X_test


def scale_down_variance(X_train, X_test):
    # Scale down the variance
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test


def one_hot_encode_labels(y_train, y_test):
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return y_train, y_test


def plot_distribution_of_training_dataset():
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

def plot_five_images_from_each_class(X_train, y_train):
    fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize=(5,10))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = X_train[y_train==j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), : :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
    plt.show()


def check_image_dimensions(X_train, X_test):
    assert(X_train.shape[1:] == (28,28)), "The dimensions of the training images are not 28x28"
    assert(X_test.shape[1:] == (28,28)), "The dimensions of the test images are not 28x28"


def check_all_images_have_labels(X_train, y_train, X_test, y_test):
    assert(X_train.shape[0] == y_train.shape[0]), "The number of images in the training set does not match the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images in the test set does not match the number of labels"


def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
