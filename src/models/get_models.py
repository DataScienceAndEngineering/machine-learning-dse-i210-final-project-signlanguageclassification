# libraries
import pickle
import tensorflow as tf

from cvzone.ClassificationModule import Classifier


# function to load trained neural network
def get_NN28x28():
    return tf.keras.models.load_model('./models/nn_combined_original_livecamera_augmented.h5')


def get_NN224x224():
    return Classifier('./models/custom_cnn_model.h5', './models/labels.txt')

