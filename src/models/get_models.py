# libraries
import pickle
import tensorflow as tf

from cvzone.ClassificationModule import Classifier


# function to load trained neural network
def get_NN28x28():
    return tf.keras.models.load_model('src/models/my_cnn_model_updated.h5')


def get_NN224x224():
    return Classifier('src/models/my_cnn_model_updated.h5', 'src/models/labels.txt')

