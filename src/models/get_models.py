#libraries
import pickle
import tensorflow as tf 

#function to load trained neural network
def get_NN():
    return tf.keras.models.load_model('models/nn_combined_original_livecamera.h5')
    