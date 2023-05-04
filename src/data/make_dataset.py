#libraries 
import click
import kaggle
import os 
import numpy as np 
import pickle 
import tensorflow as tf 
import logging 
import tempfile
from sklearn.utils import shuffle

#function for finding file
def find_file(name,path):
    for root, dir, files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

#function for applying image augmentation 
def data_aug(datagen,X,y):
    #define empty lists to hold augmented data 
    X_augmented = []
    y_augmented = []
    for i in range(X.shape[0]):
        img = X[i]
        label = y[i]
        for j in range(3):
            X_augmented.append(datagen.random_transform(img))
            y_augmented.append(label)
    X_augmented = np.array(X_augmented)

    return X_augmented.reshape(X_augmented.shape[0],28,28),np.array(y_augmented)


@click.command()
@click.argument('output_filepath', type=click.Path(exists=True))
def main(output_filepath):
    #CONSTANTS 
    #image resolution 
    res = (28,28)

    #setting level to info to display logging messages
    logging.basicConfig(level = logging.INFO)
        
    #authenticate kaggle account 
    kaggle.api.authenticate()

    #create temp directory to hold downloaded data while computations finish, then release
    with tempfile.TemporaryDirectory() as tmp_dir:
        #temp folder created 
        logging.info('Temp folder created at %s',tmp_dir)

        #download kaggle dataset, and temporarily drop into folder 
        kaggle.api.dataset_download_files('datamunge/sign-language-mnist',path=tmp_dir,unzip=True)
        logging.info('Kaggle data downloaded into temp folder')
    
        #find train and test files 
        train_path = find_file('sign_mnist_train.csv',tmp_dir)
        test_path = find_file('sign_mnist_test.csv',tmp_dir)

        #load csv data of images into numpy arrays 
        train = np.loadtxt(train_path, delimiter=',', skiprows=1)
        test = np.loadtxt(test_path, delimiter=',', skiprows=1)

        #split X and y data
        #training data
        X_train = train[:,1:]
        X_train = X_train.reshape(X_train.shape[0], res[0], res[1], 1)
        y_train = train[:,0]

        #testing data 
        X_test = test[:,1:]
        X_test = X_test.reshape(X_test.shape[0], res[0], res[1], 1)
        y_test = test[:,0]

        #create ImageDataGenerator object with defined data augmentation parameters
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,brightness_range=[0.5, 1.5],fill_mode='nearest')

        # Apply data augmentation to the training set
        logging.info('Augmentation in progress')
        X_train_augmented, y_train_augmented = data_aug(datagen,X_train,y_train)

        # Apply data augmentation to the test set
        X_test_augmented, y_test_augmented = data_aug(datagen,X_test,y_test)

        #reshaping the original numpy arrays to combine with augmented data 
        X_train = X_train.reshape(X_train.shape[0],res[0],res[1])
        X_test = X_test.reshape(X_test.shape[0],res[0],res[1])

        #concatenate the arrays along the first axis (row-wise)
        X_train_combined = np.concatenate((X_train_augmented, X_train), axis=0)
        y_train_combined = np.concatenate((y_train_augmented, y_train), axis=0)

        X_test_combined = np.concatenate((X_test_augmented, X_test), axis=0)
        y_test_combined = np.concatenate((y_test_augmented, y_test), axis=0)

        #shuffling data
        X_train_combined, y_train_combined = shuffle(X_train_combined,y_train_combined,random_state=99)
        X_test_combined, y_test_combined = shuffle(X_test_combined,y_test_combined,random_state=99)

        #add combined datasets to tuple in preparation for pickling 
        combined_augmented_data = (X_train_combined,y_train_combined,X_test_combined,y_test_combined)

        #pickling 
        with open(os.path.join(output_filepath,'combined_augmented_data_v3.pkl'),'wb') as f:
            pickle.dump(combined_augmented_data, f)
            logging.info('Pickle of original image dataset and augmented dataset dumped into %s',output_filepath)

#entry 
if __name__ == '__main__':

    main()
