# libraries
import pickle

# function for loading and returning pickle object


def load_pickle(path):
    # loading pickle object
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# function to get trained lda object


def get_lda():
    lda = load_pickle('models/trained_lda.pkl')
    return lda

# function to get trained standard scaler object


def get_sc():
    sc = load_pickle('models/trained_standard_scaler.pkl')
    return sc


def lda_transform(lda, img):
    return lda.transform(img.reshape[0], -1)
