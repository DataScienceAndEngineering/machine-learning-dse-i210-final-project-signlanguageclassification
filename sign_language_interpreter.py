# imports
from src.data import capture_video
from src.features import features_pipeline
from sklearn.pipeline import Pipeline
from src.models import get_models
import cv2 as cv
import numpy as np
import click


@click.command()
@click.option('--model', type=click.Choice(['NN-224x244', 'NN-28x28', 'PLACEHOLDER']), default='NN', required=True)
# main entry for sign_language_interpreter functionality
def main(model):
    if model == 'NN-224x244':
        print('running code for neural network')
        nn = get_models.get_NN224x224()
        capture_video.sign_interpreter(nn, model)
    elif model == 'NN-28x28':
        nn = get_models.get_NN28x28()
        capture_video.sign_interpreter(nn, model)

    else:
        print('running code for second best model')
        pipeline = Pipeline(
            [('scaler', features_pipeline.get_sc()), ('lda', features_pipeline.get_lda())])
        capture_video.sign_interpreter(pipeline)


if __name__ == '__main__':
    main()
