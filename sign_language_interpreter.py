# imports
from src.data import capture_video
from src.features import features_pipeline
from sklearn.pipeline import Pipeline
import cv2 as cv
import numpy as np
import click


@click.command()
@click.option('--model', type=click.Choice(['NN', 'PLACEHOLDER']), default='NN', required=True)
# main entry for sign_language_interpreter functionality
def main(model):

    if model == 'NN':
        print('running code for neural network')
        capture_video.sign_interpreter()

    else:
        print('running code for second best model')
        pipeline = Pipeline(
            [('scaler', features_pipeline.get_sc()), ('lda', features_pipeline.get_lda())])
        capture_video.sign_interpreter(pipeline)


if __name__ == '__main__':
    main()
