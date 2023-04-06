# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import matplotlib.pyplot as plt 
import numpy as np
import re 
from tqdm import tqdm 
from functools import reduce 


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    special_labels = {'space':0,'nothing':1,'del':2}

    with open(output_filepath,'a') as file:

        for subdir, dirs, files in os.walk(input_filepath):
            print(subdir)
            if len(dirs) != 0:
                continue

            for img in tqdm(files):
                image = plt.imread(subdir+'/'+img)
                if image.shape[2] == 3:
                    image = image.reshape(-1)
                label = re.search('^(.*?)[^a-zA-Z]',img).group(0)[:-1]
                if len(label)==1:
                    label = ord(label)
                else:
                    label = special_labels[label]

                file.write(reduce(lambda x,y:str(x)+','+str(y),image,label)+'\n')
            break
    #logger = logging.getLogger(__name__)
    #logger.info('making final data set from raw data')


if __name__ == '__main__':
    '''
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    '''
    main()
