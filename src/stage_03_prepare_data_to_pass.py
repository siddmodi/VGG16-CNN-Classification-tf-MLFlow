import logging
import os
import argparse
from src.utils.common import create_directories, read_yaml
import tensorflow as tf
import os
from pathlib import Path
import urllib.request as request
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile


STAGE = "stage_03_prepare_data_to_pass"


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)

    try:
        BATCH_SIZE = config['params']['BATCH_SIZE']
        pixels = config['params']['pixels']
        IMAGE_SIZE = (pixels,pixels)
        logging.info(f'batch size : {BATCH_SIZE} , pixels : {pixels} , img size : {IMAGE_SIZE}')


        logging.info('Image Data generation stage running........')
        datagen_kwargs = dict(
                            rescale = config['params']['rescale'],
                            validation_split=config['params']['validation_split'])

        dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        valid_generator = valid_datagen.flow_from_directory(
                                        config['Data']['main_data_dir'],
                                        subset="validation", shuffle=False,
                                        **dataflow_kwargs)  
        logging.info('Image Data generation stage completed.')


        do_data_augmentation = config['params']['augmentation']
        logging.info(f'Data augmentation is {do_data_augmentation}')

        if do_data_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotational_range=40,
                horizontal_flip=True,
                width_shift_range=0.2, 
                height_shift_range=0.2, 
                shear_range=0.2, 
                zoom_range=0.2, 
                **datagen_kwargs)
            logging.info('Data augmented')
        else:
            train_datagen = valid_datagen

        train_generator = train_datagen.flow_from_directory(
                                            config['Data']['main_data_dir'],
                                            subset="training", shuffle=True,
                                            **dataflow_kwargs)
        logging.info(f'returning train_generator : {train_generator}')

        print(f'train_generator hai ye ------------->> {train_generator}')
        print(f'valid_generator hai ye ------------->> {valid_generator}')
        return train_generator , valid_generator

    except Exception as e:
        logging.exception(f'Error Occured : {e}')
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info("\n********************")
        logging.info(f">>>>> {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e