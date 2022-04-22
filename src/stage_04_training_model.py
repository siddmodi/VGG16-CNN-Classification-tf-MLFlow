from calendar import EPOCH
from cgi import test
import logging
import os
import numpy as np
from flask import Config
# import torch.nn as nn
import argparse
from src import stage_01_get_data
# from src.stage_03_base_model_creation import CNN
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from src import stage_02_base_model_creation
from src import stage_03_prepare_data_to_pass
# from tf.keras.optimizers import SGD
# from tf.keras.losses import CategoricalCrossentropy


STAGE = "stage_04_training_model"


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)

    try:
        train_generator , valid_generator = stage_03_prepare_data_to_pass.main(config_path)

        RGB_IMAGE_SIZE = (config['params']['pixels'],
                        config['params']['pixels'], 3)

        vgg = tf.keras.applications.vgg16.VGG16(
                                input_shape=RGB_IMAGE_SIZE,
                                weights="imagenet",
                                include_top=False   
                                )

        logging.info(vgg.summary())

        for layer in vgg.layers:
            # print(layer.name)
            print(f"{layer.name:20s}: {layer.trainable}")

        for layer in vgg.layers:
            layer.trainable = False

        for layer in vgg.layers:
            print(f"{layer.name:20s}: {layer.trainable}")

        CLASSES = config['params']['classes']
        x = tf.keras.layers.Flatten()(vgg.output)
        prediction = tf.keras.layers.Dense(CLASSES, activation=config['params']['activation_function'])(x)

        model = tf.keras.models.Model(inputs=vgg.input, outputs = prediction)
        logging.info(model.summary())
        
        model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate = config['params']['learning_rate'],
                                                    momentum=config['params']['momentum']),
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics=config['params']['metrics']
                    )
        logging.info('Model compilation done')

        EPOCHS = config['params']['epochs_for_training']

        logging.info('Training started...............')
        history = model.fit(
                        train_generator,
                        epochs=config['params']['epochs'],
                        validation_data=valid_generator)
        logging.info('Training succesfully done')

        trained_model_file = config['Data']['trained_model']
        trained_model_path = os.path.join(config['Data']['model_dir'],trained_model_file )

        model.save(trained_model_path)
        logging.info(f'Model saved at {trained_model_path}')

    except Exception as e:
        logging.exception(f'Error Occured : {e}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e