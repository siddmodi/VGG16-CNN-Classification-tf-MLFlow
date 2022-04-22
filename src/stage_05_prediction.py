from importlib.resources import path
import logging
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src import stage_01_get_data 
from src.utils.common import read_yaml, create_directories
from src import stage_03_prepare_data_to_pass
from PIL import Image

STAGE = "stage_05_prediction"

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

        model_path = os.path.join(config['Data']['model_dir'],config['Data']['trained_model'])
        model = tf.keras.models.load_model(model_path)
        logging.info(f'model loaded succesfully : {model}')

        label_map = {val: key for key, val in train_generator.class_indices.items()}
        logging.info(f'label_map : {label_map}')

        input_path_test_img = input('Enter full path of image for prediction : ')
        test_img = plt.imread(input_path_test_img)
        resized_img = tf.image.resize(test_img, (224, 224))
        input_data = tf.expand_dims(resized_img, axis=0)

        pred = model.predict(input_data)
        argmax = tf.argmax(pred[0]).numpy()
        im = Image.open(input_path_test_img)
        im.show()
        print(f'predicted is : {label_map[argmax]}')    
        logging.info(f'predicted is : {label_map[argmax]}')
    
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
