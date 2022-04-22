import os
import logging
from numpy import full
import argparse
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
from keras.applications.vgg16 import VGG16

STAGE = "stage_02_base_model_creation"


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)

    IMAGE_SIZE = (224, 224)

    try:
        model = tf.keras.applications.VGG16(
                                    include_top=True, weights="imagenet", 
                                    input_tensor=None, input_shape=None, classes=1000)

        logging.info(model.summary())

        model_dir = config['Data']['model_dir']
        create_directories([model_dir])    
        logging.info(f"{model_dir} folder created") 

        base_model_file = config['Data']['base_model']
        base_model_path = os.path.join(model_dir,base_model_file )

        model.save(base_model_path)
        logging.info(f'Model saved at {base_model_path}')
    
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