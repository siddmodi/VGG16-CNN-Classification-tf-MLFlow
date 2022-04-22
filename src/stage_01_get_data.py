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


STAGE = "stage_01_get_data"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    config = read_yaml(config_path)

    try:
        data_folder_path = config['Data']['root_data_folder']
        create_directories([data_folder_path])         
        logging.info(f"{data_folder_path} folder created") 
    except Exception as e:
        logging.exception(f'Error Occured : {e}')

    data_URL = config['Data']['URL']
    data_zip_file = config['Data']['zip data']
    data_zip_path = os.path.join(data_folder_path, data_zip_file)

    if not os.path.isfile(data_zip_file):
        logging.info("downloading data...")
        filename, headers = request.urlretrieve(data_URL, data_zip_path)
        logging.info(f"filename: {filename} created with info \n{headers}")
    else:
        logging.info(f"file is already present")

    unzip_data_dirname = config['Data']['unzip_data_dirname']
    unzip_data_dir = os.path.join(data_folder_path, unzip_data_dirname)

    if not os.path.exists(unzip_data_dir):
        os.makedirs(unzip_data_dir, exist_ok=True)
        with ZipFile(data_zip_path) as f:
            f.extractall(unzip_data_dir)
        logging.info('Data extracted succesfully')
    else:
        print(f"data already extacted")



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

