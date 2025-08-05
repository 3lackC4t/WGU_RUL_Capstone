from models.autoencoder import LSTM_AutoEncoder
from models.bi_lstm import Bi_LSTM
from models.preprocessor import Preprocessor
from API.model_API import *
from time import sleep
from pathlib import Path
import threading
from dataclasses import dataclass

@dataclass
class Settings:
    FILE_PATH = Path(__file__).parent.absolute()
    MODEL_DATA_PATH = FILE_PATH / "models" / "model_data"
    DATA_PATH = FILE_PATH / "bearing_data" / "archive"


def get_features(data):
    pass

def predict_on_data(features):
    # take in features
    # use preprocessor to turn into windows
    # take required windows, predict on it
    # Return it
    # wait 1 second
    # repeat 
    pass

def main():
    # Bring in data
    app_settings = Settings()
    paths = [
        app_settings.DATA_PATH / "1st_test" / "1st_test",
        app_settings.DATA_PATH / "2nd_test" / "2nd_test",
        app_settings.DATA_PATH / "3rd_test" / "4th_test" / "txt"
    ]
    preprocessor = Preprocessor(paths)
    # tests, scaler = preprocessor.run_pipeline()

    auto_encoder = LSTM_AutoEncoder()
    autoencoder = auto_encoder.autoencoder
    encoder = auto_encoder.encoder
    bi_lstm = Bi_LSTM()

    

    # check if autoencoder exists

    # if not create autoencoder and train autoencoder

    # check if Bi-LSTM exists

    # if not create and train Bi-LSTM

    # run API

    # Main loop is to get data 


if __name__ == "__main__":
    main()