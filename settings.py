from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # File Paths
    FILE_PATH = Path(__file__).parent.absolute()
    MODEL_DATA_PATH = FILE_PATH / "models" / "model_data"
    BI_LSTM_PATH = MODEL_DATA_PATH / "bi_lstm.m5"
    AUTO_ENCODER_PATH = MODEL_DATA_PATH / "autoencoder.m5"
    ENCODER_PATH = MODEL_DATA_PATH / "encoder.m5"
    DATA_PATH = FILE_PATH / "bearing_data"
