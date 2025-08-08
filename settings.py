from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # File Paths
    FILE_PATH = Path(__file__).parent.absolute()
    MODEL_DATA_PATH = FILE_PATH / "models" / "model_data"
    BI_LSTM_PATH = MODEL_DATA_PATH / "bi_lstm.keras"
    AUTO_ENCODER_PATH = MODEL_DATA_PATH / "autoencoder.keras"
    ENCODER_PATH = MODEL_DATA_PATH / "encoder.keras"
    DATA_PATH = FILE_PATH / "bearing_data"
    TEST_PATHS = [
        DATA_PATH / "1st_test" / "1st_test",
        DATA_PATH / "2nd_test" / "2nd_test",
        DATA_PATH / "3rd_test" / "4th_test" / "txt"
    ]
