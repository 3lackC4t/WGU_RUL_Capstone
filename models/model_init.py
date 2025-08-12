import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.autoencoder import AutoEncoder
from models.bi_lstm import Bi_LSTM
from models.preprocessor import SensorPreprocessor


def check_if_init(settings):
    return (
        settings.AUTO_ENCODER_PATH.exists() and
        settings.BI_LSTM_PATH.exists() and
        settings.ENCODER_PATH.exists()
    )


def initial_build(settings):
    sensor_preprocessor = SensorPreprocessor(settings.TEST_PATHS)
    sensor_dict = sensor_preprocessor.test_data
    sensor_data = [sensor_dict[f'test_{i + 1}']['sensor_data'] for i in range(len(sensor_dict.keys()))]

    all_sensor_windows = []
    all_degradation_windows = []

    for test in sensor_data:
        sensor_column_names = [col for col in test.columns if col.startswith('sensor_')]
        degradation_values = test['degradation'].values
        degradation_windows = sensor_preprocessor.get_target_windows(degradation_values)

        for sensor_col in sensor_column_names:
            sensor_values = test[sensor_col].values

            sensor_windows = sensor_preprocessor.get_feature_windows(sensor_values) 
            min_windows = min(len(sensor_windows), len(degradation_windows))

            all_sensor_windows.extend(sensor_windows[:min_windows])
            all_degradation_windows.extend(degradation_windows[:min_windows])

    X_all = np.array(all_sensor_windows)
    print("="*60)
    print(f"X_all shape: {X_all.shape}")
    print("="*60)
    y_all = np.array(all_degradation_windows) 
    print("="*60)
    print(f"y_all shape: {y_all.shape}")
    print("="*60)
    
    if not settings.AUTO_ENCODER_PATH.exists() or  not settings.ENCODER_PATH.exists():
        new_autoencoder = AutoEncoder()
        new_autoencoder.fit_autoencoder(
            X_all, X_all,
            settings.AUTO_ENCODER_PATH,
            settings.ENCODER_PATH
        )
    else:
        new_autoencoder = AutoEncoder(settings.AUTO_ENCODER_PATH, settings.ENCODER_PATH)

    auto_encoder, encoder = new_autoencoder.autoencoder, new_autoencoder.encoder

    features = encoder.predict(X_all)
    print("="*60)
    print(f"Encoded features shape: {features.shape}")
    print("="*60)

    windows_features = features
    windows_targets = y_all

    if len(windows_features.shape) == 2:
        windows_features = windows_features.reshape(windows_features.shape[0], 1, windows_features.shape[1]) 

    X_train, X_test, y_train, y_test = train_test_split(
        windows_features, windows_targets, test_size=0.2, shuffle=True
    )

    if not settings.BI_LSTM_PATH.exists():
        new_bi_lstm = Bi_LSTM()
        new_bi_lstm.fit_bi_lstm(X_train, y_train, X_test, y_test, settings.BI_LSTM_PATH)
        bi_lstm = new_bi_lstm.model
    else:
        new_bi_lstm = Bi_LSTM(settings.BI_LSTM_PATH)
        bi_lstm = new_bi_lstm.model()

    return sensor_preprocessor, bi_lstm, encoder

def get_models(settings):
    if not check_if_init(settings):
        preproccesor, bi_lstm, encoder = initial_build(settings)
        return preproccesor, bi_lstm, encoder
    else:
        new_autoencoder = AutoEncoder(settings.AUTO_ENCODER_PATH, settings.ENCODER_PATH)
        new_bi_lstm = Bi_LSTM(settings.BI_LSTM_PATH)
         
        encoder =  new_autoencoder.encoder
        bi_lstm = new_bi_lstm.model
        preproccesor = SensorPreprocessor(settings.TEST_PATHS)
        return preproccesor, bi_lstm, encoder
