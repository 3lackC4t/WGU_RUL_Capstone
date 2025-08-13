import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    Scalar = StandardScaler()
    X_all = np.array(all_sensor_windows)
    X_all = Scalar.fit_transform(X_all)
    print(f"X_all.mean: {X_all.mean()}")
    print(f"X_all std: {X_all.std()}")
    print("="*60)
    print(f"X_all shape: {X_all.shape}")
    print("="*60)
    y_all = np.array(all_degradation_windows) 
    print(f"X_all.mean: {X_all.mean()}")
    print(f"X_all std: {X_all.std()}")
    print("="*60)
    print(f"y_all shape: {y_all.shape}")
    print("="*60)
    
    if not settings.AUTO_ENCODER_PATH.exists() or  not settings.ENCODER_PATH.exists():
        new_autoencoder = AutoEncoder(
            input_shape=X_all.shape[1],
            new_model=True, 
            model_path=settings.AUTO_ENCODER_PATH, 
            model_path_encoder=settings.ENCODER_PATH
        )
        new_autoencoder.fit_model(X_all, X_all)
        encoder = new_autoencoder.encoder
    else:
        new_autoencoder = AutoEncoder(new_model=False, model_path=settings.AUTO_ENCODER_PATH, model_path_encoder=settings.ENCODER_PATH)
        encoder = new_autoencoder.encoder

    print("Predicting on encoder input")
    windows_features = encoder.predict(X_all)
    print("="*60)
    print(f"Encoded features shape: {windows_features.shape}")
    print("="*60)

    # Reshape input for CNN 
    if len(windows_features.shape) == 2:
        windows_features = windows_features.reshape(windows_features.shape[0], 1, windows_features.shape[1]) 

    X_train, X_test, y_train, y_test = train_test_split(
        windows_features, y_all, test_size=0.2, shuffle=True
    )

    if not settings.BI_LSTM_PATH.exists():
        new_bi_lstm = Bi_LSTM(
            input_shape=windows_features.shape[2],
            model_path=settings.BI_LSTM_PATH, 
            new_model=True
        )
        new_bi_lstm.fit_model(X_train, y_train, X_test, y_test)
        bi_lstm = new_bi_lstm.model
    else:
        new_bi_lstm = Bi_LSTM(model_path=settings.BI_LSTM_PATH, new_model=False)
        bi_lstm = new_bi_lstm.model

    return sensor_preprocessor, bi_lstm, encoder

def get_models(settings):
    if not check_if_init(settings):
        preproccesor, bi_lstm, encoder = initial_build(settings)
        return preproccesor, bi_lstm, encoder
    else:
        loaded_autoencoder = AutoEncoder(model_path=settings.AUTO_ENCODER_PATH, model_path_encoder=settings.ENCODER_PATH, new_model=False)
        loaded_bi_lstm = Bi_LSTM(input_shape=256, model_path=settings.BI_LSTM_PATH, new_model=False)
         
        encoder =  loaded_autoencoder.encoder
        bi_lstm = loaded_bi_lstm.model
        preproccesor = SensorPreprocessor(init_build=False)
        return preproccesor, bi_lstm, encoder
