import keras
from keras import Sequential
from keras.saving import save_model, load_model, save_weights, load_weights
from keras import Input, Model
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Reshape,
    UpSampling1D,
)


class AutoEncoder:
    def __init__(self, model_file_ae=None, model_file_encoder=None, input_dim=64):
        self.input_dim = input_dim
        
        if model_file_ae:
            self.autoencoder = load_model(model_file_ae)
        if model_file_encoder:     
            self.encoder = load_model(model_file_encoder) 
        else: 
            self.autoencoder, self.encoder = self.build_lstm_autoencoder()

    def build_lstm_autoencoder(self):
        encoder_input = Input(shape=(self.input_dim,))

        reshaped = Reshape((self.input_dim, 1))(encoder_input)
        
        # Encoder
        x = Conv1D(32, 7, activation='relu', padding='same')(reshaped)
        x = MaxPooling1D(2)(x) # 32 -> 16
        x = Conv1D(16, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x) # 16 -> 8
        encoded = GlobalAveragePooling1D()(x)

        # Decoder

        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x) # 8 -> 16
        x = Conv1D(16, 5, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x) # 16 -> 32
        x = Conv1D(32, 7, activation='relu', padding='same')(x)
        decoded = Conv1D(1, 1, activation='linear', padding='same')(x)
        decoded = Reshape((self.input_dim,))(decoded)
        
        autoencoder = Model(encoder_input, decoded)
        encoder = Model(encoder_input, encoded)

        autoencoder.compile(optimizer="adam", loss='mse', metrics=['mae'])

        return autoencoder, encoder

    def fit_autoencoder(self, input, output, model_path_ae, model_path_encoder):
        self.autoencoder.fit(
            input,
            output,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        self.autoencoder.save(model_path_ae)
        self.encoder.save(model_path_encoder)
