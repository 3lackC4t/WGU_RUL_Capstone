import keras
from keras import Sequential
from keras.saving import save_model, load_model, save_weights, load_weights
from keras import Input, Model
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    RepeatVector
)


class LSTM_AutoEncoder:
    def __init__(self, model_file_ae, model_file_encoder, input_dim=(16, 4), latent_dim=(32)):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if model_file_ae:
            self.autoencoder = load_model(model_file_ae)
        if model_file_encoder:     
            self.encoder = load_model(model_file_encoder) 
        else: 
            self.autoencoder, self.encoder = self.build_lstm_autoencoder()

    def build_lstm_autoencoder(self):
        encoder_input = Input(shape=self.input_dim)
        
        # Encoder
        encoder = Sequential([
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            Dense(self.latent_dim)
        ])

        # Decoder
        decoder = Sequential([
            RepeatVector(self.input_dim[0]),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True)
        ])
        
        autoencoder = Model(encoder.input, decoder.ouput)
        encoder = Model(encoder_input, encoder.output)

        autoencoder.compile(optimizer="adam", loss='mse', metrics=['mae'])

        return autoencoder, encoder

    def fit_autoencoder(self, input, output, model_path):
        self.autoencoder.fit(
            input,
            output,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        save_model(model=self.autoencoder, filepath=model_path, overwrite=True)
