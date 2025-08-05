import keras
from keras import Input, Model
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    RepeatVector,
    TimeDistributed
)


class LSTM_AutoEncoder:
    def __init__(self, input_shape=(16, 4), latent_dim=(32)):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.autoencoder, self.encoder = self.build_lstm_autoencoder()

    def build_lstm_autoencoder(self):
        
        print(self.input_shape)
        encoder_input = Input(shape=self.input_shape)
        
        # Encoder
        x = LSTM(128, return_sequences=True)(encoder_input)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        # Bottleneck
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = GlobalAveragePooling1D()(attention)

        # Latent
        latent = Dense(self.latent_dim, activation='relu', name='latent_features')(x)
        
        # Decoder
        decoded = RepeatVector(self.input_shape[0])(latent)
        decoded = LSTM(64, return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(128, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.input_shape[1], activation='linear'))(decoded)
        
        autoencoder = Model(encoder_input, decoded)
        encoder = Model(encoder_input, latent)

        autoencoder.compile(optimizer="adam", loss='mse', metrics=['mae'])

        return autoencoder, encoder