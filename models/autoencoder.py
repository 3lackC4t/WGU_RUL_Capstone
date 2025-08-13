from models.model import BaseModel
from keras.saving import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import Input, Model
from keras.layers import (
    Conv1D,
    BatchNormalization,
    Dropout,
    Dense,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Reshape,
    UpSampling1D,
)


class AutoEncoder(BaseModel):
    def __init__(self, input_shape=256, epochs=100, batch_size=128, model_path=None, new_model=True, model_path_encoder=None):
        super().__init__(input_shape, epochs, batch_size, model_path, new_model)
        self.model_path_encoder = model_path_encoder
        self.callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, vebose=1),
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        ]

        if not new_model and model_path:
            self.autoencoder = load_model(model_path)
        if not new_model and model_path_encoder:     
            self.encoder = load_model(model_path_encoder) 
        else: 
            self.autoencoder, self.encoder = self.build_and_compile_model()

    def build_and_compile_model(self):
        encoder_input = Input(shape=(self.input_shape,), name='input')
        encoded = Dense(256, activation='relu')(encoder_input)
        encoded = Dense(128, activation='relu')(encoded)

        encoded_final = Dense(64, activation='relu')(encoded)

        decoded = Dense(128, activation='relu')(encoded_final)
        decoded = Dense(256, activation='sigmoid')(decoded)

        autoencoder = Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        encoder = Model(encoder_input, encoded_final)

        return autoencoder, encoder


    def build_and_compile_model_advanced(self):
        encoder_input = Input(shape=(self.input_shape,))

        reshaped = Reshape((self.input_shape, 1))(encoder_input)
        
        # Encoder
        x = Conv1D(256, 7, activation='relu', padding='same')(reshaped)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x) # 64 -> 32
        x = Dropout(0.3)(x)

        x = Conv1D(128, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x) # 32 -> 16
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)  
       
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x) # 16 -> 8
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # Reshape and restore the time dimension
        encoded = GlobalAveragePooling1D()(x)
        decoder_dense = Dense(8 * 64, activation='relu')(encoded)
        x = Reshape((8, 64))(decoder_dense)

        # Decoder
        x = UpSampling1D(2)(x) # 8 -> 16
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = UpSampling1D(2)(x) # 16 -> 32
        x = Conv1D(128, 5, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = UpSampling1D(2)(x) # 32 -> 64
        x = Conv1D(256, 7, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        decoded = Conv1D(1, 1, activation='linear', padding='same')(x)
        decoded = Reshape((self.input_shape,))(decoded)
        
        autoencoder = Model(encoder_input, decoded)
        encoder = Model(encoder_input, encoded)

        autoencoder.compile(optimizer="adam", loss='mse', metrics=['mae'])

        return autoencoder, encoder

    def fit_model(self, model_input, model_output):
        history = self.autoencoder.fit(
            model_input,
            model_output,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1
        )
        self.autoencoder.save(self.model_path)
        self.encoder.save(self.model_path_encoder)
        self.model_history = history
