import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam

class AutoEncoder:
    def __init__(self, input_dim=128):
        self.input_dim = input_dim
        self.call_backs = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        self.model = self.build_model()
        self.is_trained = False

    def build_model(self) -> Model:
        inputs = Input(shape=(self.input_dim,), name='sensor_input')

        # Encoder
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.1)(x)
        encoded = Dense(8, activation='relu')(x)

        # Decoder
        x = Dense(16, activation='relu')(encoded)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        decoded = Dense(self.input_dim, activation='linear')(x)

        # Build, compile, return
        autoencoder = Model(inputs, decoded, name='bearing_autoencoder')
        autoencoder.compile(
            optimizer=Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='mse',
            metrics=['mae']
        )

        return autoencoder
    
    def train_model(self, input_data, test_data=None) -> None:
        print(f"Training autoencoder on {len(input_data)} healthy samples with shape {input_data.shape}")

        if self.model is None:
            self.build_model()

        history = self.model.fit(
            input_data, input_data,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=self.call_backs,
            verbose=1
        )

        self.is_trained = True
        print(f"Training is completed. Final loss: {history.history['loss'][-1]:.6f}")
        return history
    
    def get_errors(self, data):
        if not self.is_trained:
            raise ValueError("Model must be trained prior to extracting errors")
        
        reconstructed = self.model.predict(data, verbose=0)

        errors = np.mean((data - reconstructed) ** 2, axis=1)
        return errors
    
    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)

    def load_model(self, filepath):
        if self.model is None and filepath:
            from keras.models import load_model
            
            self.model = load_model(filepath)
            self.is_trained = True