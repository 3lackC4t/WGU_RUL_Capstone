import numpy as np
from keras.layers import Input, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model

class AutoEncoder:
    def __init__(self, input_dim=128):
        self.input_dim = input_dim
        self.call_backs = [
            ReduceLROnPlateau(patience=5),
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
        self.model = self.build_model()
        self.is_trained = False

    def build_model(self) -> Model:
        inputs = Input(shape=(self.input_dim,), name='sensor_input')

        # Encoder
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        encoded = Dense(16, activation='relu')(x)

        # Decoder
        x = Dense(32, activation='relu')(encoded)
        x = Dense(64, activation='relu')(x)

        decoded = Dense(self.input_dim, activation='linear')(x)

        # Build, compile, return
        autoencoder = Model(inputs, decoded, name='bearing_autoencoder')
        autoencoder.compile(
            optimizer='adam',
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
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            callbacks=self.call_backs,
            verbose=1
        )

        self.is_trained = True
        print(f"Training is completed. Final loss: {history.history['loss'][-1]:.6f}")
        return history
    
    def get_error(self, data):
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