import keras as ks

class Autoencoder:
    def __init__(self, input_dim, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = [
            ks.callbacks.ReduceLROnPlateau(),
            ks.callbacks.EarlyStopping()
        ]
        self.model = None

    def build_model(self) -> None:

        encoder_input = ks.layers.Input((self.input_dim,), name="encoder_input_layer")

        x = ks.layers.Dense(128, activation='relu')(encoder_input)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Dropout(0.3)(x)

        x = ks.layers.Dense(64, activation='relu')(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Dropout(0.2)(x)

        x = ks.layers.Dense(32, activation='relu')(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Dropout(0.1)(x)

        bottle_neck = ks.layers.Dense(16, activation='relu')(x)

        decoded = ks.layers.Dense(32, activation='relu')(bottle_neck)
        decoded = ks.layers.Dense(64, activation='relu')(decoded)
        decoded = ks.layers.Dense(128, activation='relu')(decoded)
        decoded = ks.layers.Dense(self.input_dim, activation='linear')(decoded)

        autoencoder = ks.models.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.model = autoencoder

    def fit(self, data) -> None:
        if self.model:
            self.model.fit(
                data, data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,
                verbose=1
            ) 