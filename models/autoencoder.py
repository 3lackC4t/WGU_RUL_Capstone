import keras as ks

class Autoencoder:
    def __init__(self, input_dim, epochs=200, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = [
            ks.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ks.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        self.model = None

    def build_model(self) -> None:

        encoder_input = ks.layers.Input((self.input_dim,), name="encoder_input_layer")

        x = ks.layers.Dense(128, activation="relu")(encoder_input)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.Dense(64, activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.Dense(32, activation="relu")(x)
        x = ks.layers.BatchNormalization()(x)

        bottle_neck = ks.layers.Dense(16, activation="relu")(x)

        decoded = ks.layers.Dense(32, activation="relu")(bottle_neck)
        decoded = ks.layers.Dense(64, activation="relu")(decoded)
        decoded = ks.layers.Dense(128, activation="relu")(decoded)
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
            history = self.model.fit(
                data, data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,
                verbose=1
            ) 

            return history
        
    def predict_on_input(self, input_data):
        return self.model.predict(input_data)
        
    def save_model(self, model_path) -> None:
        try:
            self.model.save(model_path.as_posix())
        except FileNotFoundError or FileExistsError as e:
            print(f"Failed to save model to {model_path} [{e}]")

    def load_model(self, model_path) -> None:
        try:
            self.model = ks.saving.load_model(model_path.as_posix())
        except FileNotFoundError or FileExistsError as e:
            print(f"Failed to load model to {model_path} [{e}]")
