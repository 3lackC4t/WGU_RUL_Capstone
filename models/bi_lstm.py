from models.model import BaseModel
from keras import Sequential
from keras.saving import save_model, load_model
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    BatchNormalization, 
    Bidirectional
)

from keras.regularizers import l2

class Bi_LSTM(BaseModel):
    def __init__(self, input_shape, epochs=100, batch_size=256, model_path=None, new_model=True):
        super().__init__(input_shape, epochs, batch_size, model_path, new_model)

        if new_model and model_path:
            self.model = self.build_and_compile_model()
        else:
            self.model = load_model(model_path)

    def build_and_compile_model(self):
         model = Sequential([
            # First Layer 64 selectron Bi-LSTM
            Bidirectional(
                LSTM(64, 
                    return_sequences=True,
                    kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            Dropout(0.3),

            # Second layer 32 selectron Bi-LSTM
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.2),

            # 64, 32, and output Dense layers.
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),

            Dense(16, activation='relu'),
            
            # Tanh for squashing the output to between 0 and 1, our degradation target
            Dense(1, activation='sigmoid')
        ])
         
         model.compile(
            optimizer="adam", 
            loss='binary_crossentropy', 
            metrics=['mae', 'mse']
        )
         
         return model

    def fit_model(self, model_input, model_output, X_test, y_test):
        history = self.model.fit(
            model_input,
            model_output,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=self.callbacks,
            verbose=1
        )
        save_model(model=self.model, filepath=self.model_path, overwrite=True)
        self.model_history = history
