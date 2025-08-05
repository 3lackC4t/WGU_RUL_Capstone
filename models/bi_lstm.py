from keras import Sequential
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    BatchNormalization, 
    Bidirectional
)

from keras.regularizers import l2

class Bi_LSTM:
    def __init__(self):
        self.model = self.build_BiLSTM_model()

    def build_BiLSTM_model(self):
    
        model = Sequential([
            # First Layer 64 selectron Bi-LSTM
            Bidirectional(
                LSTM(64, 
                    return_sequences=True,
                    kernel_regularizer=l2(0.01))),
            BatchNormalization(),
            Dropout(0.3),

            # Second layer 32 selectron Bi-LSTM
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.2),

            # 64, 32, and output Dense layers.
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            
            # Tanh for squashing the output to between 0 and 1, our degradation target
            Dense(1, activation='tanh')
        ])
        
        # Optimizer uses learning rate scheduling
        
        model.compile(
            optimizer="adam", 
            loss='huber', 
            metrics=['mae', 'mse']
        )
        
        return model