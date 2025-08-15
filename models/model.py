from keras.callbacks import ReduceLROnPlateau, EarlyStopping

class BaseModel:
    def __init__(self, input_shape, epochs=100, batch_size=256, model_path=None, new_model=True):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.new_model = new_model
        self.callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, vebose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        ]

        self.model_history = None
        self.model = None

    def build_and_compile_model(self):
        pass

    def fit_model(self, input, output, x_test=None, y_test=None):
        pass