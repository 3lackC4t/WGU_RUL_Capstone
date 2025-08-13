class BaseModel:
    def __init__(self, input_shape, epochs=100, batch_size=32, model_path=None, new_model=True):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.new_model = new_model
        self.call_backs = []

        self.model_history = None
        self.model = None

    def build_and_compile_model(self):
        pass

    def fit_model(self, input, output, x_test=None, y_test=None):
        pass