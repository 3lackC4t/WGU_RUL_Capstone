from flask import Flask, request, render_template

DATA = None

app = Flask(__name__)


@app.route('/', methods=['GET'])
def dashboard():
    return "HELLO"

@app.route('/api/input', methods=['POST'])
def feed_input_data():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return render_template('DataPrediction.html')