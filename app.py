import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model rfc.sav', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/form_predict')
def form_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 0:
        out = 'Anda terkena Diabetes'
    else:
        out = 'Anda tidak terkena Diabetes'

    return render_template('result_predict.html', prediction_text='{}'.format(out))

if __name__ == "__main__":
    app.run(debug=True)