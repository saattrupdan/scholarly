from flask import Flask, request, render_template
from pathlib import Path
from src.modules import load_model

application = Flask(__name__, template_folder = Path('static'))
model_path = next(Path('.data').glob('model*.pt'))
model, _ = load_model(model_path)

@application.route('/')
def predict():
    return render_template('index.html')

@application.route('/result', methods = ['POST', 'GET'])
def result():
    import json

    data_dict = request.form if request.method == 'POST' else request.args
    if not data_dict:
        return render_template('index.html')

    preds = model.predict(data_dict['title'], data_dict['abstract'])

    if request.method == 'POST':
        return render_template('index.html', preds = preds, **data_dict)
    else:
        return json.dumps(preds)

if __name__ == '__main__':
    application.run(debug = True, host = '0.0.0.0')
