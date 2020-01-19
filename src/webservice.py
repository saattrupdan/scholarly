from flask import Flask, request, render_template, redirect
from pathlib import Path

app = Flask(__name__, template_folder = Path('static'))

@app.route('/')
def index():
    return redirect('/scholarly')

@app.route('/scholarly', methods = ['POST', 'GET'])
def result():
    import json
    from scholarly.modules import load_model

    data_dict = request.form if request.method == 'POST' else request.args
    if not data_dict:
        return render_template('scholarly.html')

    model_path = next(Path('.data').glob('scholarly_model*.pt'))
    model, _ = load_model(model_path)
    preds = model.predict(data_dict['title'], data_dict['abstract'])

    if request.method == 'POST':
        return render_template('scholarly.html', preds = preds, **data_dict)
    else:
        return json.dumps(preds)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
