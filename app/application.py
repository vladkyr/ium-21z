from flask import Flask, jsonify, request
import pandas as pd

from app.model_usage_functions import prepare_data_for_advanced_model, predict_advanced

pd.options.mode.chained_assignment = None  # default='warn'

app = Flask(__name__)





@app.route('/predict/advanced_model', methods=['GET'])
def predict_advanced_model():
    date = request.args.get('date', type=str)
    product_id = request.args.get('product_id', type=str)
    df = prepare_data_for_advanced_model(date, product_id)
    prediction = predict_advanced(df)
    return jsonify(prediction)


# return jsonify(date, product_id)


@app.route('/predict/basic_model', methods=['GET'])
def predict_basic_model():
    return "Will finish that later"


@app.route('/')
def index():
    return 'Please go to either /predict/advanced_model or /predict/basic_model to get predictions \n' \
           'Keep in mind that you have to supply date (in YYYY-MM-DD format) and product_id'


if __name__ == '__main__':
    app.run(debug=True)
