from flask import Flask, jsonify, request
import pandas as pd
from ab_testing import write_logs_to_file

from app.model_usage_functions import prepare_data_for_advanced_model, predict_advanced, predict_basic

pd.options.mode.chained_assignment = None  # default='warn'

app = Flask(__name__)


@app.route('/predict/advanced_model', methods=['GET'])
def predict_advanced_model():
    date = request.args.get('date', type=str)
    product_id = request.args.get('product_id', type=str)
    df = prepare_data_for_advanced_model(date, product_id)
    prediction = predict_advanced(df)
    response = jsonify(prediction)
    write_logs_to_file('advanced_model', request.args, response.json)
    return response


@app.route('/predict/base_model', methods=['GET'])
def predict_basic_model():
    product_id = request.args.get('product_id', type=str)
    if product_id is None:
        return 'Please provide product_id.'
    else:
        prediction = predict_basic(product_id)
        response = jsonify(prediction)
        write_logs_to_file('base_model', request.args, response.json)
        return response


@app.route('/')
def index():
    return """<xmp>
           Please go to either /predict/advanced_model or /predict/base_model to get predictions.
           
           Keep in mind that you have to supply date (in YYYY-MM-DD format) and product_id (date is not required for base model). 
           </xmp>"""


if __name__ == '__main__':
    app.run(debug=True)
