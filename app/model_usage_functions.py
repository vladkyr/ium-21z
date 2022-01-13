import pickle
import pandas as pd

def create_features(df):
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.weekofyear
    del df['date']
    category_paths_list = ['category_path_Gry i konsole;Gry komputerowe',
                           'category_path_Gry i konsole;Gry na konsole;Gry PlayStation3',
                           'category_path_Gry i konsole;Gry na konsole;Gry Xbox 360',
                           'category_path_Komputery;Drukarki i skanery;Biurowe urządzenia wielofunkcyjne',
                           'category_path_Komputery;Monitory;Monitory LCD',
                           'category_path_Sprzęt RTV;Video;Telewizory i akcesoria;Anteny RTV']
    for path in category_paths_list:
        product_path = f"category_path_{df['category_path']}"
        if product_path == path:
            df[path] = [1]
        df[path] = [0]
    return df


def prepare_data_for_advanced_model(date, product_id):
    products = pd.read_json("../data/raw/second/products.jsonl", lines=True)
    del products['product_name']
    products['product_id'] = products['product_id'].astype(str)
    df = products[products['product_id'] == product_id]
    df['date'] = [date]
    df['date'] = pd.to_datetime(df['date'])
    df = create_features(df)
    del df['product_id']
    del df['category_path']
    return df


def predict_advanced(df):
    reg = pickle.load(open(f"../models/xgboost_reg.pkl", "rb"))
    prediction = reg.predict(df)
    prediction = prediction[0]

    prediction = int(prediction)
    if prediction < 0:
        prediction = 0

    return prediction