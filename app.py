import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from model import build_model
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
###########################################
# LOAD FILES
test = pd.read_csv('./saved/test.csv')
features_col_name_sc = test.columns[1:]
scalerfile = './saved/sc.pkl'
sc = pickle.load(open(scalerfile, 'rb'))
##############################################


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    test_1 = test.copy()

    test_1.loc[len(test_1)] = int_features

    test_1 = test_1.iloc[-24:, :]

    sc.clip = False
    test_1[features_col_name_sc] = sc.transform(test_1[features_col_name_sc])

    test_1 = test_1.iloc[:24, :14].to_numpy().reshape(1, 24, 14)

    # model load
    model = build_model()
    model.load_weights('./saved/transformer_fold_0.h5')

    # # model predict
    predicted_v0 = model.predict(test_1)
    y_pred_comp1 = predicted_v0[0].flatten()
    y_pred_comp2 = predicted_v0[1].flatten()
    y_pred_comp3 = predicted_v0[2].flatten()
    y_pred_comp4 = predicted_v0[3].flatten()

    y_pred_classes_comp1 = np.array(
        list(map(lambda x: 0 if x < 0.5 else 1, y_pred_comp1)))
    y_pred_classes_comp2 = np.array(
        list(map(lambda x: 0 if x < 0.5 else 1, y_pred_comp2)))
    y_pred_classes_comp3 = np.array(
        list(map(lambda x: 0 if x < 0.5 else 1, y_pred_comp3)))
    y_pred_classes_comp4 = np.array(
        list(map(lambda x: 0 if x < 0.5 else 1, y_pred_comp4)))

    return render_template('index.html', prediction_text=f'Prediction for \
                           First(comp1_fail), Second, Third, and Fourth Classes: \
                           {y_pred_classes_comp1[0]}, \
                            {y_pred_classes_comp2[0]}, \
                            {y_pred_classes_comp3[0]}, \
                             {y_pred_classes_comp4[0]}')


if __name__ == "__main__":
    app.run(debug=True)
