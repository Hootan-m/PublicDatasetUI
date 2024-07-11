import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import json     


app=Flask(__name__)
## Load the model
XGBoostModel=pickle.load(open('XGBoostModel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    a = np.array(list(data.values())).reshape(1, -1)
    new_data=scalar.transform(a)
    output=XGBoostModel.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    b = np.array(data).reshape(1,-1)
    final_input = scalar.transform(b)
    print(final_input)
    output = XGBoostModel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The failure condition of the power transformer is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)