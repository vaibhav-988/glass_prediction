from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
log_model=pickle.load(open('glasslog.pkl','rb'))


@app.route('/')

def model():
    return 'check your glass type here'

@app.route('/glass_pred')

def glass_type():
    data=request.get_json()
    RI=data['RI']
    Na=data['Na']
    Mg=data['Mg']
    Al=data['Al']
    Ba=data['Ba']
    Fe=data['Fe']

    data_frame={"RI":[RI],"Na":[Na],"Mg":[Mg],"Al":[Al],"Ba":[Ba],"Fe":[Fe]}

    test_data=pd.DataFrame(data_frame)

    glass_type=log_model.predict(test_data)

    return jsonify ({"glass_type is":glass_type[0]})

if __name__== "__main__":
    app.run()