from flask import Flask,request,render_template,url_for,redirect
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
log_model=pickle.load(open('glasslog.pkl','rb'))


@app.route('/')

def model():
    return render_template('home.html')

@app.route('/glass_pred',methods=['POST'])

def glass_type():
    RI=request.form['RI']
    Na=request.form['Na']
    Mg=request.form['Mg']
    Al=request.form['Al']
    Ba=request.form['Ba']
    Fe=request.form['Fe']

    data_frame={"RI":[RI],"Na":[Na],"Mg":[Mg],"Al":[Al],"Ba":[Ba],"Fe":[Fe]}

    test_data=pd.DataFrame(data_frame)

    glass_type=log_model.predict(test_data)

    return render_template('after.html',data=glass_type[0])

if __name__== "__main__":
    app.run(host='0.0.0.0',port=5018)