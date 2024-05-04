import pickle
from flask import Flask,request,app,jsonify,url_for,redirect,flash,session,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
## Lodd the model
model=pickle.load(open("Aircraft_Damage_Propagation.pkl",'rb'))
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict_api',methods=['POST'])

def predict_api():
   data=request.json['data']
   print(data)
   print(np.array(list(data.values()))).reshape(1,-1)
   new_data=np.ScalarType.transform(np.array (list(data.values())).reshape(1,-1))
   output=model.predict(new_data)
   return jsonify(output[0])



if __name__=="__main__":
    app.debug(True)
 
 
