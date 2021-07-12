#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[4]:


app = Flask(__name__)
model = pickle.load(open('pfz_model.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def home():
    
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
    
        result = ''
        if output == 1:
            result = 'PFZ'
        else:
            result = 'NPFZ'
        
        return render_template('Result.html', prediction_text='The predicted output is {}'.format(result))
    else:
        return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     features = [float(x) for x in request.form.values()]
#     final_features = [np.array(features)]
#     prediction = model.predict(final_features)
    
#     output = round(prediction[0], 2)
    
#     result = ''
#     if output == 1:
#         result = 'PFZ'
#     else:
#         result = 'NPFZ'
        
#     return render_template('index.html', prediction_text='The predicted output is ${}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




