import numpy as np
from flask import Flask, request,render_template,jsonify  #"jsonify":unknown word.
import pickle

#Initialize the flask APP
app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

#default page of our web app
@app.route("/")
def home():
    return render_template('index.html')

#To use predict button in our web app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if(output==0):
        return render_template('index.html', prediction_text= 'Patient is not likely to have heart disease')
    else:
        return render_template('index.html', prediction_text='Patient is likely to have  heart disease')

if __name__ == "__main__":
    app.run(debug=True)
