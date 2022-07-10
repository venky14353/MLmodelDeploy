from flask import Flask, render_template, request
import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
import numpy as np

Classifier = pickle.load(open('model.sav','rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def diabetes():
    Pregnancies = int(request.form['Pregnancies'])
    Glucose = int(request.form['Glucose'])
    BloodPressure = int(request.form['BloodPressure'])
    SkinThickness = int(request.form['SkinThickness'])
    Insulin = int(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = int(request.form['Age'])


    #prediction
    result = Classifier.predict(np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1))
    
    if result[0] == 0:
        return "Non Diabetic"
    else:
        return "Diabetic"

    return render_template('index.html', result = result)



if __name__ == '__main__':
    app.run(debug=True)