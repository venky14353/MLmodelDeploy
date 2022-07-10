from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

lr = pickle.load(open('model.sav','rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def titanic():
    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])
    Age = int(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = int(request.form['Embarked'])

    result = lr.predict(np.array([Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]).reshape(1,-1))
    if result[0]==0:
        return "the passenger is not survived"
    else:
        return "the passenger is survived"

    return render_template('index.html',result = result)





if __name__ == '__main__':
    app.run(debug=True)