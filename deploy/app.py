from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.svm import SVC

sv = pickle.load(open('iris.sav','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('gome.html')

@app.route('/predict', methods=['POST'])

def gome():
    
    data1 = float(request.form.get('a'))
    data2 = float(request.form.get('b'))
    data3 = float(request.form.get('c'))
    data4 = float(request.form.get('d'))

    # prediction
    
    result = sv.predict(np.array([data1,data2,data3,data4]).reshape(1,-1))
    
    if result[0] == 0:
        return  'Iris-setosa'
    elif result[0] == 1:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'
         
    return render_template('gome.html', result = result)

if __name__ == '__main__':
    app.run(debug=True)