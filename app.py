from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from maternal import rf_clf


model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def pred():
    age = request.form['Age']
    systolicBP = request.form['SystolicBP']
    diastolicBP = request.form['DiastolicBP']
    bloodsugar = request.form['Blood Sugar']
    bloodpressure = request.form['Body Temperature']
    heartrate = request.form['Heart Rate']
    form_array = np.array([[age, systolicBP, diastolicBP, bloodsugar, bloodpressure, heartrate]])
    prediction = rf_clf.predict(form_array)[0]
    

    return render_template("pred.html", data=prediction)

if __name__ == "__main__":
    app.run(debug=True)




