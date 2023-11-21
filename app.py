from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application
@app.route('/')
def indeX():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(gender=request.form.get('gender'),
                          race_ethnicity=request.form.get('ethnicity'),
                          parental_level_of_education=request.form.get('parental_level_of_education'),
                          lunch=request.form.get('lunch'),
                          test_preparation_course=request.form.get('test_preparation_course'),
                          reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
        pred_df = data.data_frame()
        print(pred_df)
        print("Before Prediction")
        predictPipeline = PredictPipeline()
        print("Mid Prediction")
        result = predictPipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html',results=result[0])
    

if __name__ == '__main__':
    app.run(debug=True)
