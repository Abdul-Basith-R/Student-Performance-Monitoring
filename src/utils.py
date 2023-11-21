import os
import pandas as pd
import numpy as np
import sys
import dill
from sklearn.metrics import r2_score

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj, fileobj)
    except Exception as e:
        print(e)


def evaluate(xtrain,xtest,ytrain,ytest,models):
    try:
        reports = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xtrain,ytrain)

            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)
            train_model_score = r2_score(ytrain,ytrain_pred)
            test_model_score = r2_score(ytest,ytest_pred)

            reports[list(models.keys())[i]] = test_model_score
        return reports
    

    except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)