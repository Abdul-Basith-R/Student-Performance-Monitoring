import os
import pandas as pd
import numpy as np
import sys
import dill

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj, fileobj)
    except Exception as e:
        print(e)
