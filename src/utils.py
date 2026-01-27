from copyreg import pickle
import os
import sys
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models:dict, param:dict):
    report = {}

    for name, model in models.items():
        try:
            # Special handling for CatBoost (not fully sklearn compatible)
            if isinstance(model, CatBoostRegressor):
                # Use CatBoost's own grid_search
                grid_result = model.grid_search(param.get(name, {}), 
                                                X=X_train, 
                                                y=y_train, 
                                                cv=3, 
                                                verbose=False)
                # After grid_search, model is already fitted
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[name] = test_model_score

            else:
                # Standard sklearn models
                gs = GridSearchCV(model, param.get(name, {}), cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_test_pred = best_model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[name] = test_model_score

                # Replace unfitted model with fitted best_model
                models[name] = best_model

            return report
        
        except Exception as e:
            raise CustomException(e, sys)

    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


