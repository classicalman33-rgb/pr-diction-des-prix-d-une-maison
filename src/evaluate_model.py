from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def model_evaluation(y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        R2 = r2_score(y_test, y_pred)
        eva = {
                'MAE' : mae,
                'MSE' : mse,
                'RMSE' : rmse,
                'R²' : R2
        }
        return eva
