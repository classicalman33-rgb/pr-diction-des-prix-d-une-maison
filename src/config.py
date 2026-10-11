import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loading_data import data
from src.data_processing import preprocess, data_division
from src.find_best_params import best_params
from sklearn.linear_model import LinearRegression, Ridge, Lasso


data_processed, best_lambda = preprocess(data) # adding price_log and price_boxcox column to our data

X = data_processed.drop(columns=['price', 'price_log', "price_boxcox"]) #separates features from targets
y = data_processed['price'] # will be used to make prediction by using price 
y_log = data_processed['price_log'] # will be used to make prediction by using price_log
y_boxcox = data_processed['price_boxcox'] # will be used to make prediction by using price_boxcox 


X_train_scaled_price, X_test_scaled_price, y_train_price, y_test_price = data_division(X, y, 'price')
X_train_scaled_price_log, X_test_scaled_price_log, y_train_price_log, y_test_price_log = data_division(X, y_log, 'price_log')
X_train_scaled_price_boxcox, X_test_scaled_price_boxcox, y_train_price_boxcox, y_test_price_boxcox = data_division(X, y_boxcox, 'price_boxcox')

alpha_ridge_price , alpha_lasso_price = best_params(X_train_scaled_price, y_train_price)
alpha_ridge_price_log , alpha_lasso_price_log = best_params(X_train_scaled_price_log, y_train_price_log)
alpha_ridge_price_boxcox , alpha_lasso_price_boxcox = best_params(X_train_scaled_price_boxcox, y_train_price_boxcox)


models_price = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge_price),
        'Lasso': Lasso(alpha=alpha_lasso_price)
    }

models_price_log = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge_price_log),
        'Lasso': Lasso(alpha=alpha_lasso_price_log)
    }

models_price_boxcox = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge_price_boxcox),
        'Lasso': Lasso(alpha=alpha_lasso_price_boxcox)
    }
