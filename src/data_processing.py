from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pandas as pd

def preprocess(data):
    data = data.copy()
    
    data['price_log'] = np.log1p(data['price'])
    data['price_boxcox'], best_lambda = stats.boxcox(data['price'])
    
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)
    
    return data, best_lambda

def data_division(data, y, target : str):

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.2, random_state=42)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test