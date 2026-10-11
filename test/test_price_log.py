import numpy as np

from models.price_log import Linear_price_log, Ridge_price_log, Lasso_price_log
from src.config import X_test_scaled_price_log, y_test_price
from src.evaluate_model import model_evaluation

y_pred_lin = Linear_price_log.predict(X_test_scaled_price_log)
y_pred_rid = Ridge_price_log.predict(X_test_scaled_price_log)
y_pred_las = Lasso_price_log.predict(X_test_scaled_price_log)

y_pred_linear_inv = np.expm1(y_pred_lin)
y_pred_rid_inv = np.expm1(y_pred_rid)
y_pred_las_inv = np.expm1(y_pred_las)

Lin = model_evaluation(y_test_price, y_pred_lin)
Lin_inv = model_evaluation(y_test_price, y_pred_linear_inv)

Rid = model_evaluation(y_test_price, y_pred_rid)
Rid_inv = model_evaluation(y_test_price, y_pred_rid_inv)

Las = model_evaluation(y_test_price, y_pred_las)
Las_inv = model_evaluation(y_test_price, y_pred_las_inv)

print('Linear_price_log evaluation : ')
print(Lin)
print('Linear_price_log_reversed evaluation : ')
print(Lin_inv)
print('-------------------------------------------------------------------------------------------------------------------------------------------------')

print('Ridge_price_log evaluation : ')
print(Rid)
print('Ridge_price_log_reversed evaluation : ')
print(Rid_inv)
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print('Lasso_price_log evaluation : ')
print(Las)
print('Lasso_price_log_reversed evaluation : ')
print(Las_inv)