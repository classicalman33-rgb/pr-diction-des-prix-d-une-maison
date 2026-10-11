from scipy.special import inv_boxcox

from models.price_boxcox import Linear_price_boxcox, Ridge_price_boxcox, Lasso_price_boxcox
from src.config import X_test_scaled_price_boxcox, y_test_price, best_lambda
from src.evaluate_model import model_evaluation

y_pred_lin = Linear_price_boxcox.predict(X_test_scaled_price_boxcox)
y_pred_rid = Ridge_price_boxcox.predict(X_test_scaled_price_boxcox)
y_pred_las = Lasso_price_boxcox.predict(X_test_scaled_price_boxcox)

y_pred_linear_inv = inv_boxcox(y_pred_lin, best_lambda)
y_pred_rid_inv = inv_boxcox(y_pred_rid, best_lambda)
y_pred_las_inv = inv_boxcox(y_pred_las, best_lambda)

Lin = model_evaluation(y_test_price, y_pred_lin)
Lin_inv = model_evaluation(y_test_price, y_pred_linear_inv)

Rid = model_evaluation(y_test_price, y_pred_rid)
Rid_inv = model_evaluation(y_test_price, y_pred_rid_inv)

Las = model_evaluation(y_test_price, y_pred_las)
Las_inv = model_evaluation(y_test_price, y_pred_las_inv)

print('Linear_price_boxcox evaluation : ')
print(Lin)
print('Linear_price_boxcox_reversed evaluation : ')
print(Lin_inv)
print('-------------------------------------------------------------------------------------------------------------------------------------------------')

print('Ridge_price_boxcox evaluation : ')
print(Rid)
print('Ridge_price_boxcox_reversed evaluation : ')
print(Rid_inv)
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print('Lasso_price_boxcox evaluation : ')
print(Las)
print('Lasso_price_boxcox_reversed evaluation : ')
print(Las_inv)