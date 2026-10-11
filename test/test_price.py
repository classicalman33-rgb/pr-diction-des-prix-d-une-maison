from models.price import Linear_price, Ridge_price, Lasso_price
from src.config import X_test_scaled_price, y_test_price
from src.evaluate_model import model_evaluation

y_pred_lin = Linear_price.predict(X_test_scaled_price)
y_pred_rid = Ridge_price.predict(X_test_scaled_price)
y_pred_las = Lasso_price.predict(X_test_scaled_price)


Lin = model_evaluation(y_test_price, y_pred_lin)

Rid = model_evaluation(y_test_price, y_pred_rid)

Las = model_evaluation(y_test_price, y_pred_las)

print('Linear_price evaluation : ')
print(Lin)
print('-------------------------------------------------------------------------------------------------------------------------------------------------')

print('Ridge_price evaluation : ')
print(Rid)
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print('Lasso_price evaluation : ')
print(Las)