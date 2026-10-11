from src.config import X_train_scaled_price, y_train_price, models_price
from src.train_model import train

mod = train(models_price, X_train_scaled_price, y_train_price)

Linear_price, Ridge_price, Lasso_price = mod['Linear'], mod['Ridge'], mod['Lasso']