from src.config import X_train_scaled_price_log, y_train_price_log, models_price_log
from src.train_model import train

mod = train(models_price_log, X_train_scaled_price_log, y_train_price_log)

Linear_price_log, Ridge_price_log, Lasso_price_log = mod['Linear'], mod['Ridge'], mod['Lasso']