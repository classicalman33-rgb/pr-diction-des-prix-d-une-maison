from src.config import X_train_scaled_price_boxcox, y_train_price_boxcox, models_price_boxcox
from src.train_model import train

mod = train(models_price_boxcox, X_train_scaled_price_boxcox, y_train_price_boxcox)

Linear_price_boxcox, Ridge_price_boxcox, Lasso_price_boxcox = mod['Linear'], mod['Ridge'], mod['Lasso']