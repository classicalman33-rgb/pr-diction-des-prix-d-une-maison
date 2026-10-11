from src.config import X_train_scaled_price, y_train_price, X_test_scaled_price, y_test_price, y_train_price
from src.evaluate_model import model_evaluation
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,           # limite la profondeur des arbres
    min_samples_leaf=5,    # chaque feuille doit contenir au moins 5 exemples
    min_samples_split=10,  # empêche de diviser un nœud trop petit
    max_features='sqrt',   # limite le nombre de variables considérées à chaque split
    random_state=42
)
rf.fit(X_train_scaled_price, y_train_price)
y_pred = rf.predict(X_test_scaled_price)
print(model_evaluation(y_test_price, y_pred))

y_train_pred = rf.predict(X_train_scaled_price)
print("R² train:", r2_score(y_train_price, y_train_pred))