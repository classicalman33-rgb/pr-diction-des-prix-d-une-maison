from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np


def best_params(X_train, y_train):
    alphas = np.logspace(-3, 3, 50)  # teste 50 valeurs entre 0.001 et 1000

    # Ridge
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train, y_train)
    alpha_ridge = ridge_cv.alpha_

    # Lasso
    lasso_cv = LassoCV(alphas=alphas, cv=5)
    lasso_cv.fit(X_train, y_train)
    alpha_lasso = lasso_cv.alpha_

    return alpha_ridge, alpha_lasso