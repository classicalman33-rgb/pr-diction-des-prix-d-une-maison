from sklearn.linear_model import LinearRegression, Ridge, Lasso


def train(models, X_train, y_train):
    

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model

    return models