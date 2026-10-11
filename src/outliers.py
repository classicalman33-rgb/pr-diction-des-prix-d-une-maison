def find_outliers(data):
    out = {}

    for col in ['price',  'area',  'bedrooms',  'bathrooms',  'stories']:
        Q75 = data[col].quantile(.75)
        Q25 = data[col].quantile(.25)
        IQR = Q75 - Q25
        upper = Q75 + (1.5 * IQR)
        lower = Q25 - (1.5 * IQR)
        out[col] = float(((data[col] < lower) | (data[col] > upper)).sum())
    return out
