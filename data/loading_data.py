import pandas as pd
import os

def data_loader():
    # Récupère le dossier où se trouve le script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construit le chemin complet vers le CSV
    file_path = os.path.join(script_dir, 'Housing-selected-columns.csv')
    return pd.read_csv(file_path)

data = data_loader()