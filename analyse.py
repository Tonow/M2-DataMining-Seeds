import pandas as pd
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# import numpy as np

dataframe = pd.read_csv('seeds_dataset_classe-String.csv')
i = 0
for col in dataframe.columns:
    print(f'colonne {i}: {col}')
    i = i+1

# print(dataframe.describe(include='all'))
dataframe.describe()
