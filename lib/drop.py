import pandas as pd

def dropcolumn(path, dropcolumn):
    datacsv = pd.read_csv(path)
    datacsv.drop(columns=dropcolumn, inplace=True)
    datacsv.to_csv(path, index=False)
    return path
