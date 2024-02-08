import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
import numpy as np

def load_csv(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path in wrong format, .csv"
    df = pd.read_csv(path)
    return df

def parse_data(df):
    data = df.set_index(df["Hogwarts House"]).select_dtypes(float).replace([np.nan], 0).sort_index(axis=0)
    return data

def main():
    try:
        df = load_csv(sys.argv[1])
        data = parse_data(df)
        data_hist = [data.values, data.columns.values]
        print(data.values)
        plt.hist(data.values,bins=4,facecolor='blue')
        plt.show()
        # print(df)

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: histogram.py dataset.csv")
