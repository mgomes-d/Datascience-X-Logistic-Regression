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
    data = df.select_dtypes(float).replace([np.nan], 0)
    return data

def main():
    try:
        df = load_csv(sys.argv[1])
        data = parse_data(df)
        num_scatter = len(list(data.columns)) - 1
        colors = plt.cm.tab20.colors[:num_scatter]
        for i, column in enumerate(data.columns[1:]):
            plt.scatter(data[column].index.values, data[column].values, color = colors[i], label=column)
        plt.legend(loc="upper left")
        plt.show()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: histogram.py dataset.csv")
