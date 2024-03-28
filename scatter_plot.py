import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
import numpy as np
from utils import load_csv

def parse_data(df):
    data = df.drop(["Index", "Hogwarts House", "First Name","Last Name","Birthday","Best Hand"], axis=1)
    means = []
    for content in data:
        means.append(np.mean(data[content].dropna().values))
    for content, mean in zip(data, means):
        data[content].replace(np.nan, mean, inplace=True)
    return data

def main():
    try:
        df = load_csv(sys.argv[1])
        data = parse_data(df)
        num_scatter = len(list(data.columns)) - 1
        colors = plt.cm.tab20.colors[:num_scatter]
        for i, column in enumerate(data.columns[1:]):
            plt.scatter(data[column].index.values, data[column].values, color = colors[i], label=column)
        plt.ylabel("Score")
        plt.xlabel("Students")
        plt.title("Scatter Plot")
        plt.legend(loc="lower right")
        plt.show()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: scatter_plot.py dataset.csv")
