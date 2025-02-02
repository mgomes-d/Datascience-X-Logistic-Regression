import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd
import numpy as np
from utils.utils import load_csv

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
        # Number of histograms to display
        num_histograms = len(list(data.columns))

        # Create a 4x4 grid to subplots to accommadate
        num_rows = 4
        num_cols = 4

        #create a figure and subplors
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
        
        # Flatten the axes array to iterate through subplots easily
        axes_flat = axes.flatten()

        #Get a list of (16) distinct colors from the tab20 colormao
        colors = plt.cm.tab20.colors[:num_histograms]

        #Iterate through the Dataframe columns and plot histograms with distinct colors
        for i, (column, ax) in enumerate(zip(data.columns, axes_flat)):
            data[column].plot.hist(ax=ax, bins=15, color=colors[i], edgecolor='black')
            ax.set_title(f'{column}', fontsize = 7)
            ax.set_xlabel("score", fontsize = 7)
            ax.set_ylabel("students", fontsize = 7)
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: histogram.py dataset.csv")
