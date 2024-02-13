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
    data = df.set_index("Hogwarts House").select_dtypes(include=[float]).replace([np.nan], 0)
    return data

def main():
    try:
        df = load_csv(sys.argv[1])
        data = parse_data(df)
        data.insert(0, "Number_Student", df["Index"].values)
        num_scatter = len(list(data.columns)) - 1
        # print(data)
        # house_name = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
        # house_colors = ['b', 'g', 'r', 'y']
        g = pd.plotting.scatter_matrix(df[1:], figsize=(8,8), marker = 'o', hist_kwds = {'bins': 2}, s = 60, alpha = 0.8)
        # num_rows = 4
        # num_cols = 4
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(8,8))
        # axes_flat = axes.flatten()

        # for house, color in zip(house_name, house_colors):
        #     for i, (column, ax) in enumerate(zip(data.columns[1:], axes_flat)):
        #         data.loc[house].set_index(data.loc[house]["Number_Student"])[column].plot.scatter(x=data.loc[house][column], y=data.loc[house]["Number_Student"], ax=ax, color=color)
                # print(data.loc[house].set_index(data.loc[house]["Number_Student"])[column])
                # plt.scatter(data.loc[house][column], data.loc[house]["Number_Student"], ax=ax, color=color)
                # data.loc[house][column].plot.scatter(ax=ax, color=color)
                # students_values = data.loc[house]["Number_Student"]
                # score_values
                # scatter_values = temp["Number_Student", column]
                # print(data.loc[house][column])
                # print(scatter_values)
                # ax.set_title(f'{column}', fontsize = 7)
            # break
            # test = data.loc[house]
            # print(test)
            # print("f")
        plt.show()

        # for i, column in enumerate(data.columns[1:]):
        #     plt.scatter(data[column].index.values, data[column].values, color = colors[i], label=column)
        # plt.legend(loc="upper left")
        # plt.show()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: pair_plot.py dataset.csv")
