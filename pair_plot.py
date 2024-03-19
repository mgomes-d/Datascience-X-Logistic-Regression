import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from utils import load_csv

def main():
    try:
        # Charger les données
        data_train = load_csv(sys.argv[1])

        # Créer le pair plot ou la matrice de scatter plot
        pair_plot = sns.pairplot(data_train.drop(['Index', 'First Name', 'Last Name', 'Birthday'], axis=1), hue='Hogwarts House', height=1)

        # Afficher le pair plot
        plt.show()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: pair_plot.py dataset.csv")
