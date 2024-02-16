import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def load_csv(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path in wrong format, .csv"
    df = pd.read_csv(path)
    return df

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
        print("Wrong argument: logreg_train.py dataset_train.csv")
