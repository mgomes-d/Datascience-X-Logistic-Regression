import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import matplotlib.figure
from logreg_train import LogisticRegression

def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1).replace([np.nan], 0)
    return data

def get_training_values():
    text1 = training_iteration.get()
    text2 = step_size.get()
    print("text = ", text1, text2)

def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1).replace([np.nan], 0)
    return data
def make_graph():
    data_train = load_csv("datasets/dataset_train.csv")
    data_parsed = parse_data(data_train)
    #init Tkinter
    logistic_regression = LogisticRegression(data_parsed, True)
    logistic_regression.training()
    root = tk.Tk()
    # fig = matplotlib.figure.Figure()
    # ax = fig.subplots() 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Utilisation des sous-graphiques
    # ax1.plot([1, 2, 3], [4, 5, 6])
    # ax1.set_title('Sous-graphique 1')

    # ax2.plot([1, 2, 3], [4, 5, 6])
    # ax2.set_title('Sous-graphique 2')

    # ax3.plot([1, 2, 3], [4, 5, 6])
    # ax3.set_title('Sous-graphique 3')


    #Tkinter application

    frame = tk.Frame(root)

    canvas = FigureCanvasTkAgg(fig, master = frame)
    canvas.get_tk_widget().pack()
    frame.pack()

    training_iteration = tk.Entry(root, width=30)
    training_iteration.insert(0, "Iterations(ex: 200)")

    step_size = tk.Entry(root, width=30)
    step_size.insert(0, "Step size(ex: 0.10)")

    button = tk.Button(root, text="Hello", command=get_training_values)

    training_iteration.pack()
    step_size.pack()
    button.pack()

    root.mainloop()


def main():
    try:
        # data_train = load_csv("datasets/dataset_train.csv")
        # data_parsed = parse_data(data_train)
        make_graph()
        # x_data = [1, 2, 3]
        # y_data = [4, 5, 6]

        # Créer le graphique initial avec les données initiales
        # plt.plot(x_data, y_data, 'bo')  # Points bleus
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Mon graphique')
        # plt.grid(True)
        # plt.show(block=False)
        # input("Appuyez sur Entrée pour quitter...")  # Attendre l'entrée de l'utilisateur pour quitter

    except Exception as msg:
        print(msg, "Error")


if __name__ == "__main__":
    main()
