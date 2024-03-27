import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import matplotlib.figure
from logreg_train import LogisticRegression
from utils import load_csv
import threading

def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1).replace([np.nan], 0)
    return data

class LogisticGraph:
    def __init__(self, df):
        self.logistic_regression = LogisticRegression(df)
        self.all_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        self.thread = {}
        self.ax = {}
        self.active_threads = []
        for house in self.all_houses:
            self.thread[house] = False

    def update_graph(self, house, loss_values):
        self.root.after(100, self.update_graph_in_main_thread, house, loss_values)

    def update_graph_in_main_thread(self, house, loss_values):
        self.ax[house].clear()
        self.ax[house].plot(range(len(loss_values)), loss_values, label="New data")
        self.ax[house].set_title(house + " loss")
        self.ax[house].set_xlabel(f'Loss = {loss_values[-1]}')
        self.figures[house].canvas.draw()

    def start_thread(self, house, step_size, training_iterations):
        self.thread[house] = True
        self.logistic_regression.training(house, step_size, training_iterations)
        self.update_graph(house, self.logistic_regression.get_cost_and_reset(house))
        self.thread[house] = False

    def start_training(self, house, all_entry):
        try:
            step_size = float(all_entry[house][0].get())
            training_iterations = int(all_entry[house][1].get())
            assert step_size > 0 and training_iterations > 0, "Only Numbers > 0 are accepted"
            assert step_size <= 3, "Step_size too big"
            assert self.thread[house] is False, "Training in progress"
            training_thread = threading.Thread(target=self.start_thread, args=(house, step_size, training_iterations))
            training_thread.start()
            self.active_threads.append(training_thread)
            
        except Exception as msg:
            print("Error", msg)

    def create_graph_and_widgets(self, house, frame, all_entry):
        # fig, ax = plt.subplots(figsize=(4,7))#Win
        fig, ax = plt.subplots(figsize=(2,5))#Mac
        ax.set_title(house + " loss")
        self.figures[house] = fig
        self.ax[house] = ax

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        entry1 = tk.Entry(frame)
        # entry1.insert(0, "value of step_size")
        entry1.pack()

        entry2 = tk.Entry(frame)
        # entry2.insert(0, "Num of iterations")
        entry2.pack()
        all_entry[house] = (entry1, entry2)

        button = tk.Button(frame, text="Start training", command=lambda: self.start_training(house, all_entry))  
        button.pack()
    
    def init(self):
        self.root = tk.Tk()
        self.figures = {}
        # root.geometry("1200x800")

        frames = [tk.Frame(self.root) for _ in range(len(self.all_houses))]
        button = tk.Button(text="Store parameters", command=self.logistic_regression.store_parameters)  
        button.pack()
    
        all_entry = {}

        for house, frame in zip(self.all_houses, frames):
            self.create_graph_and_widgets(house, frame, all_entry)

        for frame in frames:
            frame.pack(side=tk.LEFT)

        self.root.mainloop()
        self.root.after(100, self.join_threads)

    def join_threads(self):
        for thread in self.active_threads:
            thread.join()
        self.active_threads = []


def main():
    try:
        data_train = load_csv("datasets/dataset_train.csv")
        data_parsed = parse_data(data_train)
        graph = LogisticGraph(data_parsed)
        graph.init()
    except Exception as msg:
        print(msg, "Error")


if __name__ == "__main__":
    main()
