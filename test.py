import matplotlib.pyplot as plt
import time
import numpy as np
import time

class Classtest:
    def __init__(self):
        plt.show(block=False)
    def hello():
        print("hello")

def test():
    for i in range(7, 1000):
        # Calculer le résultat
        result = (i - 3) * i
        
        # Afficher le résultat
        print(f"Résultat de l'étape {i+1}: {result}")
        
        # Ajouter le nouveau point au graphique
        plt.plot((i - 3), i, 'ro')  # Nouveau point rouge
        
        # Mettre à jour le graphique pour afficher le nouveau point
        # plt.draw()
        
        # Attendre un certain temps pour que l'utilisateur puisse voir le graphique
        plt.pause(0.02)  # Pause d'une seconde

def graph():
    plt.show(block=False)


def main():
    # x = np.random.randint(20, size=20)
    # y = np.random.randint(30, size=20)

    # z = []
    # for i in range(1, 21):
    #     z.append("Point" + str(i))

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)

    # plt.scatter(x, y, c='green', s=300)
    # plt.show()
    # Données initiales
    x_data = [1, 2, 3]
    y_data = [4, 5, 6]

    # Créer le graphique initial avec les données initiales
    # plt.plot(x_data, y_data, 'bo')  # Points bleus
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Mon graphique')
    # test11 = Classtest()
    # plt.grid(True)
    # plt.show(block=False)  # Afficher le graphique sans bloquer l'exécution du script
    # graph()
    # Données pour les nouveaux points
    # new_x = [4, 5, 6]
    # new_y = [7, 8, 9]

    # # Ajouter de nouveaux points et mettre à jour le graphique
    # for i in range(7, 1000):
    #     # Calculer le résultat
    #     result = (i - 3) * i
        
    #     # Afficher le résultat
    #     print(f"Résultat de l'étape {i+1}: {result}")
        
    #     # Ajouter le nouveau point au graphique
    #     plt.plot((i - 3), i, 'ro')  # Nouveau point rouge
        
    #     # Mettre à jour le graphique pour afficher le nouveau point
    #     plt.draw()
        
    #     # Attendre un certain temps pour que l'utilisateur puisse voir le graphique
    #     plt.pause(0.02)  # Pause d'une seconde
    test()
    # Attendre l'interaction de l'utilisateur pour fermer le graphique
    plt.pause(0.1)  # Temps de pause en secondes
    plt.show()  # Attendre l'entrée de l'utilisateur pour quitter

if __name__ == "__main__":
    main()
