import matplotlib.pyplot as plt
import numpy as np

def plot_3d(series_list: list[list], labels: list[str], figsize=(5, 5)):
    """Plot 3D coordinates."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for s in series_list:
        x = [data[0] for data in s]
        y = [data[1] for data in s]
        z = [data[2] for data in s]
        ax.plot(x, y, z, marker='.', markersize=0.5, color=colorpallet.pop(0))
        
    plt.legend(labels)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_2d(series_list: list[list], labels: list[str], figsize=(5, 5)):
    """Plot 2D coordinates."""
    colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(figsize=figsize)
    for s in series_list:
        x = [data[0] for data in s]
        y = [data[1] for data in s]
        plt.plot(x, y, marker='.', markersize=0.5, color=colorpallet.pop(0))
    
    plt.legend(labels)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D plot of XY coordinates')
    plt.show()


def plot_directions(position_series_list: list[list], direction_vector_list: list[list], labels: list[str], figsize=(5, 5)):
    """Plot 2D coordinates."""
    colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(figsize=figsize)
    for serie, dirs in zip(position_series_list, direction_vector_list):
        color = colorpallet.pop(0)
        x = [data[0] for data in serie]
        y = [data[1] for data in serie]
        
        V = np.array(dirs)
        
        if len(V) > 0:
            plt.quiver(x, y, V[:, 0], V[:, 1], color=color, scale=3, units="xy", width=0.025)

    plt.legend(labels)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D plot of XY coordinates')
    plt.show()