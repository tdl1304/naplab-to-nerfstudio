from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_coordinates(series_list: List[list], labels: List[str], figsize=(5, 5), is3D=False, show_scatter=True):
    """Plot 2D or 3D coordinates."""
    if is3D:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        plt.figure(figsize=figsize)
    colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3', 'b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3']
    for i, series in enumerate(series_list):
        c = colorpallet.pop(0)
        x = [data[0] for data in series]
        y = [data[1] for data in series]
        if is3D:
            z = [data[2] for data in series]
            ax.plot(x, y, z, marker='.', markersize=0.5, color=c, label=labels[i])
            if show_scatter:
                ax.scatter(x, y, z, color=c, marker='o')
        else:
            plt.plot(x, y, marker='.', markersize=0.5, color=c, label=labels[i])
            if show_scatter:
                plt.scatter(x, y, marker='.', color=c, s=100)
    
    if is3D:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('--------------------------------3D plot of XYZ coordinates--------------------------------')
    else:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('--------------------------------2D plot of XY coordinates--------------------------------')
        
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_directions(position_series_list: List[list], direction_vector_list: List[list], labels: List[str], figsize=(5, 5), is3D=False):
    """Plot 2D or 3D coordinates."""
    if is3D:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(65, 75)
    else:
        plt.figure(figsize=figsize)

    colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3', 'b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3']
    
    for serie, dirs in zip(position_series_list, direction_vector_list):
        color = colorpallet.pop(0)
        x = [data[0] for data in serie]
        y = [data[1] for data in serie]
        if is3D:
            z = [data[2] for data in serie]
        
        V = np.array(dirs)
        if is3D:
            ax.quiver(x, y, z, V[:, 0], V[:, 1], V[:, 2], color=color, length=1, arrow_length_ratio=.2)
        else:
            plt.quiver(x, y, V[:, 0], V[:, 1], color=color, scale=2, units="xy", width=0.025)

    if is3D:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        plt.xlabel('X')
        plt.ylabel('Y')
        
    plt.legend(labels)
    l = "----------------------------------"
    plt.title(f'{l}3D plot of XYZ coordinates{l}' if is3D else f'{l}2D plot of XY coordinates{l}')
    plt.tight_layout()
    plt.show()