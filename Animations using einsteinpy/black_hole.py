import numpy as np
#from einsteinpy.geodesic import TimeLike
#from einsteinpy.plotting import StaticGeodesicPlotter 
import matplotlib.pyplot as plt
import random
import warnings
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d


class StaticGeodesicPlotter:

    def __init__(self, ax=None, bh_colors = ('#000', '#FFC'), draw_ergosphere = True):
        self.ax = ax
        self.bh_colors = bh_colors
        self.draw_ergosphere = draw_ergosphere

    def draw_bh(self, a, figsize=(6,6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        fontsize=max(figsize) + 3
        self.fig.set_size_inches(figsize)
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlabel("$X\\:(GM/c^2)$", fontsize=fontsize)
        self.ax.set_ylabel("$Y\\:(GM/c^2)$", fontsize=fontsize)
        self.ax.set_zlabel("$Z\\:(GM/c^2)$", fontsize=fontsize)
        
        theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
        THETA, PHI = np.meshgrid(theta, phi)

        rh_outer = 1 + np.sqrt(1 - a**2) #raio do horizonte de eventos
        
        #conversão de coordenadas esféricas para cartesianas
        XH = rh_outer * np.sin(PHI) * np.cos(THETA)
        YH = rh_outer * np.sin(PHI) * np.sin(THETA)
        ZH = rh_outer * np.cos(PHI)

        surface1 = self.ax.plot_surface(
            XH,
            YH,
            ZH,
            rstride=1,
            cstride=1,
            color=self.bh_colors[0],
            antialiased=False,
            alpha=0.2,
            label="BH Event Horizon (Outer)",
        )

        #correção de cor
        surface1._facecolors2d = surface1._facecolor3d
        surface1._edgecolors2d = surface1._edgecolor3d

        #desenhar ergosfera, caso haja
        if self.draw_ergosphere:
            rE_outer = 1 + np.sqrt(1 - (a * np.cos(THETA) ** 2))

            XE = rE_outer * np.sin(PHI) * np.sin(THETA)
            YE = rE_outer * np.sin(PHI) * np.cos(THETA)
            ZE = rE_outer * np.cos(PHI)

            surface2 = self.ax.plot_surface(
                XE,
                YE,
                ZE,
                rstride=1,
                cstride=1,
                color=self.bh_colors[1],
                antialiased=False,
                alpha=0.1,
                label="BH Ergosphere (Outer)",
            )

            #correção de cor
            surface2._facecolors2d = surface2._facecolor3d
            surface2._edgecolors2d = surface2._edgecolor3d

        plt.legend()
        plt.show()
#class timelike(geodesic):

# Exemplo de uso
if __name__ == "__main__":
    plotter = StaticGeodesicPlotter()
    plotter.draw_bh(a=0.8, figsize=(7, 7))  

""""
Import, from the source, the function "time like" geodesic. A timelike geodesic is the curve with the longest proper time
between two events
"""


## >>>> OBJETIVO <<<<
""""
position = [4, np.pi, 3, 0.]
momentum = [0., 0.767851, 2.]
a = 0.99
steps = 400.
delta = 0.5

geodesic = TimeLike(
    metric="Kerr",
    metric_params = (a,),
    position = position,
    momentum = momentum,
    steps = steps,
    delta = delta,

    return_cartesian = True
)


# Testando no VS Code
draw_bh(a=0.9)"""