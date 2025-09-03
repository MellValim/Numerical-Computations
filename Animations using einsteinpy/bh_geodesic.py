import numpy as np
from .utils import _P, _kerr, _kerrnewman, _schwarzschild, _Z, _flow_A, _flow_B, _flow_mixed
import matplotlib.pyplot as plt
import random
import warnings
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d


class integrators:
    """
    -ref: <https://doi.org/10.3847/1538-4357/abdc28>
    -time_like = True: Um valor booleano (True ou False) que determina se a geodésica é do tipo tempo (para objetos com massa) 
    ou nulo (para fótons, ou seja, partículas sem massa).

    -definindo os parametros iniciais que o intefrador vai receber.
        rtol e atol: São as tolerâncias relativa e absoluta usadas para a verificação de erros numéricos. O integrador usa esses valores para garantir que os cálculos permaneçam precisos.
        order: A ordem do integrador que será usada. O código suporta ordens 2, 4, 6 e 8. Quanto maior a ordem, mais preciso (e computacionalmente mais caro) é o cálculo.
        omega: Um parâmetro de acoplamento que afeta a precisão e a estabilidade da integração. Ele é especialmente importante para simular trajetórias que se aproximam de singularidades ou que são "pastoreadas" (grazing geodesics).
        suppress_warnings: Um booleano que, se definido como True, desativa os avisos sobre erros numéricos que excedem as tolerâncias rtol e atol.
  
        Geodesic Integrator, based on [1]_.
        This module uses Forward Mode Automatic Differentiation
        to calculate metric derivatives to machine precision
        leading to stable simulations.
    """
    def __init__(
        self,
        metric,
        metric_params,
        q0,
        p0,
        time_like=True,
        steps=100,
        delta=0.5,
        rtol=1e-2,
        atol=1e-2,
        order=2,
        omega=1.0,
        suppress_warnings=False,
    ):
        """
        Constructor

        Parameters
        ----------
        metric : callable
            Metric Function. Currently, these metrics are supported:
            1. Schwarzschild
            2. Kerr
            3. KerrNewman
        metric_params : array_like
            Tuple of parameters to pass to the metric
            E.g., ``(a,)`` for Kerr
        q0 : array_like
            Initial 4-Position
        p0 : array_like
            Initial 4-Momentum
        time_like : bool, optional
            Determines type of Geodesic
            ``True`` for Time-like geodesics
            ``False`` for Null-like geodesics
            Defaults to ``True``
        steps : int
            Number of integration steps
            Defaults to ``50``
        delta : float
            Initial integration step-size
            Defaults to ``0.5``
        rtol : float
            Relative Tolerance
            Defaults to ``1e-2``
        atol : float
            Absolute Tolerance
            Defaults to ``1e-2``
        order : int
            Integration Order
            Defaults to ``2``
        omega : float
            Coupling between Hamiltonian Flows
            Smaller values imply smaller integration error, but too
            small values can make the equation of motion non-integrable.
            For non-capture trajectories, ``omega = 1.0`` is recommended.
            For trajectories, that either lead to a capture or a grazing
            geodesic, a decreased value of ``0.01`` or less is recommended.
            Defaults to ``1.0``
        suppress_warnings : bool
            Whether to suppress warnings during simulation
            Warnings are shown for every step, where numerical errors
            exceed specified tolerance (controlled by ``rtol`` and ``atol``)
            Defaults to ``False``

        Raises
        ------
        NotImplementedError
            If ``order`` is not in [2, 4, 6, 8]

        """
        ORDERS = {
            2: self._ord_2,
            4: self._ord_4,
            6: self._ord_6,
            8: self._ord_8,
        }
        self.metric = metric
        self.metric_params = metric_params
        self.q0 = q0
        self.p0 = p0
        self.time_like = time_like
        self.steps = steps
        self.delta = delta
        self.omega = omega
        if order not in ORDERS:
            raise NotImplementedError(
                f"Order {order} integrator has not been implemented."
            )
        self.order = order
        self.integrator = ORDERS[order]
        self.rtol = rtol
        self.atol = atol
        self.suppress_warnings = suppress_warnings

        self.step_num = 0
        self.res_list = [q0, p0, q0, p0]
        self.results = list()

    def __str__(self):
        return f"""{self.__class__.__name__}(\n\
                metric : {self.metric}\n\
                metric_params : {self.metric_params}\n\
                q0 : {self.q0},\n\
                p0 : {self.p0},\n\
                time_like : {self.time_like},\n\
                steps : {self.steps},\n\
                delta : {self.delta},\n\
                omega : {self.omega},\n\
                order : {self.order},\n\
                rtol : {self.rtol},\n\
                atol : {self.atol}\n\
                suppress_warnings : {self.suppress_warnings}
            )"""

    def __repr__(self):
        return self.__str__()

    def _ord_2(self, q1, p1, q2, p2, delta):
        """
        Order 2 Integration Scheme

        References
        ----------
        .. [1] Christian, Pierre and Chan, Chi-Kwan;
            "FANTASY : User-Friendly Symplectic Geodesic Integrator
            for Arbitrary Metrics with Automatic Differentiation";
            `2021 ApJ 909 67 <https://doi.org/10.3847/1538-4357/abdc28>`__

        """
        dl, omg = delta, self.omega
        g = self.metric
        g_prms = self.metric_params

        HA1 = np.array(
            [
                q1,
                _flow_A(g, g_prms, q1, p1, q2, p2, 0.5 * dl)[1],
                _flow_A(g, g_prms, q1, p1, q2, p2, 0.5 * dl)[0],
                p2,
            ]
        )
        HB1 = np.array(
            [
                _flow_B(g, g_prms, HA1[0], HA1[1], HA1[2], HA1[3], 0.5 * dl)[0],
                HA1[1],
                HA1[2],
                _flow_B(g, g_prms, HA1[0], HA1[1], HA1[2], HA1[3], 0.5 * dl)[1],
            ]
        )
        HC = _flow_mixed(HB1[0], HB1[1], HB1[2], HB1[3], dl, omg)
        HB2 = np.array(
            [
                _flow_B(g, g_prms, HC[0], HC[1], HC[2], HC[3], 0.5 * dl)[0],
                HC[1],
                HC[2],
                _flow_B(g, g_prms, HC[0], HC[1], HC[2], HC[3], 0.5 * dl)[1],
            ]
        )
        HA2 = np.array(
            [
                HB2[0],
                _flow_A(g, g_prms, HB2[0], HB2[1], HB2[2], HB2[3], 0.5 * dl)[1],
                _flow_A(g, g_prms, HB2[0], HB2[1], HB2[2], HB2[3], 0.5 * dl)[0],
                HB2[3],
            ]
        )

        return HA2

    def _ord_4(self, q1, p1, q2, p2, delta):
        """
        Order 4 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_2(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_2(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_2(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def _ord_6(self, q1, p1, q2, p2, delta):
        """
        Order 6 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_4(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_4(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_4(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def _ord_8(self, q1, p1, q2, p2, delta):
        """
        Order 8 Integration Scheme

        References
        ----------
        .. [1] Yoshida, Haruo,
            "Construction of higher order symplectic integrators";
             Physics Letters A, vol. 150, no. 5-7, pp. 262-268, 1990.
            `DOI: <https://doi.org/10.1016/0375-9601(90)90092-3>`__

        """
        dl = delta

        Z0, Z1 = _Z(self.order)
        step1 = self._ord_6(q1, p1, q2, p2, dl * Z1)
        step2 = self._ord_6(step1[0], step1[1], step1[2], step1[3], dl * Z0)
        step3 = self._ord_6(step2[0], step2[1], step2[2], step2[3], dl * Z1)

        return step3

    def step(self):
        """
        Advances integration by one step

        """
        rl = self.res_list

        arr = self.integrator(rl[0], rl[1], rl[2], rl[3], self.delta)

        self.res_list = arr
        self.step_num += 1

        # Stability check
        if not self.suppress_warnings:
            g = self.metric
            g_prms = self.metric_params

            q1 = arr[0]
            p1 = arr[1]
            # Ignoring
            # q_2 = arr[2]
            # p_2 = arr[3]

            const = -int(self.time_like)
            # g.p.p ~ -1 or 0 (const)
            if not np.allclose(
                g(q1, *g_prms) @ p1 @ p1, const, rtol=self.rtol, atol=self.atol
            ):
                warnings.warn(
                    f"Numerical error has exceeded specified tolerance at step = {self.step_num}.",
                    RuntimeWarning,
                )

        self.results.append(self.res_list)

class Geodesic:
    import numpy as np

# Importe a classe GeodesicIntegrator e outras funções auxiliares do seu projeto
from .geodesic_integrator import GeodesicIntegrator  # Exemplo de importação
from .utils import _P, _kerr, _kerrnewman, _schwarzschild, _Z, _flow_A, _flow_B, _flow_mixed

class Geodesic:
    """
    Definindo geodesicas
     c = G = M = ke = 1.
    """

    #Inicializando um objeto geometrico
    def __init__(
        self,
        metric: str,
        metric_params: tuple,
        position: np.ndarray,
        momentum: np.ndarray,
        time_like: bool = True,
        return_cartesian: bool = True,
        **kwargs
    ):
        """
        Args:
            metric (str): The name of the metric (e.g., 'Schwarzschild', 'Kerr').
            metric_params (tuple): Parameters for the metric (e.g., (a,) for Kerr).
            position (np.ndarray): The 3-position [r, theta, phi].
            momentum (np.ndarray): The 3-momentum [pr, p_theta, p_phi].
            time_like (bool, optional): True for time-like geodesics (massive particles),
                                        False for null-like (photons). Defaults to True.
            return_cartesian (bool, optional): Whether to return positions in Cartesian
                                               coordinates. Defaults to True.
            **kwargs: Additional keyword arguments for the GeodesicIntegrator.
        """
        _METRICS = {
            "Schwarzschild": _schwarzschild,
            "Kerr": _kerr,
            "KerrNewman": _kerrnewman,
        }

        if metric not in _METRICS and not callable(metric):
            raise NotImplementedError(
                f"Metric '{metric}' não suportado. As métricas disponíveis são: "
                f"{list(_METRICS.keys())}"
            )

        self.metric_name = metric.__name__ if callable(metric) else metric
        self.metric = metric if callable(metric) else _METRICS[metric]
        self.metric_params = (0.0,) if metric == "Schwarzschild" else metric_params

        self.position = np.array([0.0, *position])
        self.momentum = _P(self.metric, self.metric_params, self.position, momentum, time_like)
        self.time_like = time_like
        self.return_cartesian = return_cartesian

        # Armazena os kwargs para passar para o integrador
        self.integrator_kwargs = kwargs
        
        self._trajectory = self._calculate_trajectory()

    def __repr__(self):
        """String representation of the Geodesic object."""
        return (
            f"Geodesic Object:\n"
            f"  Type: {'Time-like' if self.time_like else 'Null-like'}\n"
            f"  Metric: {self.metric_name}\n"
            f"  Metric Parameters: {self.metric_params}\n"
            f"  Initial 4-Position: {self.position}\n"
            f"  Initial 4-Momentum: {self.momentum}\n"
            f"  Trajectory: (see .trajectory for full data)\n"
            f"  Output Position Coordinate System: {'Cartesian' if self.return_cartesian else 'Spherical Polar'}"
        )

    def __str__(self):
        return self.__repr__()

    @property
    def trajectory(self) -> np.ndarray:
        """Returns the calculated trajectory of the test particle."""
        return self._trajectory

    def _calculate_trajectory(self) -> np.ndarray:
        """Calculates the trajectory in spacetime."""
        geod_integrator = GeodesicIntegrator(
            metric=self.metric,
            metric_params=self.metric_params,
            q0=self.position,
            p0=self.momentum,
            time_like=self.time_like,
            **self.integrator_kwargs
        )

        steps = self.integrator_kwargs.get("steps", 50)
        for _ in range(steps):
            geod_integrator.step()

        vectors = np.array(geod_integrator.results, dtype=float)
        positions = vectors[:, 0]
        momenta = vectors[:, 1]
        
        # Converte para Cartesiano se solicitado
        if self.return_cartesian:
            t, r, th, ph = positions.T
            pt, pr, pth, pph = momenta.T
            x = r * np.sin(th) * np.cos(ph)
            y = r * np.sin(th) * np.sin(ph)
            z = r * np.cos(th)

            return np.vstack((t, x, y, z, pt, pr, pth, pph)).T

        return np.hstack((positions, momenta))
    
class TimeLike(Geodesic):
    """
    Classe derivada de Geodesic para geodésicas do tipo tempo (time-like).
    """
    def __init__(self, **kwargs):
        super().__init__(time_like=True, **kwargs)

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

        rE_outer = 1 + np.sqrt(1 - (a * np.cos(THETA) ** 2))
        
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
        plt.show()#class timelike(geodesic):

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