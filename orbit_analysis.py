import sys
sys.path.insert(0, '/home/skyler/school/ecen631/state_plotter/src/state_plotter')
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

from Plotter import Plotter
import math
import numpy as np
import cv2
from scipy.integrate import odeint

## *GPS-denied Orbit Control Analysis*
## State equations:
# $$\mathbf{x} = \pmatrix{x \cry \cr \psi \cr \alpha_{az} \crR}, \mathbf{u}
# = \pmatrix{\phi}$$
## State Dynamics
# $$\frac{d}{dt}\mathbf{x} = f(\mathbf{x},\mathbf{u}) =\pmatrix{V_a \cos\psi
# \crV_a \sin\psi \cr\frac{g}{V_a}\tan\phi \cr\frac{V_a}{R^2} (x\sin\psi - y\cos\psi)
# - \frac{g}{V_a}\tan\phi \cr\frac{V_a}{R}(x\cos\psi + y\sin\psi)}$$
## Measurement model
# $$\mathbf{y} = \pmatrix{\alpha_{az} \cr \alpha_{el}} = h(\mathbf{x},\mathbf{u})
# = \pmatrix{- \frac{g}{V_a}\tan\phi \cr\tan^{-1}(\frac{-h}{R}) - \phi}$$
#
# Assume Va and h are constants.
#
#
## Define and Generate Initial conditions

class OrbitPlotter:
    ''' Plotter wrapper for orbit analysis '''
    def __init__(self, plotting_freq=1):
        self.plotter = Plotter(plotting_freq)
        self.plotter.set_plots_per_row(2)

        # Define plot names
        plots = self._define_plots()

        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p)

        # Define state vectors for simpler input
        self._define_input_vectors()

    def _define_plots(self):
        plots = ['y x -2d',     'R',
                 'psi',         'az',
                 'phi_c'
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("state", ['x', 'y', 'psi', 'az', 'R'])

    def update(self, state, phi_c, t):
        self.plotter.add_vector_measurement("state", state, t)
        self.plotter.add_measurement("phi_c", phi_c, t)
        self.plotter.update_plots()

class OrbitAnalysis:
    def __init__(self):
        self.Va = 20.0
        self.h = 100.0
        self.g = 9.81
        self.phi_c = math.radians(10)

        self.dt = 0.01
        self.ode_int_N = 2

        sec_per_update = 1.0
        freq = sec_per_update/self.dt
        self.plotter = OrbitPlotter(plotting_freq=freq)

        # Initial conditions
        t0 = 0
        x0 = -0
        y0 = -70
        psi0 = 0
        az0 = angle_wrap(np.pi + math.atan2(y0,x0) - psi0)
        R0 = math.sqrt(x0**2 + y0**2)

        self.state = [x0, y0, psi0, az0, R0]
        self.t = t0

    def propagate(self):
        time = np.linspace(self.t, self.t + self.dt, self.ode_int_N)
        states = odeint(self._ode, self.state, time)
        self.t += self.dt
        # Update plots
        for s, t in zip(states, time):
            self.plotter.update(s, self.phi_c, t)
        # Update state variable
        self.state = states[-1]

    def _ode(self, state, t):
        x, y, psi, az, R = state

        xdot = self.Va*math.cos(psi)
        ydot = self.Va*math.sin(psi)
        psidot = self.g/self.Va*math.tan(self.phi_c)
        azdot = self.Va/R**2 * (x*math.sin(psi) - y*math.cos(psi)) - self.g/self.Va*math.tan(self.phi_c)
        Rdot = self.Va/R * (x*math.cos(psi) + y*math.sin(psi))

        return [xdot, ydot, psidot, azdot, Rdot]

def angle_wrap(x):
    xwrap = np.array(np.mod(x, 2*np.pi))
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])
    return xwrap


if __name__ == "__main__":
    analysis = OrbitAnalysis()
    cv2.namedWindow("SPACE = pause, ESC = exit")

    paused = False

    while True:
        if not paused:
            analysis.propagate()
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused ^= True
            if paused:
                print("Paused")
            else:
                print("Resumed")
        elif key == 27:
            print("Quit")
            sys.exit()
