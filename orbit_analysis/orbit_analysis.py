import sys
sys.path.insert(0, '/home/skyler/school/ecen774/state_plotter/src/state_plotter')
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
        self.plotter = Plotter(plotting_freq, time_window=30)
        self.plotter.set_plots_per_row(2)

        # Define plot names
        plots = self._define_plots()

        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p)

        # Define state vectors for simpler input
        self._define_input_vectors()

        self.R_err_thresh = 0.01
        self.R_thresh_reached = False
        self.az_err_thresh = 0.001
        self.az_thresh_reached = False

    def _define_plots(self):
        plots = ['y x y_t x_t -2d',     '_R R_error',
                 '_psi',         '_az az_error',
                 'phi_c'
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("state", ['x', 'y', 'psi', 'az', 'R'])
        self.plotter.define_input_vector("target_pos", ['x_t', 'y_t'])

    def update(self, state, R_c, az_c, phi_c, target_pos, t):
        x, y, psi, az, R = state

        self.plotter.add_vector_measurement("state", state, t)
        self.plotter.add_measurement("phi_c", phi_c, t, rad2deg=True)
        self.plotter.add_vector_measurement("target_pos", target_pos, t)
        R_err = R - R_c
        if not self.R_thresh_reached and abs(R_err) < self.R_err_thresh:
            print("R error threshold ({0}) reached at t={1}".format(self.R_err_thresh, t))
            self.R_thresh_reached = True
        az_err = az - az_c
        if not self.az_thresh_reached and abs(az_err) < self.az_err_thresh:
            print("az error threshold ({0}) reached at t={1}".format(self.az_err_thresh, t))
            self.az_thresh_reached = True
        self.plotter.add_measurement("R_error", R_err, t)
        self.plotter.add_measurement("az_error", az_err, t, rad2deg=True)
        self.plotter.update_plots()

class OrbitAnalysis:
    def __init__(self):
        self.Va = 20.0
        self.h = 100.0
        self.g = 9.81
        self.phi_c = 0

        self.dt = 0.01
        self.ode_int_N = 2

        sec_per_update = 0.5
        freq = sec_per_update/self.dt
        self.plotter = OrbitPlotter(plotting_freq=freq)

        # Initial conditions
        t0 = 0
        x0 = -200
        y0 = -0
        psi0 = 0
        az0 = angle_wrap(np.pi + math.atan2(y0,x0) - psi0)
        R0 = math.sqrt(x0**2 + y0**2)
        self.state = [x0, y0, psi0, az0, R0]
        self.target_pos = np.array([0., 0.])
        self.target_vel = np.array([2.0, 0.])
        self.t = t0

        # Orbit control params
        self.R_desired = 100.0
        self.lam = -1.0 # 1=CW, -1=CCW
        az_to_R_ratio = 0.02027
        self.kp_az = 2.3
        self.kp_R = az_to_R_ratio*self.kp_az
        self.radius_max_error = 70.0
        self.phi_c_max = math.radians(45.0)

    def propagate(self):
        # Create time vector
        time = np.linspace(self.t, self.t + self.dt, self.ode_int_N)
        # Integrate states
        states = odeint(self._ode, self.state, time)
        # self.target_pos += self.target_vel*self.dt
        # Update control
        self.update_control()
        # Update time
        self.t += self.dt
        # Update plots
        for s, t in zip(states, time):
            self.plotter.update(s, self.R_desired, self.lam*math.radians(90.0), self.phi_c, self.target_pos, t)
        # Update state variable
        self.state = states[-1]

    def update_control(self):
        x, y, psi, az, R = self.state

        az = abs(az)
        phi_ff = math.atan(self.Va**2/(self.g*self.R_desired))
        radius_error = sat(self.R_desired - R, -self.radius_max_error, self.radius_max_error)
        phi_c = self.lam*(phi_ff - self.kp_az*(math.pi/2.0 - az) - self.kp_R*radius_error)
        self.phi_c = sat(phi_c, -self.phi_c_max, self.phi_c_max)

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

def sat(x, minimum, maximum):
    return min(max(x, minimum), maximum)



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
