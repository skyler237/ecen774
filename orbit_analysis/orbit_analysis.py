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
from PID import PID

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
        plots = ['y_r x_r y_t_r x_t_r -2d', 'y x y_t x_t -2d',
                 '_R R_error',      '_az az_error',
                 'phi_c',      'psi',
                 'vx vx_e -l', 'vy vy_e'
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("state", ['x_r', 'y_r', 'psi', 'az', 'R'])
        self.plotter.define_input_vector("actual_pos", ['x', 'y'])
        self.plotter.define_input_vector("target_pos", ['x_t', 'y_t'])
        self.plotter.define_input_vector("target_rel_pos", ['x_t_r', 'y_t_r'])
        self.plotter.define_input_vector("target_vel", ['vx', 'vy'])
        self.plotter.define_input_vector("target_vel_est", ['vx_e', 'vy_e'])

    def update(self, state, R_c, az_c, phi_c, target_pos, target_vel, target_vel_est, t):
        x_r, y_r, psi, az, R = state

        self.plotter.add_vector_measurement("state", state, t)
        actual_pos = [x_r + target_pos[0], y_r + target_pos[1]]
        self.plotter.add_vector_measurement("actual_pos", actual_pos, t)
        self.plotter.add_measurement("phi_c", phi_c, t, rad2deg=True)
        self.plotter.add_vector_measurement("target_pos", target_pos, t)
        self.plotter.add_vector_measurement("target_rel_pos", [0, 0], t)
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
        self.plotter.add_vector_measurement("target_vel", target_vel, t)
        self.plotter.add_vector_measurement("target_vel_est", target_vel_est, t)
        self.plotter.update_plots()

class OrbitAnalysis:
    def __init__(self):
        self.Va = 20.0
        self.g = 9.81
        self.phi_c = 0

        self.dt = 0.01
        self.ode_int_N = 2

        sec_per_update = 0.5
        freq = sec_per_update/self.dt
        self.plotter = OrbitPlotter(plotting_freq=freq)

        # Initial conditions
        t0 = 0
        x0 = -300
        y0 = 0
        psi0 = math.radians(90)
        az0 = angle_wrap(np.pi + math.atan2(y0,x0) - psi0)
        R0 = math.sqrt(x0**2 + y0**2)
        self.state = [x0, y0, psi0, az0, R0]
        self.target_pos = np.array([0., 0.])
        self.target_vel = np.array([3.0, 2.0])
        self.t = t0

        # Orbit control params
        self.R_desired = 100.0
        self.lam = 1.0 # 1=CW, -1=CCW
        az_to_R_ratio = 0.0202
        kp_az = 3.0
        kp_R = 8.0
        # kp_R = 20.0
        kd_az = 1.0
        kd_R = 4.0
        # kd_R = 15.0
        ki_az = 0.0
        ki_R = 0.0
        self.az_PID = PID(kp_az, kd_az, ki_az)
        self.R_PID = PID(kp_R, kd_R, ki_R)
        self.Raz_PID = PID(0, 0, 0)
        self.Raz_err_thresh = 0.00
        self.radius_max_error = 30.0
        self.phi_c_max = math.radians(45.0)

        # Velocity estimation params
        self.target_vel_est = np.array([0., 0.])
        self.k_vel_est = 1.0e-3

    def propagate(self):
        # Create time vector
        time = np.linspace(self.t, self.t + self.dt, self.ode_int_N)
        # Integrate states
        states = odeint(self._ode, self.state, time)
        self.target_pos += self.target_vel*self.dt
        # Update control
        self.update_control()
        # Update time
        self.t += self.dt
        # Update plots
        for s, t in zip(states, time):
            self.plotter.update(s, self.R_desired, self.lam*math.radians(90.0),
                                self.phi_c, self.target_pos, self.target_vel, self.target_vel_est, t)
        # Update state variable
        self.state = states[-1]

    def update_control(self):
        x, y, psi, az, R = self.state

        Vg_sq = ((self.Va*math.cos(psi)-self.target_vel_est[0])**2 + (self.Va*math.sin(psi)-self.target_vel_est[1])**2)
        phi_ff = math.atan(Vg_sq/(self.R_desired*self.g))
        Vg = math.sqrt(Vg_sq)
        sign = -self.lam*np.sign(self.target_vel_est[1]*math.cos(psi) - self.target_vel_est[0]*math.sin(psi))
        az_err_pred = sign*math.acos((self.Va - self.target_vel_est[0]*math.cos(psi) - self.target_vel_est[1]*math.sin(psi))/Vg)

        az = angle_wrap(az)
        az_err = abs(az) - math.pi/2.0 - az_err_pred
        phi_az = self.az_PID.compute_control_error(az_err, self.dt)

        # print("sign = {0}".format(sign))
        # print("az_err_pred = {0}".format(az_err_pred))
        # print("az_err = {0}".format(az_err))
        # print("diff = {0}".format(az_err_pred - az_err))
        # print("phi_az = {0}".format(phi_az))
        # print("phi_ff = {0}".format(phi_ff))

        R_ratio = R/self.R_desired
        if R_ratio < 1:
            p = 8.0
            R_err = 1.0/p*R_ratio**p - 1.0/p
            # R_err = math.exp(R_ratio - 1) - 1
        else:
            R_err = math.log(R/self.R_desired)
        # R_err = math.atan(R - self.R_desired)
        # # Adapt gains based on distance
        # R_gains = self.get_R_gains(R_err)
        # self.R_PID.set_gains(*R_gains)

        print("R_err = {0}".format(R_err))


        # Compute control
        Raz_gains = self.R_PID.compute_control_error(R_err, self.dt, vector_output=True)
        self.Raz_PID.set_gains(*Raz_gains)
        Raz_err = az-az_err_pred
        if abs(Raz_err) < self.Raz_err_thresh:
            Raz_err = 0
        phi_R = self.Raz_PID.compute_control_error(Raz_err, self.dt)

        # phi_ff = math.atan(self.Va**2/(self.g*self.R_desired))

        phi_c = self.lam*(phi_ff + phi_az) + phi_R
        
        self.phi_c = sat(phi_c, -self.phi_c_max, self.phi_c_max)

    def get_R_gains(self, R_err):
        kp_R_switch = 10.0
        kp_R_blend_rate = 0.3
        kp_R_near = lambda e: 0.05*abs(e)
        kp_R_far = lambda e: 0.005
        kp_R = self.blend_func(R_err, kp_R_near, kp_R_far, kp_R_switch, kp_R_blend_rate)

        kd_R_switch = 00.0
        kd_R_blend_rate = 0.3
        kd_R_near = lambda e: 0.05*abs(e)
        kd_R_far = lambda e: 0.005
        kd_R = self.blend_func(R_err, kd_R_near, kd_R_far, kd_R_switch, kd_R_blend_rate)

        ki_R = 0

        return [kp_R, kd_R, ki_R]

    def blend_func(self, x, f1, f2, x_blend, rate):
        sigma = self.sigmoid(x, x_blend, rate)
        near = f1(x)
        far = f2(x)
        blended = (1-sigma)*near + sigma*far
        return blended

    def sigmoid(self, alpha, alpha0, M):
        top = 1.0 + math.exp(-M*(alpha - alpha0)) + math.exp(M*(alpha+alpha0))
        bottom = (1.0 + math.exp(-M*(alpha - alpha0)))*(1.0 + math.exp(M*(alpha+alpha0)))
        return top/bottom

    def _ode(self, state, t):
        x, y, psi, az, R = state
        vx_t, vy_t = self.target_vel
        if R == 0:
            R = 1e-10

        xdot = self.Va*math.cos(psi) - vx_t
        ydot = self.Va*math.sin(psi) - vy_t
        psidot = self.g/self.Va*math.tan(self.phi_c)
        azdot = (self.Va * (x*math.sin(psi) - y*math.cos(psi)) + y*vx_t - x*vy_t)/R**2  - self.g/self.Va*math.tan(self.phi_c)
        Rdot = self.Va/R * (x*math.cos(psi) + y*math.sin(psi)) - (x*vx_t + y*vy_t)/R

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
