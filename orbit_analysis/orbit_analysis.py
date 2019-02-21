import sys
# sys.path.insert(0, '/home/skyler backup/school/ecen774/state_plotter/src/state_plotter')
sys.path.insert(0, '/home/skyler/handoff/catkin_ws/src/state_plotter/src')
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

from IPython.core.debugger import set_trace
from state_plotter.Plotter import Plotter
import math
import numpy as np
import cv2
from scipy.integrate import odeint
from PID import PID
from KalmanFilter import KalmanFilter

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
            self.plotter.add_plotboxes(p)

        # Define state vectors for simpler input
        self._define_input_vectors()

        self.R_err_thresh = 0.01
        self.R_thresh_reached = False
        self.az_err_thresh = 0.001
        self.az_thresh_reached = False

    def _define_plots(self):

        plots = [['y_r x_r y_t_r x_t_r -2d', 'y x y_t x_t -2d'],
                 ['_R R_error',      '_az az_error'],
                 ['phi_c',      'psi'],
                 ['vx vx_e -l', 'vy vy_e']
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("state", ['x_r', 'y_r', 'psi', 'az', 'R'])
        self.plotter.define_input_vector("actual_pos", ['x', 'y'])
        self.plotter.define_input_vector("target_pos", ['x_t', 'y_t'])
        self.plotter.define_input_vector("target_rel_pos", ['x_t_r', 'y_t_r'])
        self.plotter.define_input_vector("target_vel", ['vx', 'vy'])
        self.plotter.define_input_vector("target_vel_est", ['vx_e', 'vy_e'])

    def update(self, state, R_c, az_err, phi_c, target_pos, target_vel, target_vel_est, t):
        x_r, y_r, psi, az, R = state

        self.plotter.add_vector_measurement("state", state, t)
        actual_pos = [x_r + target_pos[0], y_r + target_pos[1]]
        self.plotter.add_vector_measurement("actual_pos", actual_pos, t)
        self.plotter.add_measurement("phi_c", phi_c, t)
        self.plotter.add_vector_measurement("target_pos", target_pos, t)
        self.plotter.add_vector_measurement("target_rel_pos", [0, 0], t)
        R_err = R - R_c
        if not self.R_thresh_reached and abs(R_err) < self.R_err_thresh:
            print("R error threshold ({0}) reached at t={1}".format(self.R_err_thresh, t))
            self.R_thresh_reached = True

        if not self.az_thresh_reached and abs(az_err) < self.az_err_thresh:
            print("az error threshold ({0}) reached at t={1}".format(self.az_err_thresh, t))
            self.az_thresh_reached = True
        self.plotter.add_measurement("R_error", R_err, t)
        self.plotter.add_measurement("az_error", az_err, t)
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
        self.target_vel = np.array([3.0, 1.0])
        self.az_err = 0
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
        kp_gamma = 10.0
        kd_gamma = 1.0
        ki_gamma = 0.0
        self.gamma_PID = PID(kp_gamma, kd_gamma, ki_gamma)

        # Velocity estimation params
        self.target_vel_est = np.array([1e-3]*2)
        self.R_avg = 0
        self.vel_est_alpha = 0.9
        self.vel_filter = VelocityFilter(self.dt)

        # REVIEW:
        self.vrel_max = self.Va
        self.vrel_min = self.Va
        self.vrel_alpha = 0.2
        self.vtg = 0.0
        self.target_angle = 0.0
        self.vrel_prev = 0.0
        self.zero_vel_thresh = 0.01
        self.angle_alpha = 0.5
        self.dvrel = 0.0
        self.dvrel_tau = 1.0

        self.dphi = 0.0
        self.phi_prev = 0.0
        self.dphi_tau = 0.8

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
            self.plotter.update(s, self.R_desired, self.az_err,
                                self.phi_c, self.target_pos, self.target_vel, self.target_vel_est, t)
        # Update state variable
        self.state = states[-1]

    def update_control(self):
        x, y, psi, az, R = self.state



        # # Vg_sq = ((self.Va*math.cos(psi)-self.target_vel_est[0])**2 + (self.Va*math.sin(psi)-self.target_vel_est[1])**2)
        # # phi_ff = math.atan(Vg_sq/(self.R_desired*self.g))
        # # Vg = math.sqrt(Vg_sq)
        # # # Compute difference between chi and psi (chi - psi)
        # # sign = self.lam*np.sign(self.target_vel_est[1]*math.cos(psi) - self.target_vel_est[0]*math.sin(psi))
        # # az_diff = float(sign)*math.acos((self.Va - self.target_vel_est[0]*math.cos(psi) - self.target_vel_est[1]*math.sin(psi))/Vg)
        #
        # Vg_sq = ((self.Va*math.cos(psi)-self.target_vel[0])**2 + (self.Va*math.sin(psi)-self.target_vel[1])**2)
        # Vg = math.sqrt(Vg_sq)
        # # Compute difference between chi and psi (chi - psi)
        # sign = self.lam*np.sign(self.target_vel[1]*math.cos(psi) - self.target_vel[0]*math.sin(psi))
        # az_diff = float(sign)*math.acos((self.Va - self.target_vel[0]*math.cos(psi) - self.target_vel[1]*math.sin(psi))/Vg)
        #
        # az = angle_wrap(az)
        # gamma1 = az + az_diff

        R_psi = np.array([[math.cos(psi), math.sin(psi)],
                         [-math.sin(psi), math.cos(psi)]])
        rel_vel = np.array([self.Va, 0]) - R_psi.dot(self.target_vel) # Use truth for testing
        phi_ff = math.atan(rel_vel.dot(rel_vel)/(self.R_desired*self.g))
        cross = rel_vel[0]*math.sin(az) - rel_vel[1]*math.cos(az)
        dot = rel_vel[0]*math.cos(az) + rel_vel[1]*math.sin(az)
        gamma = np.arctan2(cross, dot)
        # # az_err = abs(az) - math.pi/2.0 - az_diff
        # az_err = gamma - self.lam*math.pi/2.0
        # self.az_err = az_err # hold on to this for plotting
        # phi_az = self.az_PID.compute_control_error(az_err, self.dt)
        #
        # # print("sign = {0}".format(sign))
        # # print("az_diff = {0}".format(az_diff))
        # # print("az_err = {0}".format(az_err))
        # # print("diff = {0}".format(az_diff - az_err))
        # # print("phi_az = {0}".format(phi_az))
        # # print("phi_ff = {0}".format(phi_ff))
        #
        # R_ratio = R/self.R_desired
        # if R_ratio < 1:
        #     p = 8.0
        #     R_err = 1.0/p*R_ratio**p - 1.0/p
        #     # R_err = math.exp(R_ratio - 1) - 1
        # else:
        #     R_err = math.log(R/self.R_desired)
        # # R_err = math.atan(R - self.R_desired)
        # # # Adapt gains based on distance
        # # R_gains = self.get_R_gains(R_err)
        # # self.R_PID.set_gains(*R_gains)
        #
        # # FIXME: Uncomment this ??
        # # Compute control
        # Raz_gains = self.R_PID.compute_control_error(R_err, self.dt, vector_output=True)
        # self.Raz_PID.set_gains(*Raz_gains)
        # Raz_err = gamma
        # if abs(Raz_err) < self.Raz_err_thresh:
        #     Raz_err = 0
        # phi_R = self.Raz_PID.compute_control_error(Raz_err, self.dt)
        # # print("phi_R = {0}".format(phi_R))
        #
        # # phi_ff = math.atan(self.Va**2/(self.g*self.R_desired))
        #
        # phi_c = self.lam*(phi_ff + phi_az) + phi_R

        # REVIEW: Testing combined control appproach
        Vg_max = self.Va + np.linalg.norm(self.target_vel)
        gamma_rate = self.g/Vg_max*math.tan(self.phi_c_max)*0.2
        gamma_d = self.lam*(math.atan(-gamma_rate*(R-self.R_desired))+math.pi/2.0)
        gamma_err = angle_wrap(gamma-gamma_d)

        phi_gamma = self.gamma_PID.compute_control_error(gamma_err, self.dt)
        phi_c = self.lam*phi_ff + phi_gamma


        self.phi_c = sat(phi_c, -self.phi_c_max, self.phi_c_max)

        self.R_avg = 0.99*(R - self.R_desired) + 0.01*self.R_avg
        if abs(self.R_avg) < 10:
            # self.estimate_target_velocity(phi_c, az_diff, psi)
            self.estimate_target_velocity(phi_c, 0.0, psi)

    def estimate_target_velocity(self, phi, az_diff, psi):
        vrel_sq = abs(self.g*self.R_desired*math.tan(phi))
        vrel = np.sqrt(vrel_sq)
        vrel_diff = vrel - self.vrel_prev
        self.dvrel = self.dvrel_tau*(vrel_diff/self.dt) + (1.-self.dvrel_tau)*self.dvrel
        self.vrel_prev = vrel

        delta_vel = vrel - self.Va

        # if abs(self.dvrel) < self.zero_vel_thresh:
        #     if delta_vel > 0:
        #         self.vrel_max = self.vrel_alpha*vrel + (1.-self.vrel_alpha)*self.vrel_max
        #         print("vrel_max = ", self.vrel_max)
        #     else:
        #         self.vrel_min = self.vrel_alpha*vrel + (1.-self.vrel_alpha)*self.vrel_min
        #         print("vrel_min = ", self.vrel_min)
        self.vrel_min = self.Va - np.linalg.norm(self.target_vel)
        self.vrel_max = self.Va + np.linalg.norm(self.target_vel)
        minmax_diff = self.vrel_max - self.vrel_min
        self.vtg = minmax_diff/2.0

        if self.vtg == 0:
            return

        # angle_meas = -np.sign(self.dvrel)*np.arccos(np.clip((self.Va**2 + self.vtg**2 - vrel_sq)/(2.*self.Va*self.vtg) ,-1, 1))
        # FIXME: Testing 2nd angle measurement model
        self.dphi = self.dphi_tau*(phi - self.phi_prev)/self.dt + (1.-self.dphi_tau)*self.dphi
        self.phi_prev = phi

        phi_ff = np.arctan(self.Va**2/(self.g*self.R_desired))
        angle_meas = self.dphi/((phi-phi_ff)*(self.g/self.Va*np.tan(phi)))
        # angle_meas = np.arccos(np.clip((self.Va**2 + self.vtg**2 - vrel_sq)/(2.*self.Va*self.vtg) ,-1, 1))
        delta_angle = angle_meas - self.target_angle
        if abs(delta_angle) > np.pi:
            angle_meas -= np.sign(delta_angle)*np.pi*2
            delta_angle -= np.sign(delta_angle)*np.pi*2
        # FIXME:
        alpha = np.clip(abs(self.dvrel), 0, self.angle_alpha)
        self.target_angle = angle_wrap(alpha*angle_meas + (1.-alpha)*self.target_angle)
        # self.target_angle = angle_meas

        # Vg_hat = math.sqrt(abs(self.g*self.R_desired*math.tan(phi)))
        # # FIXME: for testing vs. truth
        # Vg = math.sqrt((self.Va*math.cos(psi)-self.target_vel[0])**2 + (self.Va*math.sin(psi)-self.target_vel[1])**2)
        # # print("Vg err = {0}".format(Vg_hat - Vg))
        # chi_hat = angle_wrap(psi - az_diff)
        # # FIXME: for testing
        # sign = -self.lam*np.sign(self.target_vel[1]*math.cos(psi) - self.target_vel[0]*math.sin(psi))
        # chi = psi + sign*math.acos((self.Va - self.target_vel[0]*math.cos(psi) - self.target_vel[1]*math.sin(psi))/Vg)
        # # print("chi err = {0}".format(chi_hat - chi))
        #
        # Vt_meas = np.array([self.Va*math.cos(psi) - Vg_hat*math.cos(chi_hat), self.Va*math.sin(psi) - Vg_hat*math.sin(chi_hat)])
        #
        # # print("Vt meas = {0}".format(Vt_meas))
        # if np.linalg.norm(Vt_meas) < self.Va:
        #     self.target_vel_est = self.vel_filter.run(np.hstack((Vt_meas)))[0:2]
        #
        # # self.target_vel_est = (1.0 - self.vel_est_alpha)*Vt_meas + self.vel_est_alpha*self.target_vel_est
        # # print("Target vel estimate = {0}".format(self.target_vel_est))
        #
        # # FIXME: Use truth for testing
        # # self.target_vel_est = np.array([0.001, 0.])
        # # self.target_vel_est = self.target_vel
        target_psi = psi + self.target_angle
        # target_psi = self.target_angle
        self.target_vel_est = self.vtg*np.array([np.cos(target_psi), np.sin(target_psi)])

        # # TEST2
        # self.dphi = self.dphi_tau*(phi - self.phi_prev)/self.dt + (1.-self.dphi_tau)*self.dphi
        # self.phi_prev = phi
        #
        # phi_ff = np.arctan(self.Va**2/(self.g*self.R_desired))
        # delta_phi = phi - phi_ff
        #
        # vel_est = np.zeros((2,1))
        # vel_est[0] = -delta_phi*self.g*self.R_desired/(2*self.Va)
        # vel_est[1] = (self.dphi*self.R_desired)/(2*np.tan(phi))
        #
        # R_psi = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
        # self.target_vel_est = R_psi.dot(vel_est)


    def get_R_gains(self, R_err):
        kp_R_switch = 10.0
        kp_R_blend_rate = 0.3
        kp_R_near = lambda e: 0.05*abs(e)
        kp_R_far = lambda e: 0.005
        kp_R = self.blend_func(R_err, kp_R_near, kp_R_far, kp_R_switch, kp_R_blend_rate)

        kd_R_switch = 0.0
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

class VelocityFilter:
    def __init__(self, dt):
        self.filter = KalmanFilter()

        self.filter.F = np.kron(np.array([[1., dt],
                                         [0., 1.]]), np.eye(2))
        self.filter.H = np.kron(np.array([1., 0.]), np.eye(2))

        self.filter.P = np.diag([1., 1., 1e-9, 1e-9])
        self.filter.Q = np.kron(np.diag([1e-9]*2), np.eye(2))
        self.filter.R = np.diag([5.0]*2)

        self.filter.xhat = np.array([0., 0., 0., 0.])

    def predict(self):
        return self.filter.predict()

    def update(self, meas):
        return self.filter.update(meas)

    def run(self, meas):
        self.update(meas)
        return self.predict()

def angle_wrap(x):
    xwrap = np.array(np.mod(x, 2*np.pi))
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])
    if np.size(xwrap) == 1:
        return float(xwrap)
    else:
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
        elif key == ord('z'):
            vel = np.copy(analysis.target_vel)
            analysis.target_vel = np.array([vel[1], -vel[0]])
        elif key == 27:
            print("Quit")
            sys.exit()
