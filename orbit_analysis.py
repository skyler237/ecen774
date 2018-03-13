import math
import numpy as np
from Plotter import Plotter

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
    def __init__(self, plotting_freq):
        self.plotter = Plotter(plotting_freq)
        self.plotter.set_plots_per_row(2)

        # Define plot names
        plots = self._define_plots()
        legends = self._define_legends()

        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p, include_legend=(p[0] in legends))

        # Define state vectors for simpler input
        self._define_input_vectors()

    def _define_plots(self):
        plots = ['x',     'R',
                 'psi',   'az',
                 ]
        return plots

    def _define_legends(self):
        legends = []
        return legends

    def _define_input_vectors(self):
        self.plotter.define_input_vector("position", ['x', 'y', 'z'])
        self.plotter.define_input_vector("velocity", ['xdot', 'ydot', 'zdot'])
        self.plotter.define_input_vector("orientation", ['phi', 'theta', 'psi'])
        self.plotter.define_input_vector("imu", ['ax', 'ay', 'az', 'p', 'q', 'r'])
        pass

    def update_sim_data(self, uav_sim):
        self.t = uav_sim.get_sim_time()
        self.plotter.add_vector_measurement("position",     uav_sim.get_position(), self.t)
        self.plotter.add_vector_measurement("velocity",     uav_sim.get_body_velocity(), self.t)
        self.plotter.add_vector_measurement("orientation",  uav_sim.get_euler(), self.t)
        self.plotter.add_vector_measurement("imu",          uav_sim.get_imu(), self.t)


Va = 20.0
h = 100.0
phi = deg2rad(10)

dt = 0.01
ode_dt = dt/10
plot_pause = 0.01
x0 = -0
x_min = x0-10
x_max = x0+10
y0 = -70
y_min = y0-10
y_max = y0+10
psi0 = 0
psi_min = psi0-0.1
psi_max = psi0+0.1
az0 = angle_wrap(pi + atan2(y0,x0) - psi0)
az_min = az0-0.1
az_max = az0+0.1
R0 = sqrt(x0^2 + y0^2)
R_min = R0-5
R_max = R0+5

X0 = [x0 y0 psi0 az0 R0]

## Generate and plot trajectories

t0 = 0
tfinal = 10
N = (tfinal - t0)/dt
X = X0

clf

## Setup plots
figure(1)
title('XY')
hold on

subplot(4,1,2)
title('psi')
hold on

subplot(4,1,3)
title('az')
hold on

subplot(4,1,4)
title('R')
hold on

## Generate trajectory
for i=0:N-1
    t1 = t0 + dt*i
    t2 = t1 + dt
    tspan = [t1:ode_dt:t2]
    [t,X] = ode45(@(t,X) orbit_ode(t,X,phi,Va),tspan,X)
    x = X(:,1)
    x_min = min(x_min, min(x))
    x_max = max(x_max, max(x))
    y = X(:,2)
    y_min = min(y_min, min(y))
    y_max = max(y_max, max(y))
    psi = rad2deg(angle_wrap(X(:,3)))
    psi_min = min(psi_min, min(psi))
    psi_max = max(psi_max, max(psi))
    az = rad2deg(angle_wrap(X(:,4)))
    az_min = min(az_min, min(az))
    az_max = max(az_max, max(az))
    R = X(:,5)
    R_min = min(R_min, min(R))
    R_max = max(R_max, max(R))
    X = X(end,:)
    # Update plots
    # XY plot
    subplot(4,1,1)
    plot(y,x,'k-','linewidth',2)
    xlim([x_min x_max])
    ylim([y_min y_max])
    pause(plot_pause)
    # psi plot
    subplot(4,1,2)
    plot(t,psi,'g-','linewidth',2)
    xlim([t0 t2])
    ylim([psi_min psi_max])
    pause(plot_pause)
    # az plot
    subplot(4,1,3)
    plot(t,az,'b-','linewidth',2)
    xlim([t0 t2])
    ylim([az_min az_max])
    pause(plot_pause)
    # R plot
    subplot(4,1,4)
    plot(t,R,'r-','linewidth',2)
    xlim([t0 t2])
    ylim([R_min R_max])
    pause(plot_pause)
end

## *Define ODE*

function [Xdot] = orbit_ode(t, X, phi, Va)
  g = 9.81

  x = X(1)
  y = X(2)
  psi = X(3)
  az = X(4)
  R = X(5)

  xdot = Va*cos(psi)
  ydot = Va*sin(psi)
  psidot = g/Va*tan(phi)
  azdot = Va/R^2 * (x*sin(psi) - y*cos(psi)) - g/Va*tan(phi)
  Rdot = Va/R * (x*cos(psi) + y*sin(psi))

  Xdot = [xdot ydot psidot azdot Rdot]

end
## *Helper functions*

function [out] = angle_wrap(theta)
   angle_mod = mod(theta,2*pi)
   out = zeros(length(angle_mod),1)
   for i=1:length(angle_mod)
       angle = angle_mod(i)
       if angle > pi
           out(i) = angle - 2*pi
       elseif angle < -pi
           out(i) = angle_mod + 2*pi
       else
           out(i) = angle
       end
   end
end
