# Inverted Pendulum Parameter File
import numpy as np
# import control as cnt

# Physical parameters of the inverted pendulum known to the controller
m1 = 0.25     # Mass of the pendulum, kg
m2 = 1.0      # Mass of the cart, kg
ell = 0.5    # Length of the rod, m
g = 9.8       # Gravity, m/s**2
b = 0.05      # Damping coefficient, Ns

# parameters for animation
w = 0.5       # Width of the cart, m
h = 0.15      # Height of the cart, m
gap = 0.005   # Gap between the cart and x-axis
radius = 0.06 # Radius of circular part of pendulum

# Initial Conditions
z0 = 0.0                # ,m
theta0 = 0.0*np.pi/180  # ,rads
zdot0 = 0.0             # ,m/s
thetadot0 = 0.0         # ,rads/s

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 50.0  # End time of simulation
Ts = 0.01  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate

# saturation limits
F_max = 5.0                # Max Force, N

#########################################################################
########################### Control params ##############################
#########################################################################

# dirty derivative parameters
sigma = 0.05  # cutoff freq for dirty derivative
beta = (2 * sigma - Ts) / (2 * sigma + Ts)  # dirty derivative gain

####################################################
#       PD Control: Time Design Strategy
####################################################
# tuning parameters
tr_th = 0.35          # Rise time for inner loop (theta)
zeta_th = 0.707       # Damping Coefficient for inner loop (theta)
M = 20.0              # Time scale separation between inner and outer loop
zeta_z = 0.707        # Damping Coefficient fop outer loop (z)

# saturation limits
F_max = 5             		  # Max Force, N
error_max = 1        		  # Max step size,m
theta_max = 30.0*np.pi/180.0  # Max theta, rads

#---------------------------------------------------
#                    Inner Loop
#---------------------------------------------------
# parameters of the open loop transfer function
b0_th = -2.0/(m2*(ell/2.0))
a1_th = 0.0
a0_th = -2.0*(m1+m2)*g/(m2*(ell/2.0))

# coefficients for desired inner loop
# Delta_des(s) = s^2 + alpha1*s + alpha0 = s^2 + 2*zeta*wn*s + wn^2
wn_th = 2.2/tr_th     # Natural frequency
alpha1_th = 2.0*zeta_th*wn_th
alpha0_th = wn_th**2

# compute gains
# Delta(s) = s^2 + (a1 + b0*kd)*s + (a0 + b0*kp)
kp_th = (alpha0_th-a0_th)/b0_th
kd_th = (alpha1_th-a1_th)/b0_th
DC_gain = kp_th/((m1+m2)*g+kp_th)

#---------------------------------------------------
#                    Outer Loop
#---------------------------------------------------
# parameters of the open loop transfer function
b0_z = (m1*g/m2)
a1_z = b/m2
a0_z = 0

# coefficients for desired outer loop
# Delta_des(s) = s^2 + alpha1*s + alpha0 = s^2 + 2*zeta*wn*s + wn^2
tr_z = M*tr_th  # desired rise time, s
wn_z = 2.2/tr_z  # desired natural frequency
alpha1_z = 2.0*zeta_z*wn_z
alpha0_z = wn_z**2

# compute gains
# Delta(s) = s^2 + (a1 + b0*kd*DC_gain)*s + (a0 + b0*kp*DC_gain)
kp_z = (alpha0_z-a0_z)/(DC_gain*b0_z)
kd_z = (alpha1_z-a1_z)/(DC_gain*b0_z)

print('DC_gain', DC_gain)
print('kp_th: ', kp_th)
print('kd_th: ', kd_th)
print('kp_z: ', kp_z)
print('kd_z: ', kd_z)
