import sys
sys.path.insert(0, '/home/skyler/school/ecen774/state_plotter/src/state_plotter')

from Plotter import Plotter
from IPython.core.debugger import set_trace

import numpy as np

class PlotWrapper:
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


    def _define_plots(self):
        plots = ['theta theta_r', 'z z_r',
                 'thetadot thetadot_r', 'zdot zdot_r'
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("state", ['theta', 'z', 'thetadot', 'zdot'])
        self.plotter.define_input_vector("reference", ['theta_r', 'z_r', 'thetadot_r', 'zdot_r'])

    def update(self, state, reference, t):
        self.plotter.add_vector_measurement('state', state, t)
        self.plotter.add_vector_measurement('reference', reference, t)
        self.plotter.update_plots()


plotter = PlotWrapper(5)

a0 = -3.0
a1 = -3.0
a2 = -0.0
a3 = 1.0
a4 = -3.0
a5 = -5.0
Aref = np.array([[a0, 0., 1., 0.],
                 [0., a1, 0., 1.],
                 [0., 0., a2, a3],
                 [0., 0., a4, a5]])
Bref = -Aref

x_r = np.array([0.0, 2.0, 0.0, 0.0]).reshape(4,1)
x0  = np.array([0.1, 0.1, 0.1, 0.1]).reshape(4,1)

x = x0
t0 = 0
tfinal = 5
dt = 0.01
tcnt = int((tfinal-t0)/dt + 1)
for t in np.linspace(t0, tfinal, tcnt):
    # set_trace()
    xdot = Aref.dot(x) + Bref.dot(x_r)
    x += xdot*dt
    plotter.update(x.ravel(),x_r.ravel(),t)

while True:
    pass
