import pendulumParam as P
from PDControl import PDControl
from IPython.core.debugger import set_trace

import numpy as np

class pendulumController:
    '''
        This class inherits other controllers in order to organize multiple controllers.
    '''

    def __init__(self):
        # Instantiates the SS_ctrl object
        self.zCtrl = PDControl(P.kp_z, P.kd_z, P.theta_max, P.beta, P.Ts)
        self.thetaCtrl = PDControl(P.kp_th, P.kd_th, P.F_max, P.beta, P.Ts)


        self.w = 3.0
        self.z = 1.0
        a0 = -3.0
        a1 = -3.0
        a2 = -0.0
        a3 = 1.0
        a4 = -self.w**2
        a5 = -2.0*self.z*self.w
        # self.Aref = np.array([[a0, 0., 1., 0.],
        #                       [0., a1, 0., 1.],
        #                       [0., 0., a2, a3],
        #                       [0., 0., a4, a5]])
        self.Aref = np.array([[a2, a3],
                              [a4, a5]])
        self.Bref = -self.Aref

        self.Kp = np.zeros((2,2))
        self.Kr = np.zeros((2,2))
        self.gamma_x = 1.0
        self.gamma_r = 1.0

    def u(self, y_r, y):
        # r = np.array([0., y_r[0]]).reshape(2,1)
        # y = np.array(y).reshape(2,1)
        # # Update adaptive gains
        # Kpdot = -self.gamma_x*y.T*(y-r)
        # self.Kp += Kpdot*P.Ts
        #
        # Krdot = -self.gamma_r*r.T*(y-r)
        # self.Kr += Krdot*P.Ts
        #
        # u = self.Kp*y + self.Kr*r
        # y_r = list(y_r)
        # y = list(y)

        # Take 2!
        e = y[1] - y_r
        r = y_r

        P12 = 1.0/(2*self.w**2)
        P21 = P12
        P22 = 1.0/(4*self.z*self.w)*(1.0 + 1.0/self.w**2)


        return [u]
