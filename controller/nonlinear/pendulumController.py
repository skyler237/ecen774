import pendulumParam as P
from PDControl import PDControl

class pendulumController:
    '''
        This class inherits other controllers in order to organize multiple controllers.
    '''

    def __init__(self):
        # Instantiates the SS_ctrl object
        self.zCtrl = PDControl(P.kp_z, P.kd_z, P.theta_max, P.beta, P.Ts)
        self.thetaCtrl = PDControl(P.kp_th, P.kd_th, P.F_max, P.beta, P.Ts)

    def u(self, y_r, y):
        # y_r is the referenced input
        # y is the current state
        z_r = y_r[0]
        z = y[0]
        theta = y[1]
        # the reference angle for theta comes from the outer loop PD control
        theta_r = self.zCtrl.PD(z_r, z, flag=False)
        # the force applied to the cart comes from the inner loop PD control
        F = self.thetaCtrl.PD(theta_r, theta, flag=False)
        return [F]
