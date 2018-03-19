%% Problem 1.3 - Simulate Lorentz System
y0_1 = [0 2 0];
y0_2 = [0 -2 0];
y0_3 = [0 2.01 0];

tspan = [0 40];

[t,y1] = ode45(@lorentz,tspan,y0_1);
[t,y2] = ode45(@lorentz,tspan,y0_2);
[t,y3] = ode45(@lorentz,tspan,y0_3);


clf;
subplot(3,1,1)
plot3(y1(:,1), y1(:,2), y1(:,3))
subplot(3,1,2)
plot3(y2(:,1), y2(:,2), y2(:,3))
subplot(3,1,3)
plot3(y3(:,1), y3(:,2), y3(:,3))

% ODE function
function [yout] = lorentz(t, yin)
  sigma = 10;
  b = 8/3;
  r = 28;

  x = yin(1);
  y = yin(2);
  z = yin(3);
  xdot = sigma*(y - x);
  ydot = r*x - y - x*z;
  zdot = x*y - b*z;

  yout = [xdot; ydot; zdot];

end


%% Observations
% While the general behavior of the system doesn't change with the initial 
% conditions, the direction of the orbits and the density about each center
% can change with small changes in the initial conditions


