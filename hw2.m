%% **HW2 - Extra problem 2: Euler Equations**

%% Generate initial conditions
N = 10;
w_rand = randn(3,N);

tspan = [0 40];

[t,y1] = ode45(@euler_ode,tspan,y0_1);
[t,y2] = ode45(@euler_ode,tspan,y0_2);
[t,y3] = ode45(@euler_ode,tspan,y0_3);


clf;
subplot(3,1,1)
plot3(y1(:,1), y1(:,2), y1(:,3))
subplot(3,1,2)
plot3(y2(:,1), y2(:,2), y2(:,3))
subplot(3,1,3)
plot3(y3(:,1), y3(:,2), y3(:,3))

% Define ODE
function [yout] = euler_ode(t, yin)
  I1 = 3;
  I2 = 2;
  I3 = 1;
  
  a = (I2 - I3)/I1;
  b = (I3 - I1)/I2;
  c = (I1 - I2)/I3;

  w1 = yin(1);
  w2 = yin(2);
  w3 = yin(3);
  w1dot = a*w2*w3;
  w2dot = b*w1*w3;
  w3dot = c*w1*w2;

  yout = [w1dot; w2dot; w3dot];

end