%% Problem 1.3 - Simulate Lorentz System
x_max = 25;
x_step = 15;
y_max = 25;
y_step = 15;
z_max = 25;
z_step = 15;
[y0_x, y0_y, y0_z] = meshgrid(-x_max:x_step:x_max, -y_max:y_step:y_max, -z_max:z_step:z_max);

xsize = size(y0_x);
ysize = size(y0_y);
zsize = size(y0_z);
y0 = [reshape(y0_x, 1,xsize(1)*xsize(2)*xsize(3)); 
      reshape(y0_y, 1,ysize(1)*ysize(2)*ysize(3));
      reshape(y0_z, 1,zsize(1)*zsize(2)*zsize(3))];

tspan = [0 30];

figure(1)
hold on
for i=1:length(y0)
   y_init = y0(:,i);
   [t, y] = ode45(@lorentz,tspan,y_init);
   plot3(y(:,1), y(:,2), y(:,3))
end

[t,y1] = ode45(@lorentz,tspan,y0_1);
[t,y2] = ode45(@lorentz,tspan,y0_2);
[t,y3] = ode45(@lorentz,tspan,y0_3);


clf;
% subplot(3,1,1)
% plot3(y1(:,1), y1(:,2), y1(:,3))
% subplot(3,1,2)
plot3(y2(:,1), y2(:,2), y2(:,3))
% subplot(3,1,3)
% plot3(y3(:,1), y3(:,2), y3(:,3))
% 
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
