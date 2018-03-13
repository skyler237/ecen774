%% Problem 1.3 - Simulate Lorentz System
x_max = 3;
x_step = .5;
y_max = 3;
y_step = .5;
z_max = 25;
z_step = 15;
% [y0_x, y0_y, y0_z] = meshgrid(-x_max:x_step:x_max, -y_max:y_step:y_max, -z_max:z_step:z_max);
[y0_x, y0_y] = meshgrid(-x_max:x_step:x_max, -y_max:y_step:y_max);

xsize = size(y0_x);
ysize = size(y0_y);
% zsize = size(y0_z);
y0 = [reshape(y0_x, 1,xsize(1)*xsize(2)); %*xsize(3));
      reshape(y0_y, 1,ysize(1)*ysize(2))]; %*ysize(3))];
%       reshape(y0_z, 1,zsize(1)*zsize(2)*zsize(3))];

tspan = [0 10];

clf;
figure(1)
hold on
for i=1:length(y0)
   y_init = y0(:,i);
   [t, y] = ode45(@my_ode,tspan,y_init);
%    plot3(y(:,1), y(:,2)), y(:,3));
    plot(y(:,1), y(:,2))
end
axis([-5 5 -5 5])

% ODE function
function [yout] = my_ode(t, yin)
  x = yin(1);
  y = yin(2);
  xdot = y;
  ydot = -x + 1/16*x^5 - y;

  yout = [xdot; ydot];

end
