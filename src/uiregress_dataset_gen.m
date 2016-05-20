clear all;
addpath('export_fig');

X = linspace(-5,5,20); Y = X.^3 + X.^2 - 1;
X = X'; Y = Y';
save('poly.mat');
clear all;

X = linspace(-5, 5, 20); Y = 2.*X + 1;
X = X'; Y = Y';
save('lin.mat');
clear all;

X = linspace(-pi,pi,20) ; Y = exp(-X.^2).*sin(10.*X);
X = X'; Y = Y';
save('sinc.mat');
clear all;

X1 = linspace(-5, 5, 200); Y1 = 2.*X1 + 1;
X2 = linspace(-5,5,200); Y2 = X1.^3 + X1.^2 - 1;
X3 = linspace(-pi,pi,200) ; Y3 = exp(-X1.^2).*sin(10.*X1);

figure('Color',[1 1 1]);
subplot(1,3,1);
plot(X1, Y1, 'b-');
title('Linear function: Y = 2X +1');
subplot(1,3,2);
plot(X2, Y2, 'b-');
title('Cubic function: Y = X^3 + X^2 - 1');
subplot(1,3,3);
plot(X3, Y3, 'b-');
title('Sinc function: Y = e^{-X^2}sin(10X)');

export_fig('datasets.pdf');