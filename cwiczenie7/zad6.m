close all
clear all
clc

% Zadanie 1
load("LM01Data")
f = @(x) x(1)*sin(x(2)*t+x(3))-y;
x0 = [1.0, 100*pi, 0.0];

Odp1 = lsqnonlin(f, x0)

% Zadanie 2
load("LM04Data")
a = 3;
f = @(x) x(1) *exp(-t * a).*sin(x(2)*t+x(3))-y;
x0 = [1, 5*pi, 1];

Odp2 = lsqnonlin(f, x0)

% Zadanie 3
load("inertialData")
f = @(x) x(1) * (1 - exp(-t/x(2))) - y;
x0 = [1 1];

Odp3 = lsqnonlin(f,x0)

% Zadanie 4
load("twoInertialData.mat")
f = @(x) x(1) * (1 - (1 / (x(2) - x(3))) * ((x(2) * exp(-t/x(2))) - (x(3) * exp(-t/x(3)))) ) - y;
x0 = [1 2 1];

Odp4 = lsqnonlin(f,x0)

% Zadanie 5
load("reductionData.mat")
y = y * 1000;
t = t * 1000;
f = @(x) x(1) * (1 - exp(-x(2) * t) .* (cos(x(3) * t) + (x(2) / x(3)) * sin(x(3) * t))) - y;
x0 = [1 1 1];

Odp5 = lsqnonlin(f,x0) / 1000