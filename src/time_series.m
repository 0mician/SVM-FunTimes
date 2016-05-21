clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

load '../datasets/santafe';

order = 1;
X = windowize(Z, 1:(order+1));
Y = X(:, end);
X = X(:, 1:order);
plot(Z, Y);

gam = 10; sig2 = 10;
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

Xnew = Z((end-order+1):end)';
Z(end+1) = simlssvm({X, Y, 'f', gam, sig2},{alpha, b}, Xnew);

horizon = length(Ztest)-order;
Zpt = predict({X, Y, 'f', gam, sig2},Ztest(1:order), horizon);
plot([Ztest(order+1:end) Zpt]);

test_size = 100;
Ztrain = Z(1:length(Z) - test_size);
Ztest = Z(length(Z) - test_size+1:end);
