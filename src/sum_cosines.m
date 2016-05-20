clc; clear all; close all;
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

% dataset (training and validation)
X = (-10:0.1:10)'; Y = cos(X) + cos(2.*X) + 0.1.*randn(length(X), 1);
XX = (-10:0.01:10); YY= cos(XX) + cos(2.*XX);

Xtrain = X(1:2:length(X)); Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X)); Ytest = Y(2:2:length(Y));

% different models
%gam = [100, ] 
%sig2 = [1, ]
gam = 10; sig2 = 0.01;
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'});
figure; plotlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b});
YtestEst = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b}, Xtest);

% visualization
figure('Color',[1 1 1]);
subplot(2,2,1);

plot(XX, YY, 'k-'); hold on;
plot(Xtest, Ytest, '.'); hold on;
plot(Xtest, YtestEst, 'r+');
legend('true', 'Ytest', 'YtestEst');