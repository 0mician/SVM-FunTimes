clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

% dataset (training and validation)
X = (-10:0.1:10)'; Y = cos(X) + cos(2.*X) + 0.1.*randn(length(X), 1);
XX = (-10:0.01:10); YY= cos(XX) + cos(2.*XX);

Xtrain = X(1:2:length(X)); Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X)); Ytest = Y(2:2:length(Y));

% different models parameters exploration

gam_list = [1000, 100, 10, 1]; 
sig2_list = [0.1, 0.1, 0.1, 0.1];
figure('Color',[1 1 1]);

for i = 1:4
    gam = gam_list(i); sig2 = sig2_list(i);
    [alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'});
    YtestEst = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b}, Xtest);
    %figure; plotlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b});
    subplot(2,2,i);
    plot(XX, YY, 'k-'); hold on; plot(Xtest, Ytest, '.'); hold on; plot(Xtest, YtestEst, 'r+');
    legend('true', 'Ytest', 'YtestEst');
end

export_fig('sumcos1.pdf');

gam_list = [1000, 1000, 1000, 1000]; 
sig2_list = [0.01, 0.1, 1, 10];
figure('Color',[1 1 1]);

for i = 1:4
    gam = gam_list(i); sig2 = sig2_list(i);
    [alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'});
    YtestEst = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b}, Xtest);
    %figure; plotlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b});
    subplot(2,2,i);
    plot(XX, YY, 'k-'); hold on; plot(Xtest, Ytest, '.'); hold on; plot(Xtest, YtestEst, 'r+');
    legend('true', 'Ytest', 'YtestEst');
end

export_fig('sumcos2.pdf');

% Systematic exploration of parameters' space
gam_list = logspace(-2,6,100); sig2_list = logspace(-2,1,100);
err_matrix = zeros(100,100); i = 1; j = 1;
true_ys = cos(Xtest) + cos(2.*Xtest);

for sig2=sig2_list,
    j = 1;
    for gam=gam_list,
        [alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'});
        YtestEst = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'},{alpha, b}, Xtest);
        err_matrix(i, j) = sum((YtestEst - Ytest).^2);
        j = j + 1;
    end
    i = i + 1;
end

figure('Color',[1,1,1]);
h = surf(gam_list, sig2_list, err_matrix);
set(get(h,'Parent'),'XScale','log');
set(get(h,'Parent'),'YScale','log');

export_fig('sumcos_surf.pdf');

% Hyper parameters tuning
% gam = 100, sig2 = 0.1;
% cost_crossval = crossvalidate({Xtrain, Ytrain, 'f', gam, sig2}, 10);
% cost_loo = leaveoneout({Xtrain, Ytrain, 'f', gam, sig2});

figure('Color',[1,1,1]);

tic;
[gam, sig2, cost] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel','csa'},'gridsearch', 'crossvalidatelssvm',{10,'mse'});
toc;
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2});
subplot(2,2,2);
plotlssvm({Xtrain, Ytrain, 'f', gam, sig2}, {alpha, b});


tic;
[gam, sig2, cost] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel','csa'},'simplex', 'crossvalidatelssvm',{10,'mse'});
toc;
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2});
subplot(2,2,1);
plotlssvm({Xtrain, Ytrain, 'f', gam, sig2}, {alpha, b});

tic;
[gam, sig2, cost] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel','ds'},'gridsearch', 'crossvalidatelssvm',{10,'mse'});
toc;
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2});
subplot(2,2,3);
plotlssvm({Xtrain, Ytrain, 'f', gam, sig2}, {alpha, b});

tic;
[gam, sig2, cost] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel','ds'},'simplex', 'crossvalidatelssvm',{10,'mse'});
toc;
[alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam, sig2});
subplot(2,2,4);
plotlssvm({Xtrain, Ytrain, 'f', gam, sig2}, {alpha, b});

export_fig('hyperparam_tuning.pdf');

% Bayesian framework
sig2 = 0.5; gam = 10;
criterion_L1 = bay_lssvm({X,Y,'f',gam,sig2},1);
criterion_L2 = bay_lssvm({X,Y,'f',gam,sig2},2);
criterion_L3 = bay_lssvm({X,Y,'f',gam,sig2},3);

%
gam=10; sig2=0.05;
[~,alpha,b] = bay_optimize({X,Y,'f',gam,sig2}, 1);
[~,gam] = bay_optimize({X,Y,'f',gam,sig2},2);
[~,sig2] = bay_optimize({X,Y,'f',gam,sig2},3);
sig2e = bay_errorbar({X,Y,'f',gam,sig2},'figure');

export_fig('sumcos_bay1.pdf');

load '../datasets/iris';
figure('Color', [1 1 1]);
hold on;
gam_list = [1 10 100]; sig2_list = [0.01 0.1 1];
i = 1;
for gam=gam_list
    for sig2=sig2_list
        subplot(3,3,i);
        hold on;
        bay_modoutClass({X,Y,'c',gam,sig2},'figure');
        i = i + 1;
    end
end

export_fig('bayes_iris.pdf');

% ARD
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) +0.3.*randn(100,1);
[selected, ranking, costs2] = bay_lssvmARD({X,Y,'class', 100, 0.1});

% robust regression
X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));
Y_nn = Y;

outliers_x = [15 17 19 41 44 46];

out = [15 17 19];
Y(out) = 0.7 + 0.3*rand(size(out));
outliers_y = Y(out);
out = [41 44 46];
Y(out) = 1.5 + 0.2*rand(size(out));
outliers_y = [outliers_y' Y(out)'];

weight_functions = {'whuber', 'whampel', 'wlogistic', 'wmyriad'};
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');

% different noise impact on non robut lssvm
figure('Color', [1 1 1]);
sig2 = 0.1; gam = 100;
subplot(1,2,1);
[alpha, b] = trainlssvm({X, Y_nn, 'f', gam, sig2, 'RBF_kernel'});
plotlssvm({X, Y_nn, 'f', gam, sig2, 'RBF_kernel'},{alpha, b});
subplot(1,2,2);
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'});
plotlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'},{alpha, b});
hold on;
plot(X(outliers_x), outliers_y, 'r*');

export_fig('robust_outliers.pdf');

figure('Color', [1 1 1]);
for i=1:4
    wFun = weight_functions{i};
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model = tunelssvm(model,'simplex','rcrossvalidatelssvm',{10,'mae'},wFun);
    model = robustlssvm(model);
    subplot(2,2,i);
    plotlssvm(model);
end

for i=1:4
    subplot(2,2,i);
    hold on;
    plot(X(outliers_x), outliers_y, 'r*');
end

export_fig('bayes_robust.pdf');