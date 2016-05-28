clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

load '../datasets/santafe';

% plot dataset
figure('Color', [1 1 1]);
plot(1:1000, Z, 'b', 1001:1200, Ztest, 'r');
hold on;
plot(Ztest);

% normalize
[Z, settingsMAPMINMAX]= mapminmax(Z')
[Z, settingsMAPSTD] = mapstd(Z)
Z=Z';

mse_error = zeros(20,1); i = 1;
for order=5:5:100
    X=windowize(Z,1:(order+1));
    Y=X(:,end); X=X(:,1:order);

    [gam,sig2]=tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
    %[alpha,b]=trainlssvm({X,Y,'f',gam,sig2});

    horizon = length(Ztest)-order;
    Zpt = predict({X, Y, 'f', gam, sig2},Ztest(1:order), horizon);
    mse_error(i) = sqrt(mse(Zpt-Ztest(order+1:end)));
    i = i + 1;
end

figure('Color', [1 1 1]);
subplot(1,2,1);
plot(5:5:100, mse_error);

% building best model based on previous estimation of order
order=60;
X=windowize(Z,1:(order+1));
Y=X(:,end); X=X(:,1:order);
[gam,sig2]=tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
%[alpha,b]=trainlssvm({X,Y,'f',gam,sig2});
%horizon = length(Ztest)-order;
Zpt = predict({X, Y, 'f', gam, sig2},Ztest(1:order), horizon);
Zpt = mapstd('reverse', Zpt, settingsMAPSTD);
Zpt = mapminmax.reverse(Zpt, settingsMAPMINMAX);
subplot(1,2,2);
plot([Ztest(order+1:end) Zpt]);

export_fig('santafe_prediction.pdf');

% [Z, settingsMAPMINMAX]= mapminmax(Z')
% [Z, settingsMAPSTD] = mapstd(Z)
% Z=Z';
% X = Z(:, 1);
% Xt = Ztest(:, 1);
% lag = 59; 
% Xu = windowize(Z,1:lag+1);
% Xtra = Xu(1:end-lag,1:lag);
% %training set
% Ytra = Xu(1:end-lag,end);
% Xs=Z(end-lag+1:end,1);
% %starting point for iterative prediction
% [gam, sig2]=tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
% [alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'});
% %predict
% prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,200);
% prediction = mapstd('reverse', prediction, settingsMAPSTD);
% prediction1 = mapminmax.reverse(prediction, settingsMAPMINMAX);
% figure('Color', [1 1 1 ]);
% subplot(1,2,2);
% plot([prediction1 Xt(1:200)]);
% legend('prediction', 'target');
% mse = mse(prediction1-Xt(1:200));
% mse_lssvm = sum((prediction1 - Xt(1:200)).^2) / numel(Xt(1:200));
