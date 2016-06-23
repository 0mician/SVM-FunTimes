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
