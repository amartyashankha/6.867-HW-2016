function pegasos_test

% load data from csv files
data = importdata(strcat('data/data3_train.csv'));
global X
X = data(:,1:2);
Y = data(:,3);
[n, d] = size(X);


disp('======Gaussian Kernel SVM======');
% Carry out training.
global alpha
global gamma
global K
epochs = 1000;
lambda = .02;
gamma = 2^2;

K = zeros(n,n);
%%% TODO: Compute the kernel matrix %%%

%%% TODO: Implement train_gaussianSVM %%%
train_gaussianSVM(X, Y, lambda, epochs);


% Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
%%% TODO:  define predict_gaussianSVM(x) %%%


hold on;

% plot training results
plotDecisionBoundary(X, Y, @predict_gaussianSVM, [-1,0,1], 'Gaussian Kernel SVM');


