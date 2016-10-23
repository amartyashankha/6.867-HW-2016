function pegasos_test

% load data from csv files
data = importdata(strcat('data/data3_train.csv'));
X = data(:,1:2);
Y = data(:,3);

disp('======Linear SVM======');
% Carry out training.

%%% TODO: Implement train_linearSVM %%%
epochs = 1000;
lambda = .02;
global w
train_linearSVM(X, Y, lambda, epochs);


% Define the predict_linearSVM(x) function, which uses global trained parameters, w
%%% TODO: define predict_linearSVM(x) %%%

hold on;
% plot training results
plotDecisionBoundary(X, Y, @predict_linearSVM, [-1,0,1], 'Linear SVM');




