function [X,Y] = lassoTrainData()

data = importdata('lasso_train.txt');

X = data(1,:);
Y = data(2,:);
