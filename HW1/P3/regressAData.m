function [X,Y] = regressAData()

data = importdata('regressA_train.txt');

X = data(1,:);
Y = data(2,:);