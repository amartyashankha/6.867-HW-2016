function [X,Y] = regressBData()

data = importdata('regressB_train.txt');

X = data(1,:);
Y = data(2,:);