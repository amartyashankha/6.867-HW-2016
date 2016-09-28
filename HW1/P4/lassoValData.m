function [X,Y] = lassoValData()

data = importdata('lasso_validate.txt');

X = data(1,:);
Y = data(2,:);
