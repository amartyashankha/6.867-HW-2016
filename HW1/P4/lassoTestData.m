function [X,Y] = lassoTestData()

data = importdata('lasso_test.txt');

X = data(1,:);
Y = data(2,:);
