function [X,Y] = validateData()

data = importdata('regress_validate.txt');

X = data(1,:);
Y = data(2,:);