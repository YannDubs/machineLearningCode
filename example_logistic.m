
%% Load data, standardize columns, and add bias
load logisticData.mat
[n,d] = size(X);
t = size(Xvalidate,1);

% Standardize columns
[X,mu,sigma] = standardizeCols(X);

% Perform the *same* transformation of the test data
Xvalidate = standardizeCols(Xvalidate,mu,sigma);

% Add bias
X = [ones(n,1) X];

% Also add a bias to the test data
Xvalidate = [ones(t,1) Xvalidate];

%% Fit logistic regression model,
% then report number of non-zeroes and validation error

model = logReg(X,y);
    
numberOfNonZero = nnz(model.w)

yhat = model.predict(model,X);
trainingError = sum(yhat ~= y)/n

yhat = model.predict(model,Xvalidate);
validationError = sum(yhat ~= yvalidate)/t

%% Fit logistic regression model with L2 regularization,
% then report number of non-zeroes and validation error

model = logRegL2(X,y,1);
    
numberOfNonZeroL2 = nnz(model.w)

yhat = model.predict(model,X);
trainingErrorL2 = sum(yhat ~= y)/n

yhat = model.predict(model,Xvalidate);
validationErrorL2 = sum(yhat ~= yvalidate)/t

%% Fit logistic regression model with L1 regularization,
% then report number of non-zeroes and validation error

model = logRegL1(X,y,1);
    
numberOfNonZeroL1 = nnz(model.w)

yhat = model.predict(model,X);
trainingErrorL1 = sum(yhat ~= y)/n

yhat = model.predict(model,Xvalidate);
validationErrorL1 = sum(yhat ~= yvalidate)/t

%% Fit logistic regression model with L0 regularization,
% then report number of non-zeroes and validation error

model = logRegL0(X,y,1);
    
numberOfNonZeroL0 = nnz(model.w)

yhat = model.predict(model,X);
trainingErrorL0 = sum(yhat ~= y)/n

yhat = model.predict(model,Xvalidate);
validationErrorL0 = sum(yhat ~= yvalidate)/t