function [model] = leastSquaresRBF(X,y,sigma)

[n,d] = size(X);

Z = rbfBasis(X,X,sigma);

% Solve least squares problem
lambda = 1e-12;
w = (Z'*Z + lambda*eye(n))\Z'*y;

model.X = X;
model.w = w;
model.sigma = sigma;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
Z = rbfBasis(Xtest,model.X,model.sigma);
yhat = Z*model.w;
end

function [Z] = rbfBasis(X1,X2,sigma)
n1 = size(X1,1);
n2 = size(X2,1);
d = size(X1,2);
den = 1/sqrt(2*pi*sigma^2);
D = X1.^2*ones(d,n2) + ones(n1,d)*(X2').^2 - 2*X1*X2';
Z = den*exp(-D/(2*sigma^2));
end