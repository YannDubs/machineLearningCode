function [model] = leastSquaresBias(X,y)

[n,~] = size(X);
% simply adds a biais: column of ones
Z=[ones(n,1) X];
% Solve least squares problem
w = (Z'*Z)\Z'*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
[n,~] = size(Xhat);
Zhat=[ones(n,1) Xhat];
yhat = Zhat*w;
end