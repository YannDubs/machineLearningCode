function [model] = leastSquaresGradient(X,y)

[n,d] = size(X);

% Initial guess
w = zeros(d,1);

% This is how you compute the function and gradient:
[f,g] = funObj(w,X,y);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w,@funObj,X,y);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
g2
end

% Solve least squares problem
w = findMin(@funObj,w,100,1,X,y);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end

function [f,g] = funObj(w,X,y)
    f = (1/2)*sum((X*w-y).^2);
    g = X'*(X*w - y);
end