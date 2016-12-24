function [model] = robustRegression(X,y)

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
end

% Solve least squares problem
w = findMin(@funObj,w,100,0,X,y);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end

function [f,g] = funObj(w,X,y)
    [n,~]=size(y);
    
    sumf=0;
    sumg=0;
    for i=1:n
        sumf=sumf+log(exp(w'*X(i,:)'-y(i)) + exp(-w'*X(i,:)'+y(i)));
        sumg=sumg + ((exp(2*(w'*X(i,:)'-y(i))) - 1)/ (exp(2*(w'*X(i,:)'-y(i))) + 1) * X(i,1));
    end
    f = sumf;
    g = sumg;
end