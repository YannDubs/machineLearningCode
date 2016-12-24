function [model] = leastSquaresBasis(X,y,p,j,i)

Z=polyBasis(X,p,j,i);
% Solve least squares problem
w = (Z'*Z)\Z'*y;

model.w = w;
model.predict = @predict;
model.p = p;
model.j = j;
model.i = i;

end

function [yhat] = predict(model,Xhat)
w = model.w;
p = model.p;
j = model.j;
i = model.i;
Zhat=polyBasis(Xhat,p,j,i);
yhat = Zhat*w;
end

function [Z] = polyBasis(X,p,j,i)
[n,~] = size(X);
Z=[ones(n,1) X ];
%for i=p
    Z=[Z cos(1/j*X)  sin(1/p*X) sin(i*X)];
%end
end