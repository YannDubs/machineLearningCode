function [model] = leastSquaresBasis(X,y,p)

Z=polyBasis(X,p);
% Solve least squares problem
w = (Z'*Z)\Z'*y;

model.w = w;
model.predict = @predict;
model.p = p;

end

function [yhat] = predict(model,Xhat)
w = model.w;
p = model.p;
Zhat=polyBasis(Xhat,p);
yhat = Zhat*w;
end

function [Z] = polyBasis(X,p)
Z=[];
for i=0:p
    Z=[Z sin(X).^i];
end
end