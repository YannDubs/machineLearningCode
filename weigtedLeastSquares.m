function [model] = weigtedLeastSquares(X,y,Z)

% Solves weigted least squares problem
w = (X'*Z*X)\X'*Z*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
yhat = Xhat*w;
end