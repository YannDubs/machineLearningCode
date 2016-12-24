function [model] = decisionForest(X,y,depth,nBootstraps)

[n d]=size(X);

% Fit model to each boostrap sample of data
for m = 1:nBootstraps
    index=randi(n,1,n);
    Xtrain=X(index,:);
    ytrain=y(index);
    model.subModel{m} = randomTree(Xtrain,ytrain,depth);
end

model.predict = @predict;

end

function [y] = predict(model,X)

% Predict using each model
for m = 1:length(model.subModel)
    y(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end

% Take the most common label
y = mode(y,2);
end