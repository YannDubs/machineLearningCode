function [model] = softMaxClassifier(X,y)
    [n,d] = size(X);
    k = max(y);
    w0 = zeros(d*k,1);
    w = findMin(@softMax,w0,400,0,X,y); 

    model.W = reshape(w,[d k]);
    model.predict = @predict;
    
end

function [f,g] = softMax(w,X,y)
    [n,d] = size(X);
    k = max (y);
    W = reshape(w,[d k]);
    denom = sum(exp(X*W),2);
    f = sum(diag(-X'*W(:,y([1:n]))'))+ sum(log(denom));
    g=zeros(d,k);
    for c = [1:k]
        g(:,c) = g(:,c) - sum(X(y==c,:)',2) + (X'*(exp(X*W(:,c))./denom));
    end
    g=g(:);
end

function [yhat] = predict(model,X)
W = model.W;
    [~,yhat] = max(X*W,[],2);
end