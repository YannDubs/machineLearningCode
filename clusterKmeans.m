function [model Wi dist] = clusterKmeans(X,k,doPlot)
% [model] = clusterKmeans(X,k,doPlot)
%
% K-means clustering

[n,d] = size(X);
y = ones(n,1);
Wi=zeros(k,d,1);
dist=zeros(n,k,1);
j=1;
% Choose random points to initialize means
W = zeros(k,d);
for k = 1:k
    i = ceil(rand*n);
    W(k,:) = X(i,:);
end

X2 = X.^2*ones(d,k);
while 1
    y_old = y;
        Wi(:,:,j)=W;
           % Compute (squared) Euclidean distance between each data point and each mean
    distances = X2 + ones(n,d)*(W').^2 - 2*X*W';
    dist(:,:,j)=distances;
        j=j+1;
        
    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end
    
 
    
    % Assign each data point to closest mean
    [~,y] = min(distances,[],2);
    
    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end
    
    % Compute mean of each cluster
    for k = 1:k
        W(k,:) = mean(X(y==k,:),1);
    end
    
    changes = sum(y ~= y_old);
    %fprintf('Running K-means, difference = %f\n',changes);
    
    % Stop if no point changed cluster
    if changes == 0
        break;
    end

end

model.W = W;
model.y = y;
model.predict = @predict;
model.error = @error;
end

function [y] = predict(model,X)
[t,d] = size(X);
W = model.W;
k = size(W,1);

% Compute Euclidean distance between each data point and each mean
X2 = X.^2*ones(d,k);
distances = sqrt(X2 + ones(t,d)*(W').^2 - 2*X*W');

% Assign each data point to closest mean
[~,y] = min(distances,[],2);
end

function [err] = error(model,X)
% Array containing the means of each data points
Wy=model.W(model.y,:);

% Compute Euclidean distance between each data point and its mean
Wy=(Wy-X).^2;
err=sum(sum(Wy));
end