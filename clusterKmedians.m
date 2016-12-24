function [model] = clusterKmedians(X,k,doPlot)
% [model] = clusterKmedians(X,k,doPlot)
%
% K-medians clustering
%

[n,d] = size(X);
y = ones(n,1);

% Choose random points to initialize medians
W = zeros(k,d);
for k = 1:k
    i = ceil(rand*n);
    W(k,:) = X(i,:);
end

while 1
    y_old = y;
    
    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end
    
    distances=zeros(n,k);
    for j=1:k
        % Compute L1 distance between each data point and each median
        distances(:,j) = sum(abs(X - ones(n,1)*W(j,:)),2);
    end
    
    % Assign each data point to closest median
    [~,y] = min(distances,[],2);
    
    % Draw visualization
    if doPlot && d == 2
       clustering2Dplot(X,y,W)
    end
    
    % Compute median of each cluster
    for k = 1:k
        W(k,:) = median(X(y==k,:),1);
    end
    
    changes = sum(y ~= y_old);
    %fprintf('Running K-medans, difference = %f\n',changes);
    
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

% Compute L1 distance between each data point and each median
distances=zeros(t,k);
for j=1:k
    % Compute L1 distance between each data point and each median
    distances(:,j) = sum(abs(X - ones(t,1)*W(j,:)),2);
end
    
    % Assign each data point to closest median
    [~,y] = min(distances,[],2);

end

function [err] = error(model,X)
% Array containing the meadians of each data points
Wy=model.W(model.y,:);

% Compute L1 distance between each data point and its mean
Wy=abs(Wy-X);
err=sum(sum(Wy));
end