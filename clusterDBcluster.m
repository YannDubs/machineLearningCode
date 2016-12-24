function [model nExpanded] = clusterDBcluster(X,radius,minPts,doPlot)

[n,d] = size(X);

% Compute distances between all points
D = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*X';

% This will be the cluster of each object.
y = zeros(n,1);

% This variable will keep track of whether we've visited each object.
visited = zeros(n,1);
nExpanded = ones(n,1);

% k will count the number of clusters we've found
k = 0;
for i = 1:n
    if ~visited(i) 
        % We only need to consider examples that have never been visited
        visited(i) = 1;
        neighbors = find(D(:,i) <= radius);
        if length(neighbors) >= minPts 
            % We found a new cluster
            nExpanded(i) = 0;
            k = k + 1;
            [visited,y] = expand(X,i,neighbors,k,radius,minPts,D,visited,y,false);
        end
    end
end

if doPlot && size(X,2) == 2
    % Make plot
    clf;hold on;
    colors = getColorsRGB;
    h = plot(X(y==0,1),X(y==0,2),'.');
    set(h,'Color',[0 0 0]);
    for k = 1:k
        h = plot(X(y==k,1),X(y==k,2),'.');
        set(h,'Color',colors(k,:));
    end
end
        
model.Xtrain = X;
model.y = y;
end

function [visited,y] = expand(X,i,neighbors,k,radius,minPts,D,visited,y,doPlot)
y(i) = k;
ind = 0;
while 1
    ind = ind+1;
    if ind > length(neighbors)
        break;
    end
    n = neighbors(ind);
    y(n) = k;
    
    if ~visited(n)
        visited(n) = 1;
        neighbors2 = find(D(:,n) <= radius);
        if length(neighbors2) >= minPts
            neighbors = [neighbors;setdiff(neighbors2,neighbors)];
        end
    end
    
    if doPlot && size(X,2) == 2
        % Make plot
        clf;hold on;
        colors = getColorsRGB;
        h = plot(X(y==0,1),X(y==0,2),'.');
        set(h,'Color',[0 0 0]);
        for k = 1:k
            h = plot(X(y==k,1),X(y==k,2),'.');
            set(h,'Color',colors(k,:));
        end
        pause(.01);
    end
    
end
end