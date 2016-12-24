function [model] = cnn(X,y,k)
% [model] = knn(X,y,k)
%
% Implementation of k-nearest neighbour classifer
[n,d] = size(X);

model.X = X(1,:);
model.y = y(1);
model.k = 1;
model.predict = @predict;
ind=[];
for i=1:n
    % if the next element is incorrectly classified then add it to the
    % subset. Else do nothing
    if y(i) ~= model.predict(model,X(i,:))
        model.X(end+1,:) = X(i,:);
        model.y(end+1) = y(i);
        ind= [ind i];
    end
end

model.k = 1;
model.ind = ind;
model.c = max(y);

end

function [yhat] = predict(model,Xtest)
[n,d] = size(model.X);
[t,d] = size(Xtest);
% D gives the square distance. n*t matrice. 
% Note that if dist^2 is bigger <=> |dist| bigger so we don't need to take sqrt 
D = model.X.^2*ones(d,t) + ones(n,d)*(Xtest').^2 - 2*model.X*Xtest';

% Finds k nearest training neigbours in O(n)
for j=1:t
    % chooses K values. Sorts them and goes through all the vector. If
    % it finds a value smaller than the K stored values then saves it.
    testValue=(D(:,j))';
    [currentKMin indices]=sort(testValue(1:model.k));
    for i=model.k+1:n
        if currentKMin(end)>testValue(i);
            currentKMin(end)=testValue(i);
            indices(end)=i;
            [currentKMin subIndex]=sort(currentKMin);
            % sort the index array depending on the sorting of the element
            indices=indices(subIndex);
        end
    end
    
    % Assigns y as the mode of K nearest neighbours
    yhat(j,1)=mode(model.y(indices));
end

end