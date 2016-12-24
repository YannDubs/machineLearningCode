function [Z] = visualizeISOMAPnoINF(X,k,names)

[n,d] = size(X);

% Compute all distances
D = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*X';
D = sqrt(abs(D));

Dknn=zeros(n);
%makes an array containing [0 50 100 ...] repeated for n columns
% sorts the array and keeps track of the indexes
% makes a directed weighted graph
r=repmat(0:n:n^2-1,n,1);
[sortD I]=sort(D);
Ibis=I+r;
k=2;
Dknn(Ibis(2:1+k,:))=sortD(2:1+k,:);
% transform te graph into an undirected graph 
%(KNN if on of the two are in KNN of the other one)
% simply adds the values of the transpose which are different
DknnT=Dknn';
Dknn(DknnT~=Dknn)=Dknn(DknnT~=Dknn)+DknnT(DknnT~=Dknn);

% computes the geodisc distance with djikstra
Ddjik=zeros(n);
% the matrix will be symmetric so computes upper half and replicate for
% bottom
for source = 1:n
    for destination = source:n
        Ddjik(source,destination) = dijkstra(Dknn, source, destination);
    end
end
Ddjik = Ddjik + Ddjik' - diag(diag(Ddjik));

%computes the maximum value not infinite and assign it to all infinite
%values
Ddjik(isinf(Ddjik(:,:)))=max(max(Ddjik(isfinite(Ddjik(:,:)))));

% Initialize low-dimensional representation with PCA
model = dimRedPCA(X,2);
Z = model.compress(model,X);

Z(:) = findMin(@stress,Z(:),500,0,Ddjik,names);

end

function [f,g] = stress(Z,D,names)

n = length(D);
k = numel(Z)/n;

Z = reshape(Z,[n k]);

f = 0;
g = zeros(n,k);
for i = 1:n
    for j = i+1:n
        % Objective Function
        Dz = norm(Z(i,:)-Z(j,:));
        s = D(i,j) - Dz;
        f = f + (1/2)*s^2;
        
        % Gradient
        df = s;
        dgi = (Z(i,:)-Z(j,:))/Dz;
        dgj = (Z(j,:)-Z(i,:))/Dz;
        g(i,:) = g(i,:) - df*dgi;
        g(j,:) = g(j,:) - df*dgj;
    end
end
g = g(:);

% Make plot if using 2D representation
if k == 2
    figure(3);
    clf;
    plot(Z(:,1),Z(:,2),'.');
    if ~isempty(names)
        hold on;
        for i = 1:n
            text(Z(i,1),Z(i,2),names(i,:));
        end
    end
    pause(.01)
end
end