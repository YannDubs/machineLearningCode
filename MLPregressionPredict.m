function [y] = MLPregressionPredict(Ww,X,nHidden,typeNN)

[nInstances,nVars] = size(X);

% Form Weights
W1 = reshape(Ww(1:nVars*nHidden(1)),nVars,nHidden(1));
%W1(1)=0; 
startIndex = nVars*nHidden(1);
for layer = 2:length(nHidden)
    WmReshape =reshape(Ww(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)),nHidden(layer-1),nHidden(layer));
    %WmReshape(:,1)=0;
    Wm{layer-1} = WmReshape;
    startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
end
w = Ww(startIndex+1:startIndex+nHidden(end));

if strcmp(typeNN,'overfit')
    %h = @(x) (sin(x)+1)/2;
    h = @(x) log(2+sin(x))./log(3) % Activation function
else
    h = @(x) log(1+exp(x)); % Activation function
end

% Compute Output
y = zeros(nInstances,1);
for i = 1:nInstances
    innerProduct{1} = X(i,:)*W1;
    z{1} = h(innerProduct{1});
    z{1}(1)=1;
    for layer = 2:length(nHidden)
        innerProduct{layer} = z{layer-1}*Wm{layer-1};
        z{layer} = h(innerProduct{layer});
        z{layer}(1)=1;
    end
    z{end}(1)=1;
    y(i) = z{end}*w;
end
