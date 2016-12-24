function [f,g] = MLPregressionLoss(Ww,X,y,nHidden,lambda,typeNN)
[nInstances,nVars] = size(X);

% Form Weights
W1 = reshape(Ww(1:nVars*nHidden(1)),nVars,nHidden(1)); % Weight matrix W(1)
W1(1)=0; 
startIndex = nVars*nHidden(1);
for layer = 2:length(nHidden) % Weight matrix W(m) for m > 1
  WmReshape = reshape(Ww(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)),nHidden(layer-1),nHidden(layer));
  WmReshape(:,1)=0;
  Wm{layer-1} = WmReshape;
  startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
end
w = Ww(startIndex+1:startIndex+nHidden(end)); % Final weight vector 'w'

if strcmp(typeNN,'overfit')
    %h = @(x) (sin(x)+1)/2; % Activation function
    %dh = @(z) (cos(z))/2; % Derivative of activiation function
    h = @(x) log(2+sin(x))./log(3); % Activation function
    dh = @(z) (cos(z)./(sin(z)+2))./log(3); % Derivative of activiation function
else
    h = @(x) log(1+exp(x)); % Activation function
    dh = @(z) 1./(1+exp(-z)); % Derivative of activiation function
end

% Initialize gradient vector
f = 0;
if nargout > 1
    gradInput = zeros(size(W1));
    for layer = 2:length(nHidden)
       gradHidden{layer-1} = zeros(size(Wm{layer-1})); 
    end
    gradOutput = zeros(size(w));
end

% Compute Output
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
    yhat = z{end}*w;
    
    r = yhat-y(i);
    f = f + r^2;
    
    if nargout > 1
        dr = 2*r;
        err = dr;

        % Output Weights
        gradOutput = gradOutput + err*z{end}';

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            backprop = err*(dh(innerProduct{end}).*w');
            gradHidden{end} = gradHidden{end} + z{end-1}'*backprop;

            % Other Hidden Layers
            for layer = length(nHidden)-2:-1:1
                backprop = (backprop*Wm{layer+1}').*dh(innerProduct{layer+1});
                gradHidden{layer} = gradHidden{layer} + z{layer}'*backprop;
            end

            % Input Weights
            backprop = (backprop*Wm{1}').*dh(innerProduct{1});
            gradInput = gradInput + X(i,:)'*backprop;
        else
            % Input Weights
            gradInput = gradInput + err*X(i,:)'*(dh(innerProduct{end}).*w');
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(Ww));
    gradInput(1)=0;
    g(1:nVars*nHidden(1)) = gradInput(:);
    startIndex = nVars*nHidden(1);
    for layer = 2:length(nHidden)
        gradHidden{layer-1}(:,1)=0;
        g(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)) = gradHidden{layer-1};
        startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
    end
    g(startIndex+1:startIndex+nHidden(end)) = gradOutput;
end


if strcmp(typeNN,'underfit')
    f=f+lambda(1) /2 .* (w'*w)+lambda(end) /2 .* (W1*W1');
    dReg=zeros(size(Ww));
    dReg(1:length(W1'))=lambda(1).*W1';
    dReg(end-length(w)+1:end)=lambda(end).*w;
    for i = 2:length(nHidden)
        wm=Wm(i-1);
        vmArr=[wm{:}];
        vmVec=vmArr(:);
        f = f+ lambda(i)/2 * norm(vmArr,'fro')^2;
        zeroIndex=find(dReg==0,1);
        dReg(zeroIndex:zeroIndex+length(vmVec)-1)=lambda(i).*vmVec;
    end

    gnz=(g~=0);
    g(gnz)=g(gnz)+dReg(gnz);
end






