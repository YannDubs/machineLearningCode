function [probabilities] = randomWalk(M, alpha, nMax)
   [n ~]=size(M);
   node=randi(n,1);
   probabilities=zeros(n,1);
   for i = 1:nMax
       probabilities(node)= probabilities(node) + 1/nMax;
       % there is a probability of alpha to go random 
       % will always go random if no neigbhours 
       nNonZero=nnz(M(:,node));
       x=rand;
       if x<alpha | nNonZero == 0
           node=randi(n,1);
       else
           % gives all indices of non zero value of column (node)
           % then choses randomly one
           idx = find(M(:,node));
           node = idx(randi(nNonZero));
       end
   end
end