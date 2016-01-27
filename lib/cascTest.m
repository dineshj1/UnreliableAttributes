function [Yhard, Ysoft] = cascTest(model, X, opts)
% Test a casc
% X is NxD, each D-dimensional row is a data point
% model comes from cascTrain()
% Yhard are hard assignments to X's, Ysoft is NxK array of
% probabilities, where there are K classes.

if nargin < 3, opts= struct; end

d= model.depth;

[N, D]= size(X);
nd= 2*d+1;
numInternals = d;
numLeafs= d+1;

Yhard= zeros(N, 1);
u= model.classes;
if nargout>1, Ysoft= zeros(N, length(u)); end

% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
if storage([N nd]) < 100
    dataix= zeros(N, nd); % boolean indicator of data at each node
else
    dataix= sparse(N, nd);
end

% Propagate data down the casc using weak classifiers at each node
for n = 1: numInternals

    % get relevant data at this node
    if n==1
        reld = ones(N, 1)==1; % reld is 1 for indices that are to be propagated to current node
        Xrel= X;
    else
        reld = dataix(:, n)==1;
        Xrel = X(reld, :);
    end
    if size(Xrel,1)==0, continue; end % empty branch, ah well

    yhat= weakTest(model.weakModels{n}, Xrel, opts);

    if model.weakModels{n}.posTest
      dataix(reld, n+1)= yhat; % setting relevant indices for children nodes
      dataix(reld, n+d+1)= 1 - yhat; % since yhat is in {0,1} and double
    else
      dataix(reld, n+1)= 1-yhat;
      dataix(reld, n+d+1)=yhat;
    end
end

% Go over leafs and assign class probabilities
for n=d+1:2*d+1
    ff= find(dataix(:, n)==1);

    hc= model.leafdist(n-d, :);
    vm= max(hc);
    miopt= find(hc==vm);
    mi= miopt(randi(length(miopt), 1)); %choose a class arbitrarily if ties exist
    Yhard(ff)= u(mi);

    if nargout > 1
        Ysoft(ff, :)= repmat(hc, length(ff), 1);
    end
end
