function [Yhard, Ysoft] = treeTest(model, X, opts)
% Test a tree
% X is NxD, each D-dimensional row is a data point
% model comes from treeTrain()
% Yhard are hard assignments to X's, Ysoft is NxK array of
% probabilities, where there are K classes.

if nargin < 3, opts= struct; end

d= model.depth;

[N, D]= size(X);
nd= 2^d - 1;
numInternals = (nd+1)/2 - 1;
numLeafs= (nd+1)/2;

Yhard= zeros(N, 1);
u= model.classes;
if opts.debug
  u
end
if nargout>1, Ysoft= zeros(N, length(u)); end

% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
if storage([N nd]) < 100
    dataix= zeros(N, nd); % boolean indicator of data at each node
else
    dataix= sparse(N, nd);
end

% Propagate data down the tree using weak classifiers at each node
for n = 1: numInternals
    if opts.debug>4
      n
      if n==30
        abort;
      end
    end

    % get relevant data at this node
    if n==1
        reld = ones(N, 1)==1; % reld is 1 for indices that are to be propagated to current node
        Xrel= X;
    else
        reld = dataix(:, n)==1;
        Xrel = X(reld, :);
    end
    if size(Xrel,1)==0, continue; end % empty branch, ah well
    if opts.debug, n, end
    if n==5,
      5; % dummy
    end
    yhat= weakTest(model.weakModels{n}, Xrel, opts);
    if opts.debug>4
      if n==5,
        disp(model.weakModels{n});
        disp(model.weakModels{n}.quantOpts);
        yhat
      end
    end
    dataix(reld, 2*n)= yhat; % setting relevant indices for children nodes
    dataix(reld, 2*n+1)= 1 - yhat; % since yhat is in {0,1} and double
end

% Go over leafs and assign class probabilities
for n= (nd+1)/2 : nd
    ff= find(dataix(:, n)==1);

    hc= model.leafdist(n - (nd+1)/2 + 1, :);
    vm= max(hc);
    miopt= find(hc==vm);
    %try
      mi= miopt(randi(length(miopt), 1)); %choose a class arbitrarily if ties exist
    %catch
    %  5; % dummy
    %end

    Yhard(ff)= u(mi);

    if nargout > 1
        Ysoft(ff, :)= repmat(hc, length(ff), 1);
    end
end
