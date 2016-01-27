function model = cascTrain(X, Y, opts, cascPrior)
% Train a random casc
% X is NxD, each D-dimensional row is a data point
% Y is Nx1 discrete labels of classes
% returned model is to be directly plugged into cascTest

% cascPrior must have
%   a targetSign field that encodes the DNF formula of concepts that the target concept is known to represent i.e. the logical composition side information or the "Signature"
%   NOTE For now, assuming the logical formula is only a conjunction. With other types of formulae, will allow all the conjunctive clauses to be specified, and would have to change the code accordingly.
%   (proposed) an otherSigns field that encodes the signatures of other clusters if any that are known to exist among the negatives of the current class in the data

D=size(X,2);
if nargin<4
  cascPrior=struct('targetSign', zeros(1,D)/D);
end

d= 5; % max depth of the casc
%priorFrac= 0.2; % fraction of nodes to apply prior to, at each level in the casc
select=true;

if nargin < 3, opts= struct; end
if isfield(opts, 'depth'), d= opts.depth; end
if isfield(opts, 'select'), select= opts.select; end
%if isfield(opts, 'priorFrac'), priorFrac= opts.priorFrac; end % determines the fraction of nodes at each level that will be pushed towards target class composition acquired from side information
%priorFrac=max(min(priorFrac,1),0); % keeping priorFrac between 0 and 1

%u= unique(Y);
u=[0;1];
assert(all(ismember(Y,u))); % all data is labeled as 0 and 1 (this is so that we know for sure what is positive)
[N, D]= size(X);
nd= 2*d+1;
numInternals = d;
numLeafs= d+1;

weakModels= cell(1, numInternals);
% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
if storage([N nd]) < 500 % increasing limit to allow storage as full array in RAM
    dataix= zeros(N, nd); % boolean indicator of data at each node
else
    dataix= sparse(N, nd);
end

leafdist= zeros(numLeafs, length(u)); % leaf distribution

% Propagate data down the casc while training weak classifiers at each node
nodeConjunctions=zeros(numInternals+numLeafs,D); % stores the logical formulae of each node
nodeDist=zeros(numInternals+numLeafs,1); % stores the distance of the logical formulae of each node from each target clause
nodeDist(1,:)=pdist2(zeros(1,D), cascPrior.targetSign, 'cityblock'); % root node does not represent any logical formula


for n = 1: numInternals
  % get relevant data at this node
  if n==1
      reld = ones(N, 1)==1;
      Xrel= X;
      Yrel= Y;
  else
      reld = dataix(:, n)==1;
      Xrel = X(reld, :);
      Yrel = Y(reld);
  end

  % set prior based on what nodes were selected before
    tmp=nodeConjunctions(n,:);
    targetSign=cascPrior.targetSign;
    tmp(tmp~=0)=targetSign(tmp~=0); % to avoid reuse of variables
    if select, nodePrior.pdf=abs(targetSign-tmp)+eps; % previously used variables are always set to 0 (i.e. eps) after this step
    else nodePrior.pdf=abs(targetSign);
    end

    nodePrior.pdf=nodePrior.pdf/sum(nodePrior.pdf); %making a valid pdf

  nodePrior.priorMethod=cascPrior.priorMethod;
  nodePrior.threshMethod='';% irrelevant at this point

  % train weak model
  weakModels{n}= weakTrain(Xrel, Yrel, opts, nodePrior);

  % update nodeConjunctions of the children: n+1 (the positive node, to be continued) and n+d (the negative (leaf) node, to be discontinued) and their distance from targetConjunction (assuming only decision stumps)
  delta=zeros(1,D); delta(weakModels{n}.r)=1;
  nc0=nodeConjunctions(n,:);



  % split data to child nodes
  yhat= weakTest(weakModels{n}, Xrel, opts);

  cntpos1=sum(yhat==1 & Yrel==1);
  cntpos0=sum(yhat==0 & Yrel==1);
  if cntpos1>cntpos0
    weakModels{n}.posTest=true; % indicates that if answer to question is yes, indicates positive class
    dataix(reld, n+1)= yhat;  % the one with more positive data out of yhat and 1-yhat
    dataix(reld, n+d+1)= 1-yhat; % the other one
  else
    weakModels{n}.posTest=false; % indicates that if answer to question is yes, indicates negatives class
    dataix(reld,n+1)= 1-yhat;
    dataix(reld,n+d+1)=yhat;
  end

  %if n+1<=numInternals % only calculating nodeDist for the internal (positive) node
    if cntpos1>cntpos0
      nodeConjunctions(n+1,:)=min(max(nc0+delta,-1),+1);%nc0+delta if yhat represents more positive data, else nc0-delta; % left child
      nodeConjunctions(n+d+1,:)=min(max(nc0-delta,-1),+1);%nc0+delta if yhat represents more positive data, else nc0-delta; % left child
    else
       nodeConjunctions(n+1,:)=min(max(nc0-delta,-1),+1);%nc0+delta if yhat represents more positive data, else nc0-delta; % left child
       nodeConjunctions(n+d+1,:)=min(max(nc0+delta,-1),+1);%nc0+delta if yhat represents more positive data, else nc0-delta; % left child
     end
    % Calculate distances from each conjunctive clause in the DNF
    nodeDist(n+1,:)=pdist2(abs(nodeConjunctions(n+1,:)), cascPrior.targetSign, 'cityblock');
  %end
end

% Go over leaf nodes and assign class statistics
for n=d+1:2*d+1
    reld= dataix(:, n);
    hc = histc(Y(reld==1), u);
    hc = hc + 1; % Dirichlet prior
    leafdist(n-d, :)= hc / sum(hc);
end

model.leafdist= leafdist;
model.depth= d;
model.classes= u;
model.weakModels= weakModels;
end
