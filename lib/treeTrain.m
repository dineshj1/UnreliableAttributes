function model = treeTrain(X, Y, opts, treePrior)
% Train a random tree
% X is (a cell array of) NxD, each D-dimensional row is a data point
% Y is (a cell array of) Nx1 discrete labels of classes
% returned model is to be directly plugged into treeTest

% treePrior must have
%   a samplingProb field that encodes the DNF formula of concepts that the target concept is known to represent i.e. the logical composition side information or the "Signature"
%   NOTE For now, assuming the logical formula is only a conjunction. With other types of formulae, will allow all the conjunctive clauses to be specified, and would have to change the code accordingly.
%   (proposed) an otherSigns field that encodes the signatures of other clusters if any that are known to exist among the negatives of the current class in the data

% if opts.valProp==false, opts.classifierKnowledge is a numAttr-length cell array such that opts.classifierKnowledge{k} is a ?\times 3 matrix whose first column=FPR, second column=TPR at the threshold value specified by the third column.

% if opts.valProp==true, opts.valData is a valData structure with fields X (being the (binary/continuous) attribute predictions) and annot (being the actual attribute values)

assert(iscell(X));
numDatasets=length(X);
assert(numDatasets<=2, 'Higher than 2 datasets not currently supported');
if numDatasets==1
  opts.dsWts=[1 0]; % doesn't really matter but just for consistency, setting all the weight on first dataset
end

D=size(X{1},2);
if nargin<4
  treePrior=struct('samplingProb', zeros(1,D)/D);
end

d= 5; % max depth of the tree
priorFrac= 0.2; % fraction of nodes to apply prior to, at each level in the tree

if nargin < 3, opts= struct; end
if isfield(opts, 'depth'), d= opts.depth; end
if isfield(opts, 'priorFrac'), priorFrac= opts.priorFrac; end % determines the fraction of nodes at each level that will be pushed towards target class composition acquired from side information
if ~isfield(opts, 'select'), opts.select= true; end % determines the fraction of nodes at each level that will be pushed towards target class composition acquired from side information
if ~isfield(opts, 'classifierKnowledge')
  numAttr=size(X{1},2);
  opts.classifierKnowledge=repmat({[ 0,1,1;0,1,0.5;1,1,0]}, numAttr, 1);
end
opts.valProp=false;
if isfield(opts, 'valData')
  opts.valProp=true;
  valData=opts.valData;
end

priorFrac=max(min(priorFrac,1),0); % keeping priorFrac between 0 and 1
if length(Y)>1
  u= union(Y{:});
else
  u=unique(Y{1});
end

[N, D]= size(X{1});
nd= 2^d - 1;
numInternals = (nd+1)/2 - 1;
numLeafs= (nd+1)/2;
numClauses=size(treePrior.samplingProb,1);

weakModels= cell(1, numInternals);
% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
dataix=cell(1,numDatasets);
for dsno=1:numDatasets
  N= size(X{dsno},1);
  if storage([N nd]) < 500 % increasing limit to allow storage as full array in RAM
      dataix{dsno}= zeros(N, nd); % boolean indicator of data at each node
  else
      dataix{dsno}= sparse(N, nd);
  end
end


leafdist= zeros(numLeafs, length(u)); % leaf distribution

% Propagate data down the tree while training weak classifiers at each node
nodeConjunctions=zeros(numInternals+numLeafs,D); % stores the logical formulae of each node
nodeDist=zeros(numInternals+numLeafs,numClauses); % stores the distance of the logical formulae of each node from each target clause
nodeDist(1,:)=pdist2(zeros(1,D), treePrior.samplingProb, 'cityblock'); % root node does not represent any logical formula

selNodes=false(numInternals,numClauses); % stores which clause, if any, each node is assigned to
currlevel=0;
tgtDsNo=1; % TODO pass this as a parameter?
srcDsNo=length(X);
for n = 1: numInternals
  if opts.debug>5
    if n==20
      abort; % too much information to display for full tree at once
    end
  end
  if opts.debug
    fprintf('Node %d/%d\n',n, numInternals);
  end
  % is this the beginning of a new level? i.e is n=2^m for integer m?
  if ~bitand(n,n-1) % new level
    currlevel=currlevel+1;
    % select opt.priorFrac nodes among the n nodes at this level to apply prior to
    % numSel=ceil(priorFrac*n);
    % must access each node's distance from target conjunction
    levelMembers=n:2*n-1;
    %if ~opts.select
    %  numSel=round(priorFrac*numNodes);
    %  tmp=levelMembers(randperm(n));
    %  if numSel>0
    %   selNodes(n:2*n-1,:)=tmp(1:numSel);
    %  end
    %else
      % assign specific clauses to current level nodes
      % binary integer program optimization
      tmp=assignNodes(nodeDist(levelMembers,:),priorFrac, opts.select);
      selNodes(n:2*n-1,:)=tmp;
      % idea: sort nodes by distance from each clause, then assign rank(node,clause) to each node for each clause
      % each node is then assigned to the clause for which it ranks highest

      % Qn: Does this preserve clauses in the tree paths, or does it trend to break them up?

      % old code
      %[~,ord]=sort(nodeDist(levelMembers), 'ascend');
      %selNodes(levelMembers(ord(1:numSel)))=true;
    %end
  end

  % get relevant data at this node
  reld=cell(1,numDatasets);
  for dsno=1:numDatasets
    if n==1
      if opts.valProp, valreld = ones(size(valData.X,1),1); end
      for dsno=1:numDatasets
        reld{dsno} = ones(size(X{dsno},1), 1); % reld is a fraction between 0 and 1 or each combination of (data point, node) depending on the probability with which that data point reaches that node.
        %Xrel{dsno}= X{dsno};
        %Yrel{dsno}= Y{dsno};
      end
    else
      if opts.valProp, valreld=valdataix(:,n); end
      for dsno=1:numDatasets
        reld{dsno} = dataix{dsno}(:, n);
        %Xrel{dsno} = X{dsno};%(reld{dsno}, :);
        %Yrel{dsno} = Y{dsno};%(reld{dsno});
      end
    end
  end

  % set prior for each node based on what nodes were selected before
  if sum(selNodes(n,:)) % if node is assigned to some clause
    tmp=nodeConjunctions(n,:);
    targetSign=mean(treePrior.samplingProb(selNodes(n,:),:),1); % taking mean to account for the possibility of a node being allotted to multiple clauses
    if ~opts.select
      nodePrior.pdf=targetSign; % not accounting for previously selected nodes
    else
      tmp(tmp~=0)=targetSign(tmp~=0); % to avoid reuse of variables
      nodePrior.pdf=abs(targetSign-tmp)+eps; % previously used variables are always set to 0 (i.e. eps) after this step
      nodePrior.pdf=nodePrior.pdf/sum(nodePrior.pdf); %making a valid pdf
    end
  else
    nodePrior.pdf=ones(1,D)/D;
  end
  nodePrior.priorMethod=treePrior.priorMethod;
  nodePrior.threshMethod='';% irrelevant at this point

  % train weak model
  if opts.valProp
    % instead of passing classifierKnowledge, pass confidence scores and ground truth attribute annotations for all validation data propagated to this node. i.e. a nval x nattr x 2 matrix
    valData.reld=valreld;
    weakModels{n}=weakTrain(X, Y, opts, nodePrior, reld, valData); % valData must have X (attribute predictions) and annot (attribute annotations) and reld
  else
    weakModels{n}=weakTrain(X, Y, opts, nodePrior, reld, opts.classifierKnowledge);
  end
%   if opts.debug
%     % temporary
%     fprintf('Level %d, Node %d, Igain:%f\n',currlevel, n, weakModels{n}.Igain_net);
%     disp(weakModels{n}.Igain_ds);
%   end

  % add metadata to weak model
  weakModels{n}.ypos=-currlevel;
  weakModels{n}.gaps=2^(d-currlevel);
  weakModels{n}.xpos=weakModels{n}.gaps*(mean(levelMembers)-n);
  weakModels{n}.levelPos=find(levelMembers==n);
  weakModels{n}.neg=sum(reld{tgtDsNo}.*(Y{tgtDsNo}==0))/sum(reld{tgtDsNo})+eps;
  weakModels{n}.pos=sum(reld{tgtDsNo}.*(Y{tgtDsNo}==1))/sum(reld{tgtDsNo})+eps;

  if opts.debug
    disp(weakModels{n});
    disp(weakModels{n}.quantOpts);
  end

  % update nodeConjunctions of the children: 2n and 2n+1 and their distance from targetConjunction (assuming only decision stumps)
  delta=zeros(1,D); delta(weakModels{n}.r)=1;
  nc0=nodeConjunctions(n,:);
  %if 2*n+1<=numInternals
    nodeConjunctions(2*n,:)=min(max(nc0-delta,-1),+1); % left child
    nodeConjunctions(2*n+1,:)=min(max(nc0+delta,-1),+1); % right child
    % Calculate distances from each conjunctive clause in the DNF
    nodeDist(2*n,:)=pdist2(abs(nodeConjunctions(2*n,:)), treePrior.samplingProb, 'cityblock');
    %sum(abs(nodeConjunctions(2*n,:)-targetSign));
    nodeDist(2*n+1,:)=pdist2(abs(nodeConjunctions(2*n+1,:)), treePrior.samplingProb, 'cityblock'); % fixed a bug that was assigning this value to be the same as nodeDist(2*n)
  %end

  % split data to child nodes
  for dsno=1:numDatasets
    if isempty(X{dsno}), continue; end
    wTopts=opts;
    if dsno==srcDsNo
      wTopts.useFpTp=1;
    else
      wTopts.useFpTp=0;
    end
    yhat{dsno}= weakTest(weakModels{n}, X{dsno}, wTopts, weakModels{n}.assumedCK); % works both in the valProp and ~valProp(i.e. wCK) cases, also note that the last argument, rather weirdly must depend on the actual attribute being used by the model
    dataix{dsno}(:, 2*n)= reld{dsno}.*yhat{dsno};  % left child
    dataix{dsno}(:, 2*n+1)= reld{dsno}.*(1-yhat{dsno}); % right child % since yhat is in {0,1} and double
    if opts.debug
      sum(dataix{dsno}(:,2*n))
      sum(dataix{dsno}(:,2*n+1))
    end
  end

  if opts.valProp
    % Also split validation data to child nodes
    wTopts=opts; wtopts.useFpTp=false;
    yhatVal= weakTest(weakModels{n}, valData.X, wTopts); % no classifier knowledge needs to be provided here - we will make discrete choices about which data points should go where
    valdataix(:, 2*n)= valreld.*yhatVal;  % left child
    valdataix(:, 2*n+1)= valreld.*(1-yhatVal); % right child % since yhatVal is in {0,1} and double
    if opts.debug
      sum(valdataix(:,2*n))
      sum(valdataix(:,2*n+1))
    end
  end
end

% Go over leaf nodes and assign class statistics
%fprintf('Assigning node probabilities');
for n= (nd+1)/2 : nd
  pdf_net=zeros(length(u),1);
  for dsno=1:numDatasets
    if isempty(X{dsno}), continue; end
    reld{dsno}= dataix{dsno}(:, n);
    epsilon=max(1e-3*sum(reld{dsno}),eps); % small compared to the number of data points
    [~,bins]= histc(Y{dsno}, u);
    hc=accumarray(bins, reld{dsno})+epsilon;
    pdf{dsno}=hc/sum(hc);
    pdf_net=pdf_net+pdf{dsno}*opts.dsWts(dsno);
  end

  pdf_net=pdf_net/sum(pdf_net);% should actually already be normalized
  leafdist(n - (nd+1)/2 + 1, :)= pdf_net;

  if opts.debug
    n
    pdf_net
  end
  assert(all(~isnan(pdf_net)));
end

%if opts.debug
%  Igains=cellfun(@(x) x.Igain_net, weakModels);
%  disp(sum(Igains));
%  disp(sum(Igains)/length(weakModels));
%end
%model.Igain=sum(Igains);
model.leafdist= leafdist;
model.depth= d;
model.classes= u;
model.weakModels= weakModels;
end
