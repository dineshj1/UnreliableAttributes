function model = weakTrain(X, Y, opts, prior, origProb, clsfrInfo)
% weak random learner
% can currently train:
% 1. decision stump: look along random dimension of data, choose threshold
% that maximizes information gain in class labels
% 2. 2D linear decision learner: same as decision stump but in 2D. I know,
% in general this could all be folded into single linear stump, but I make
% distinction for historical, cultural, and efficiency reasons.
% 3. Conic section learner: second order learning in 2D. i.e. x*y is a
% feature in addition to x, y and offset (as in 2.)
% 4. Distance learner. Picks a data point in train set and a threshold. The
% label is computed based on distance to the data point

% classifierKnowledge is a numAttr-length cell array such that opts.classifierKnowledge{k} is a ?\times 3 matrix whose first column=FPR, second column=TPR at the threshold value specified by the third column.


classifierID= 1; % by default use decision stumps only
numVarsPerNode= 5;
numSplitsPerVar= 5;
classifierCommitFirst= true;
dsWts=[1,0]; % dataset weights when determining information gain
tgtDsNo=1;
srcDsNo=length(X);

if nargin < 3, opts = struct; end
if isfield(opts, 'classifierID'), classifierID = opts.classifierID; end
if isfield(opts, 'numVarsPerNode'), numVarsPerNode = opts.numVarsPerNode; end
if isfield(opts, 'numSplitsPerVar'), numSplitsPerVar = opts.numSplitsPerVar; end
if isfield(opts, 'classifierCommitFirst'), classifierCommitFirst = opts.classifierCommitFirst; end
if isfield(opts, 'dsWts'), dsWts = opts.dsWts; end
if ~isfield(opts, 'ROC'), opts.ROC=false; end
if ~isfield(opts, 'numbins'), opts.numbins=2; end
if ~isfield(opts, 'leafFrac'), opts.leafFrac=0; end % no stopping criterion!

if classifierCommitFirst
    % commit to a weak learner first, then optimize its parameters only. In
    % this variation, different weak learners don't compete for a node.
    if length(classifierID)>1
        classifierID= classifierID(randi(length(classifierID)));
    end
end
valData=[];
classifierKnowledge={[]};
if opts.valProp
  valData=clsfrInfo;
else
  classifierKnowledge=clsfrInfo;
end

if ~iscell(X)
  X={X};
  Y={Y};
end

if length(Y)>1
  u= union(Y{:}); % unique values among classes
else
  u=unique(Y{1});
end

[~, D]= size(X{tgtDsNo});
numDatasets=length(X);

if nargin < 4, prior = struct('pdf', ones(1,D)/D, 'priorMethod', '', 'threshMethod', ''); end % uniform prior
% Making sure prior has all required fields etc.
if ~isfield(prior, 'pdf')
  prior.pdf=ones(1,D)/D;
elseif isempty(prior.pdf)
  prior.pdf=ones(1,D)/D;
end

if ~isfield(prior, 'priorMethod')
  prior.priorMethod='varSel';
elseif isempty(prior.priorMethod)
  prior.priorMethod='varSel';
end

if ~isfield(prior, 'threshMethod') % option not functional any more
  prior.threshMethod='';
end

assert(length(prior.pdf)==D, sprintf('prior.pdf must be a D(=%d)-length vector',D));
assert(all(prior.pdf>=0), 'prior pdf must be non-negative');


if nargin<5,
  for i=1:length(datasets)
    origProb{i}=ones(1,length(X{srcDsNo}));
  end  % source dataset
end

% setting boundaries variable (cdf used in variable selection)
switch prior.priorMethod
  case 'varSel' % prior on variable selection
    boundaries=cumsum(prior.pdf);
  case 'splitEval' % prior used in assigning scores to splits
    boundaries=cumsum(ones(1,D)/D); % making boundaries into the cdf of a uniform distribution so that no bias is used in variable selection
  otherwise
    error(sprintf('Unknown priorMethod %s', prior.priorMethod));
end
%assert(boundaries(end)==1, 'Prior probabilties do not add to 1'); % not
%necessary. Compensated for later when dropping pin.
N=0;
for dsno=1:numDatasets
  N=N+size(X{dsno},1);
end

minProb=10^-5;
%emptyClsfr=false;
% if numDatasets==2
%   tgtDsNo=setdiff(1:numDatasets, srcDsNo);
% stopping criterion -------------------------------
try
  % 1- enough data
  assert((sum(origProb{srcDsNo})+sum(origProb{tgtDsNo}))>opts.leafFrac*(length(origProb{srcDsNo})+length(origProb{tgtDsNo})),'emptyClsfr'); % also works when srcDsNo and tgtDsNo are the same i.e. only one dataset
  % 2 - not all belonging to same class
  assert(length(union(Y{srcDsNo}(origProb{srcDsNo}>minProb),Y{tgtDsNo}(origProb{tgtDsNo}>minProb)))>1,'emptyClsfr');
  % assert any other criteria if necessary
catch err % empty classifier
  if strcmp(err.message,'emptyClsfr')
    model.classifierID= 0;
    model.r=[];
    model.t=[];
    model.quantOpts=struct('range',[],'numbins',[]);
    model.Igain_net=0;
    model.Igain_ds=0;
    model.assumedCK=[0 1];
    return;
  else
    %err.identifier
    %getReport(err)
    rethrow(err);
  end
end
%if sum(origProb{srcDsNo}>minProb)+sum(origProb{tgtDsNo}>minProb)==0
%  emptyClsfr=true;
%end
% elseif numDatasets==1
%   if sum(origProb>minProb) == 0 % no samples in source dataset
%       % edge case. No data reached this leaf. Don't do anything...
%      emptyClsfr=true;
%   end
% else
%   abort
% end
%if emptyClsfr
%  model.classifierID= 0;
%  model.r=[];
%  model.t=[];
%  model.quantOpts=struct('range',[],'numbins',[]);
%  model.Igain_net=0;
%  model.Igain_ds=0;
%  model.assumedCK=[0 1];
%  return;
%end

bestgain= -Inf;
model = struct;
% Go over all applicable classifiers and generate candidate weak models
for classf = classifierID

    modelCandidate= struct;
    maxgain_net= -Inf;
    %maxgain_ds= -Inf*ones(1,numDatasets);

    if classf == 1
        % Decision stump

        % proceed to pick optimal splitting value t, based on Information Gain
        numSplits=numSplitsPerVar*numVarsPerNode;

        for q= 1:numSplits

            if mod(q-1,numSplitsPerVar)==0
              % drop a pin and find the bin (bins specified by boundaries) that the pin drops in
              pindrop=rand(1)*boundaries(end);
              r=find(pindrop<boundaries,1);
              if opts.debug>5
                  fprintf('r:%d,',r);
              end

              tmin=Inf; tmax=-Inf;
              levs=[];
              for dsno=1:numDatasets
                if isempty(X{dsno})
                  continue;
                end
                col= X{dsno}(:, r);
                if dsno==srcDsNo
                  col=col(origProb{dsno}>minProb);
                else
                  continue; % not including validation data while computing levels...TODO is this okay?
                end
                levs=union(levs,unique(col));
                %if ~isempty(col)
                %  tmin= min(tmin,min(col));
                %  tmax= max(tmax,max(col));
                %end
              end
              tmin=min(tmin,min(levs));
              tmax=max(tmax,max(levs));
              clear('col');
              % temporary(?)
              tmin=0; tmax=1;
            end

            %if opts.valProp
            %  tmp=(valData.reld>minProb);
            %  levs=union(levs, unique(valData.X(tmp,r)));
            %end

            % setting threshold
            if opts.ROC

              % % this passage meant for testing ROC vs noROC
              % t=mean(levs);
              % numSplitsPerVar=1;

              if mod(q-1,numSplitsPerVar)==0 % first threshold is set to the simple rule-based split for each new candidate variable
                t=mean(levs);
              else
                if opts.valProp
                  % select from among the unique confidence values in valData?
                  bins=cumsum(valData.reld);
                  bins=bins/(bins(end)+eps);
                  pindrop=rand(1);
                  binNo=find(bins>pindrop,1);
                  if isempty(binNo)
                    [~,binNo]=min(abs(bins-pindrop));
                  end
                  t=valData.X(binNo,r);
                else
                  t=rand(1);
                  t=t*(tmax-tmin)+tmin;
                end
              end
            elseif ~isempty(levs) % binary data
              t=mean(levs);
              numSplitsPerVar=1; % if same every time, no point repeating with the same variable
            else
              t=0.5; % empty => cannot do any better
              %error('No unique values?? Something fishy!');
            end
            if opts.debug>5
              fprintf('t:%f,',t);
            end
            % computing FpTp
            if opts.valProp
              relInd=find(valData.reld>0.5);

              %range=[0,1];
              range=[min(valData.annot(:,r)), max(valData.annot(:,r))];

              %if ~isempty(relInd)
              %  range=[min(valData.annot(relInd,r)), max(valData.annot(relInd,r))];
              %else
              %  range=[min(valData.annot(:,r)), max(valData.annot(:,r))];
              %end
              quantOpts=struct('numbins', opts.numbins, 'range', range);
                annot=scoresToBins(valData.annot(:,r), quantOpts);
              annot_levs=1:scoresToBins(1, quantOpts);
              FpTp=NaN(1,length(annot_levs));

              annotlev_counts=hist(annot(relInd),annot_levs);
              if opts.debug>5
                disp(annotlev_counts);
              end
              if annotlev_counts>10 %TODO vary this
                minreld=0.5; % only relevant indices will be counted (remember reld is binary)
              else
                if opts.debug>5
                  fprintf('Not enough data to compute node-specific FpTp\n');
                end
                minreld=-Inf; % all indices will be counted
              end

              for i=1:length(annot_levs)
                currlev = annot_levs(i);
                P = sum(valData.X(:,r)>=t & annot==currlev & valData.reld>minreld)+eps; % number of <currlev> positives
                FpTp(i)=P/sum(annot==currlev & valData.reld>minreld);
                if FpTp(i)==Inf || FpTp(i)==-Inf,
                  if opts.debug>7
                    fprintf('\tCorrecting for FpTp=+/- Inf\n');
                  end
                  if annot_levs(i)/length(annot_levs)>=0.5
                    FpTp(i)=1;
                  else
                    FpTp(i)=0;
                  end
                elseif isnan(FpTp(i))
                  error('Something fishy.. this should already have been checked for');
                end
              end
            else % for the precomputed ROC curve case
              % unsupported for more than 2 annotation levels at the moment
              range=[0,1];
              if t<eps
                FpTp=[0,1]; % everything is >=0
              elseif t>1-eps
                FpTp=[0,1];% all positives, and no negatives are mapped to 1>=1
              else
                tIndx=find(classifierKnowledge{r}(:,3)>=t-eps,1,'last');
                if isempty(tIndx)
                  tIndx=find(abs(t-classifierKnowledge{r}(:,3))==min(abs(t-classifierKnowledge{r}(:,3))),1);
                  [~,tIndx]=min(abs(t-classifierKnowledge{r}(:,3)));
                end

                t=classifierKnowledge{r}(tIndx,3);
                FpTp=classifierKnowledge{r}(tIndx,1:2);
              end
            end

            if opts.debug>5
              disp(FpTp);
            end

            Igain_net=0;
            Igain_ds=zeros(1,numDatasets);
            for dsno=1:numDatasets
              if isempty(X{dsno})
                continue;
              end
              col = X{dsno}(:, r);
              %bins=col+1; % when signatures are restricted to 0 and 1
              if dsno==srcDsNo
                bins=scoresToBins(col, opts);
                p0 = FpTp(bins)'; % probability with which classifier r says 0 for ith point when actually X(i,model.r)
              else
                p0 = (col>=t);
              end
              prob_right=origProb{dsno}.*p0;

              if all(prob_right==0) || all(1-prob_right==1) %|| length(unique(Y{dsno}))==1
                Igain_ds(dsno) = 0;
              else
                Igain_ds(dsno) = evalDecision(Y{dsno}, prob_right, u, origProb{dsno});
              end

              if strcmp(prior.priorMethod, 'splitEval')
                % Igain should be multiplied by the prior for that particular variable?
                Igain_ds(dsno)=Igain_ds(dsno)*prior.pdf(r);
              end
%               if ~isnan(Igain_ds(dsno))
                Igain_net=Igain_net+Igain_ds(dsno)*dsWts(dsno);
%               else
%                 Igain_net=Igain_net+0; % 0 is equal to worst possible information gain value from two class data
%               end
            end

            if opts.debug>5
              fprintf('Igain_net: %f\n', Igain_net);
            end

            if Igain_net>maxgain_net
              maxgain_net = Igain_net;
              maxgain_ds = Igain_ds;
              modelCandidate.r= r;
              %modelCandidate.tIndx= tIndx;
              modelCandidate.t= t;
              modelCandidate.quantOpts.range=range;
              modelCandidate.quantOpts.numbins=opts.numbins;
              modelCandidate.assumedCK=FpTp; % contains the assumed TPR and FPR (either supplied, or computed in the valProp validation data propagation case)
              if ~isdeployed
                % temporary
                modelCandidate.Igain_net= Igain_net;
                modelCandidate.Igain_ds= Igain_ds;
              end
            end
        end
    else
        fprintf('Error in weak train! Classifier with ID = %d does not exist.\n', classf);
    end

    % see if this particular classifier has the best information gain so
    % far, and if so, save it as the best choice for this node
    if maxgain_net >= bestgain
        bestgain = maxgain_net;
        model= modelCandidate;
        model.classifierID= classf;
    end
end

end

function Igain= evalDecision(Y, rightProb, u, origProb)
% gives Information Gain provided a boolean decision array for what goes
% left or right. u is unique vector of class labels at this node

    leftProb=origProb-rightProb;
    %YL= Y(dec);
    %YR= Y(~dec);
    H= classEntropy(Y, u, origProb);
    HL= classEntropy(Y, u, leftProb);
    HR= classEntropy(Y, u, rightProb);
    Igain= H - sum(leftProb)/sum(origProb)*HL - sum(rightProb)/sum(origProb)*HR;
    %try
    assert(~isnan(Igain));
    %catch
    %  5;
    %end
    if Igain<0 % artifact caused by inclusion of epsilon in classEntropy computation
      Igain=0;
    end
%     try
%       if ~isnan(Igain), assert(Igain>=0); end
%     catch
%       0; % dummy
%     end
end

% Helper function for class entropy used with Decision Stump
function H= classEntropy(y, u, prob)
    epsilon=1e-3*sum(prob); % small compared to the number of data points

    epsilon=max(epsilon,eps); % to avoid NaNs
    [~,bins]= histc(y, u);
    cdist=accumarray(bins, prob)+epsilon;

    cdist= cdist/sum(cdist);
    cdist= cdist .* log(cdist);
    H= -sum(cdist);
end
