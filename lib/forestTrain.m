function model = forestTrain(X, Y, opts)
    % X is (a cell array of ) NxD matrices, where rows are data points
    % for convenience, for now we assume X is 0 mean unit variance. If it
    % isn't, preprocess your data with
    %
    % X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
    %
    % If this condition isn't satisfied, some weak learners won't work out
    % of the box in current implementation.
    %
    % Y is ( a cell array of ) discrete Nx1 vector of labels
    % model can be plugged into forestTest()
    %
    % decent default opts are:
    % opts.depth= 9;
    % opts.numTrees= 100; %(but more is _ALWAYS_ better, monotonically, no exceptions)
    % opts.numSplits= 5;
    % opts.numSplitsPerVar
    % opts.classifierID= 2
    % opts.classifierCommitFirst= true;
    % opts.priorFrac % no longer in use: supposed to be fraction of nodes at each level to apply prior to
    % opts.select % if true, doesn't keep track of nodes that have previously been selected
    % opts.classifierKnowledge % as of now, fpr and tpr rates for each attribute
    % opts.dsWts % weights to be assigned to various datasets in X (and Y) while computing information gains

    % which means use depth 9 trees, train 100 of them, use 5 random
    % splits when training each weak learner. The last option controls
    % whether each node commits to a weak learner at random and then trains
    % it best it can, or whether each node tries all weak learners and
    % takes the one that does best. Empirically, it appears leaving this as
    % true (default) gives slightly better looking results.
    %
    opts.debug=0;
    if isdeployed
      opts.debug=0;
    end
    numTrees= 100;
    verbose= true;
    priorMethod='varSel';
    if ~iscell(X) % from now on, X will be a cell array, with each cell containing one dataset
      X={X};
      Y={Y};
    end
    assert(length(X)==length(Y));

    if nargin < 3, opts= struct; end
    if ~isfield(opts,'samplingProb'), D=size(X{1},2); samplingProb=zeros(1,D); % will lead to uniform prior at each stage
    else samplingProb=opts.samplingProb; end
    samplingProb=(samplingProb+eps)/(max(samplingProb(:))+eps);

    if isfield(opts, 'numTrees'), numTrees= opts.numTrees; end
    numTrees=max(numTrees,1); % preventing empty forests
    if isfield(opts, 'verbose'), verbose= opts.verbose; end
    if isfield(opts, 'priorMethod'), priorMethod= opts.priorMethod; end
    treePrior=struct('samplingProb', samplingProb, 'priorMethod', priorMethod);
    treeModels= cell(1, numTrees);
    t0=tic;
    for i=1:numTrees
        if opts.debug>=3
          if i==2
            abort;
          end
        end

        % print info if verbose
        if verbose
            p10= floor(numTrees/10);
            if mod(i, p10)==0 || i==1 || i==numTrees
                fprintf('Training tree %d/%d (%fs)...\n', i, numTrees, toc(t0));
            end
        end

        % TODO? have a bagging option i.e. each tree is trained on a randomly chosen subset of data
        treeModels{i} = treeTrain(X, Y, opts, treePrior);
    end

    model.treeModels = treeModels;
end
