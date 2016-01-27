function model = cascForestTrain(X, Y, opts, targetSign)
    % X is NxD, where rows are data points
    % for convenience, for now we assume X is 0 mean unit variance. If it
    % isn't, preprocess your data with
    %
    % X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
    %
    % If this condition isn't satisfied, some weak learners won't work out
    % of the box in current implementation.
    %
    % Y is discrete Nx1 vector of labels
    % model can be plugged into forestTest()
    %
    % decent default opts are:
    % opts.depth= 9;
    % opts.numCascs= 100; %(but more is _ALWAYS_ better, monotonically, no exceptions)
    % opts.numSplits= 5;
    % opts.classifierID= 2
    % opts.classifierCommitFirst= true;
    %
    % which means use depth 9 trees, train 100 of them, use 5 random
    % splits when training each weak learner. The last option controls
    % whether each node commits to a weak learner at random and then trains
    % it best it can, or whether each node tries all weak learners and
    % takes the one that does best. Empirically, it appears leaving this as
    % true (default) gives slightly better looking results.
    %

    numCascs= 100;
    verbose= true;
    priorMethod='varSel';

    if nargin < 3, opts= struct; end
    if nargin < 4, D=size(X,2); targetSign=zeros(1,D); end % will lead to uniform prior at each stage
    if isfield(opts, 'numCascs'), numCascs= opts.numCascs;
    elseif isfield(opts, 'numTrees'), numCascs= opts.numTrees;
    end
    numCascs=max(numCascs,1);
    if isfield(opts, 'verbose'), verbose= opts.verbose; end
    if isfield(opts, 'priorMethod'), verbose= opts.priorMethod; end

    assert(isrow(targetSign), 'targetSign must be a row vector');

    %% select only top few entries from targetSign, make the rest equal to 0
    %[~,ord]=sort(abs(targetSign), 'descend');
    %tmp=zeros(1,length(targetSign)); tmp(ord(1:opts.depth))=1;
    %targetSign=targetSign.*tmp;

    targetSign=(targetSign+eps)/(max(targetSign(:)+eps));

    cascPrior=struct('targetSign', targetSign, 'priorMethod', priorMethod);
    cascModels= cell(1, numCascs);
    for i=1:numCascs
        % TODO? have a bagging option i.e. each cascade is trained on a randomly chosen subset of data
        cascModels{i} = cascTrain(X, Y, opts, cascPrior);

        % print info if verbose
        if verbose
            p10= floor(numCascs/10);
            if mod(i, p10)==0 || i==1 || i==numCascs
                fprintf('Training tree %d/%d...\n', i, numCascs);
            end
        end
    end

    model.cascModels = cascModels;
end
