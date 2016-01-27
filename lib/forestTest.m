function [Yhard, Ysoft] = forestTest(model, X, opts)
    % X is NxD, where rows are data points
    % model comes from forestTrain()
    % Yhard are hard assignments to X's, Ysoft is NxK array of
    % probabilities, where there are K classes.

    if nargin<3, opts= struct; end

    opts.debug=0;
    if isdeployed
      opts.debug=0;
    end
    if opts.debug
      disp(X(1,:))
    end

    numTrees= length(model.treeModels);
    u= model.treeModels{1}.classes; % Assume we have at least one tree!
    Ysoft= zeros(size(X,1), length(u));
    for i=1:numTrees
        if opts.debug>3
          if i==2
            abort;
          end
        end
        [~, ysoft] = treeTest(model.treeModels{i}, X, opts);
        Ysoft= Ysoft + ysoft; %TODO is there a better way to aggregate the ensemble scores?
        if opts.debug
          Ysoft
        end
    end

    Ysoft = Ysoft/numTrees;
    [~, ix]= max(Ysoft, [], 2);
    Yhard = u(ix);
end
