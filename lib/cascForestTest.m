function [Yhard, Ysoft] = cascForestTest(model, X, opts)
    % X is NxD, where rows are data points
    % model comes from forestTrain()
    % Yhard are hard assignments to X's, Ysoft is NxK array of
    % probabilities, where there are K classes.

    if nargin<3, opts= struct; end

    numCascs= length(model.cascModels);
    u= model.cascModels{1}.classes; % Assume we have at least one casc!
    Ysoft= zeros(size(X,1), length(u));
    for i=1:numCascs
        [~, ysoft] = cascTest(model.cascModels{i}, X, opts);
        Ysoft= Ysoft + ysoft; %TODO is there a better way to aggregate the ensemble scores?
    end

    Ysoft = Ysoft/numCascs;
    [~, ix]= max(Ysoft, [], 2);
    Yhard = u(ix);
end
