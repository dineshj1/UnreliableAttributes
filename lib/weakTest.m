function yhat = weakTest(model, X, opts, FpTp)
% this version works only for binary data
% X is NxD as usual.
% see weakTrain for more info.
% yhat should be changed to be the probability with which each passed data point is less than threshold

%5;rand(1)

% FpTp is a 1 by 2 matrix which encodes such that
% FpTp(1,1)=probability that detection is 1 when truly 0 i.e FPR
% FpTp(1,2)=probability that detection is 1 when truly 1 i.e TPR
% Note that FpTp must refer to the classifier for dimension # model.r and threshold #model.t, which is a little weird
useFpTp=false;
if isfield(opts, 'useFpTp')
  useFpTp=opts.useFpTp;
end

if nargin < 3, opts = struct; end
if nargin < 4,
  FpTp=[0 1]; % default - perfect predictions
end

[N, D]= size(X);

switch model.classifierID
  case 1
    if useFpTp
      bins=scoresToBins(X(:,model.r), model.quantOpts);
      yhat=1-FpTp(bins)';
    else
      yhat= X(:,model.r)<model.t;
    end
  case 0
    %no classifier was fit (stopping criterion reached)
    yhat= zeros(N, 1); % effectively stopping the branch at the node where the stopping criterion was reached.
    %yhat= double(rand(N, 1));
    %yhat= double(rand(N, 1)<0.5);
  otherwise
    fprintf('Error in weak test! Classifier with ID = %d does not exist.\n', classifierID);
end

yhat(yhat<eps)=0;
yhat(yhat>1-eps)=1;

end
