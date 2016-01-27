function [Yhard, dummy, Ysoft] = multiForestTest(dummyY, X, model, dummyOpts)
  % casts forestTest into the predFoo interface required by classifyData
  dummy=0;
  Yh=[];
  Ys=[];
  for i=1:length(model.RF)
    [tmph, tmps] = forestTest(model.RF(i), X, dummyOpts);
    Yh=[Yh tmph];
    posClsInd=(model.RF(i).treeModels{1}.classes==1);
    Ys=[Ys tmps(:,posClsInd)];
  end
  Yhard=max(Yh,[],2); % if any entry is 1.
  Ysoft=1-prod(1-Ys,2);
end
