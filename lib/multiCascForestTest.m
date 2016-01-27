function [Yhard, dummy, Ysoft] = multiCascForestTest(dummyY, X, model, dummyOpts)
  % casts forestTest into the predFoo interface required by classifyData
  dummy=0;
  Yh=[];
  Ys=[];
  for i=1:length(model.cascForest)
    [tmph, tmps] = cascForestTest(model.cascForest(i), X, dummyOpts);
    Yh=[Yh tmph];
    posClsInd= (model.cascForest(i).cascModels{1}.classes==1);
    Ys=[Ys tmps(:,posClsInd)];
  end
  Yhard=max(Yh,[],2);
  Ysoft=1-prod(1-Ys,2);
end
