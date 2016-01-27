function [Yhard, dummy, Ysoft] = cascForestTest_wrap(dummyY, X, model, dummyOpts)
  % casts forestTest into the predFoo interface required by classifyData
  dummy=0;
  [Yhard, Ysoft] = cascForestTest(model, X, dummyOpts);

  posClsInd= (model.cascModels{1}.classes==1);
  Ysoft=Ysoft(:,posClsInd);
end
