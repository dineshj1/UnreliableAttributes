function [Yhard, dummy, Ysoft] = forestTest_wrap(dummyY, X, model, dummyOpts)
  % casts forestTest into the predFoo interface required by classifyData
  dummy=0;
  [Yhard, Ysoft] = forestTest(model, X, dummyOpts);

  [clsList, ord]=sort(model.treeModels{1}.classes,'ascend');
  Ysoft=Ysoft(:,ord);

  if length(clsList)<=2 % return only positive class probability (convention)
    Ysoft=Ysoft(:,end);% last column is the positive class
  end

  %posClsInd=(model.treeModels{1}.classes==1);
  %Ysoft=Ysoft(:,posClsInd);
end
