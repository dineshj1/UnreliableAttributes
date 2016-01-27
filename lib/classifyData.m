function [res, overallRes] = classifyData(model, Data, targetCol, predfoo)
  % res is a binClassRes instance
  %res=binClassRes;
  overallRes=[];

  %NOTE: any changes here might also need to be copied into jdCrossValidate.m
  if nargin<4
    predfoo=@liblin_predict_wrap; % other predfoos must also have the same interface as liblin_predict to be compatible
  end


  % making inputFormat argument unnecessary (because we can just look at the function itself) - to remove once we have corrected all calling instances (mainly in stagedLearning.m)
  args=[];
  [pred, ~, conf]=predfoo(Data.Y(:,targetCol), Data.X, model, args);

  [clsNames, ~, newClsVec]=unique(Data.Y(:,targetCol)); % also sorts in ascending
  numCls=length(clsNames);
  if numCls<=2
    res=evalbinPreds(Data.Y(:,targetCol), pred, conf);
  else
    [res, overallRes]=evalMultiClassPreds(conf, newClsVec);
  end
end

