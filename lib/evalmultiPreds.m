function [binRes, overallRes] =evalmultiPreds(Confidence, clsVec)
  assert(size(Confidence, 2)>1);
  [~, MAP_classPred]=max(Confidence,[],2);
  overall_acc=(sum(MAP_classPred==clsVec)/length(MAP_classPred))*100;
  [confmat,order]=confusionmat(clsVec, MAP_classPred);
  %confmat(i,j) is a count of observations known to be in group i but predicted to be in group j.

  %normalize columns
  Confusion.mat=confmat./repmat(sum(confmat,1),size(confmat,1),1);
  %normalize rows
  Confusion.mat=confmat./repmat(sum(confmat,2),1,size(confmat,1));
  Confusion.order=order;
  overallRes.overallAcc=overall_acc;
  overallRes.confusion=Confusion;

  % Class-wise AP performance
  %testingClasses=unique(clsVec);
  %numCls=length(testingClasses);
  %assert(size(Confidence, 2)>=numCls);
  assert(all(clsVec<=size(Confidence,2)));
  for i=1:size(Confidence,2)
    binRes(i)=evalbinPreds(double(clsVec == i), double (MAP_classPred == i), Confidence(:,i));
  end
end
