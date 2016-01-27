function res=evalbinPreds(target,pred,conf)

  if nargin<3
    conf=pred;
  end

  res=binClassRes;
  if size(target,1)>1 % need to check, especially for the hard data
    % Accuracy
    res.acc=sum(pred==target)/length(pred)*100;
    % Fscore
    [res.Fscore res.Prec res.Rec res.TP res.TN res.FP res.FN]=Fmeasure(pred, target);
    % Precision-Recall and ROC curves
    lTe=target;

    res.numSamples=length(lTe);
    res.posFrac=sum(lTe==1)/length(lTe);
    skew=res.posFrac;
    try
      [res.misc.reca, res.misc.prec,~,res.AP]=perfcurve(lTe,conf,1, 'xCrit', 'reca', 'yCrit', 'prec');
      res.minAP=1+(1-skew)*log(1-skew)/skew;
      res.randAP=skew;
      res.normAP=(res.AP-res.minAP)./(res.randAP-res.minAP);

      [res.misc.fpr,res.misc.tpr,~,res.AUC]=perfcurve(lTe,conf,1, 'xCrit', 'FPR', 'yCrit', 'TPR');
    catch err
      if length(unique(lTe))<2
        [res.misc.reca, res.misc.prec, res.misc.fpr,res.misc.tpr]=deal(0);
        res.AP=0.5;
        res.AUC=0.5;
        clear lTe;
      else
        getReport(err)
        fprintf('Unknown error!');
        abort
      end
    end
  else
    [res.acc res.Fscore res.Prec res.Rec res.TP res.TN res.FP res.FN res.misc.reca res.misc.prec res.misc.fpr res.misc.tpr res.AP res.AUC] = deal(0);
  end

  res.pred=pred;
  res.conf=conf;
  res.setRes();% setting dependent variables such as TPR, FPR etc.
end
