function [Fscore Prec Rec TP TN FP FN] = Fmeasure(prediction, truth, k)

  % Check that predictions and truth have the same length
  assert(length(prediction)==length(truth));

  if nargin<3, k=1; end

  numTests = length(truth);
  TP = sum(prediction==truth & prediction==1);
  FP = sum(prediction~=truth & prediction==1);
  TN = sum(prediction==truth & prediction==0);
  FN = sum(prediction~=truth & prediction==0);
  
  assert(TP+FP+TN+FN==numTests);
  assert(TP+FP == sum(prediction==1));
  assert(TP+FN == sum(truth==1));

  if TP~=0
    Prec = TP/(TP+FP);
    Rec = TP/(TP+FN);
    Fscore = (1+k.^2).*Prec.*Rec./((k.^2).*Prec+Rec);
  else
    Prec=1e-5;
    Rec=1e-5;
    Fscore = 1e-5;
  end
end
