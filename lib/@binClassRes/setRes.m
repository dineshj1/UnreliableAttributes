function obj=setRes(obj,varargin)
%   if length(varargin)<1
%     warning('No members set.');
%     return;
%   end
  if ~isempty(varargin)
    switch varargin{1}
      case 'any'
        varNames=varargin(2:2:end);
        varVals=varargin(3:2:end);
      case 'fixOP'
        varNames={'acc','Fscore','Prec','Rec','TP','TN','FP','FN'};
        assert(length(varargin)-1<=length(varNames));
        varVals=varargin(2:end);
      case 'varOP'
        varNames={'AUC','AP'};
        assert(length(varargin)-1<=length(varNames));
        varVals=varargin(2:end);
      case 'outputs'
        varNames={'pred','conf'};
        assert(length(varargin)-1<=length(varNames));
        varVals=varargin(2:end);
      otherwise
        error('Unknown option');
    end

    assert(length(varVals)<=length(varNames),'Too many input values');
    for i=1:length(varVals)
      try
        eval(sprintf('obj.%s=%f;',varNames{i},varVals{i}));
      catch err
        disp(getReport(err));
      end
    end
  end

  if ~any([isempty(obj.TP), isempty(obj.FP), isempty(obj.TN), isempty(obj.FN)])
    obj.numSamples=obj.TP+obj.FP+obj.TN+obj.FN; % make sure numSamples is consistent
    obj.posFrac=(obj.TP+obj.FN)/obj.numSamples;% make sure posFrac is consistent 
    obj.TPR=obj.TP/(obj.TP+obj.FN);
    obj.FNR=obj.FN/(obj.TP+obj.FN);
    obj.FPR=obj.FP/(obj.FP+obj.TN);
    obj.TNR=obj.TN/(obj.FP+obj.TN);
  end
end
