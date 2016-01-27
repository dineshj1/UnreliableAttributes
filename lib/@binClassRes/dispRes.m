function obj=dispRes(obj,varargin)
  if isempty(varargin)
      varargin='all';
  end
  switch varargin{1}
    case 'any'
      varNames=varargin(2:end);
    case 'fixOP'
      varNames={'acc','Fscore','Prec','Rec','TP','TN','FP','FN'};
    case 'varOP'
      varNames={'AP','AUC'};
    case 'all'
      varNames={'acc','Fscore','Prec','Rec','TP','TN','FP','FN','AP','AUC','TPR','FPR','TNR','FNR'};
    otherwise
      error('Unknown option');
  end

  for i=1:length(varNames)
    %if ismember(varNames{i},{'TPR','FPR','TNR','FNR'}) % not stored currently but computed for display here anyway
    %  value=[];
    %  try
    %    switch varNames{i}
    %      case 'TPR'
    %        value=obj.TP/(obj.TP+obj.FN);
    %      case 'FPR'
    %        value=obj.FP/(obj.FP+obj.TN);
    %      case 'TNR'
    %        value=obj.TN/(obj.FP+obj.TN);
    %      case 'FNR'
    %        value=obj.FN/(obj.TP+obj.FN);
    %      otherwise
    %    end
    %  catch
    %    disp(getReport(err));
    %  end
    %  if ~isempty(value)
    %    fprintf('\t%s:%d\n',varNames{i}, value);
    %  end
    %else
    try
      value=eval(sprintf('obj.%s',varNames{i}));
      if ~isempty(value)
        fprintf('\t%s:%d\n',varNames{i},value);
      end
    catch err
      disp(getReport(err));
    end
  end

  if ~isempty(obj.posFrac)
    fprintf('DATA (+:%g,-:%g).\n', obj.posFrac,1-obj.posFrac);
  else
    fprintf('DATA unknown\n');
  end
end  
