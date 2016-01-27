function this=setData(this,varargin)
  if length(varargin)<2
    warning('No members set.');
    %return;
  end
  switch varargin{1}
    case 'spec' % specified variable names
      varNames=varargin(2:2:end);
      varVals=varargin(3:2:end);
    case 'fixOP'
      varNames={'X','Y','Z'};
      assert(length(varargin)-1<=length(varNames));
      varVals=varargin(2:end);
    otherwise
      error('Unknown option');
  end

  assert(length(varVals)<=length(varNames),'JD: Too many input values');
  for i=1:length(varVals)
    try
      eval(sprintf('obj.%s=%s;',varNames{i},'varVals{i}'));
    catch err
      disp(getReport(err));
    end
  end

  this.setDependentVars();

  % check for consistency
  assert(this.check(), 'Invalid data settings');
end  
