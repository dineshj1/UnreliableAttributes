classdef data< handle
  properties (SetAccess=public, GetAccess=public)
    X; % usually base features
    Y; % usually target variables 
    Z; % to allow for other target variables. e.g. Y could be attributes and Z could be classes

    % % TODO type of data
    % Ytype; % classification/regression etc.
    % for Z too?
    
  end
  properties (SetAccess=private, GetAccess=public)
    numInstances;
    lenX; lenY; lenZ; % number of columns in each
  end
  methods (Access=private)
    this=setDependentVars(this);
  end
  methods (Access=public)
    % constructor, for dynamic use esp. 
    function this=data(x,y,z)
      if nargin~=0
        this.X=x;
        this.Y=y;
      end
      if nargin>2
        this.Z=z;
      end
      this.setDependentVars();
      assert(this.check(), 'Invalid data settings');
    end
    
    % check consistency
    val=check(this);

    % set data
    this=setData(varargin);

    % print summary of data
    dispSummary(this);
  end
end 
