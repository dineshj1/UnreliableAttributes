function h=drawFigure(varargin)
  if isdeployed
    fprintf('Making invisible figure\n');
    h=figure(varargin{:}, 'Visible', 'off');
  else
    h=figure(varargin{:});
  end
end
