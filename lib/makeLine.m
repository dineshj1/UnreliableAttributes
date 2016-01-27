function makeLine(txt, character, numTimes, numWhiteLines)
  if nargin<1
      txt='';
  end
  if nargin<2
    character='=';
  end
  if nargin<3
    numTimes=50;
  end
  if nargin<4
    numWhiteLines=0;
  end
  fprintf('%s\n', txt);
  fprintf('%s\n', repmat(character, 1, numTimes));
  for i=1:numWhiteLines
    fprintf('\n');
  end
end
