function dispSummary(this)
  fprintf('Dataset summary:');
  varNames='numInstances','lenX','lenY','lenZ';
  for i=1:length(varNames)
    fprintf('\t%s:%d\n',varNames{i},eval(sprintf('this.%s', varNames{i})));
  end
  % if type is known, can do other things such as display fraction of positives, or distribution over classes etc(TODO)
end
