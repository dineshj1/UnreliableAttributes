function bins=scoresToBins(scores, opts)
  if ~isfield(opts, 'numbins')
    opts.numbins=2;
  end
  if ~isfield(opts, 'range')
    opts.range=[0,1];
  end
  diff=opts.range(2)-opts.range(1);
  %bin_ul=(range(1)+diff/numbins):(diff/numbins):(range(1)-diff/numbins);
  bins=ceil(max(scores-opts.range(1),eps)*opts.numbins/diff);

  bins=max(bins,1);
  bins=min(bins,opts.numbins);
end
