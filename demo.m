system('wget http://vision.cs.utexas.edu/projects/unreliableAttr/animals.mat');
zeroshot('pretrain_data', './animals.mat', 'RFtrees', 1, 'RFsplitsPerVar', 7, 'RFvarsPerNode', 10, 'RFdepth', 9, 'flipFrac', 0.15)
