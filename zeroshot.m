function zeroshot(varargin)
  % Will test order discovery in attribute+class learning
  rng('shuffle');
  try
    close all
    addpath(genpath('./lib/'));
    params=parseArgs(varargin);
    %% Learning on top of basic classifiers
    if 1
       if 1 % load file
         if isempty(params.pretrain_data)
           if ~exist('numNews','var')
             numNews=params.numNews;
           end
           matfilename=sprintf('%s/%s_set%d_%d_%d_CP0%s.mat',params.OPfolder,params.filenameHeader, params.subSplitNo, params.cluster, params.process, repmat('(new)',1,numNews));
         else
           matfilename=params.pretrain_data;
         end
         makeLine(sprintf('CP1: Loading from %s', matfilename),'|',100);
         origParams=params; % backup
         load(matfilename);
         params=origParams; clear('origParams');% clearing to avoid getting saved
       end

      %restrict all relevant variables to only use *valid* attributes i.e. attributes that have (enough) pos and neg samples in training, testing AND validation
      IndSetNames={'trainingInd','valInd','testingInd'};
      numAttr=size(class_attrib_mat.mean,2);
      numClasses=size(class_attrib_mat.mean,1);
      valAttr=true(1,numAttr);
      for setno=1:length(IndSetNames)
        ind=eval(sprintf('%s',IndSetNames{setno}));
        if isempty(ind), continue; end
        numPos=sum(attributematrix(ind,1:numAttr));
        numNeg=sum(~attributematrix(ind,1:numAttr));
        assert(all(numPos+numNeg==length(ind)), 'Sanity check failure');
        valAttr= valAttr & numPos>0 & numNeg>0;
      end
      attrPred(:,~valAttr)=[];
      attrConf(:,~valAttr)=[];
      val_featRes(~valAttr)=[];
      attributematrix(:,~valAttr)=[];
      %attr_perclass(:,~valAttr)=[];
      %attr_perimage(:,~valAttr)=[];
      cont_attributematrix(:,~valAttr)=[];
      %attributes(~valAttr)=[];
      %concepts(~valAttr)=[];
      class_attrib_mat.mean(:,~valAttr)=[];
      class_attrib_mat.annot(:,~valAttr)=[];
      %class_attrib_mat.std(:,~valAttr)=[];

      numAttr=size(attributematrix,2);
      basicConcepts=1:numAttr;

  %figure, bar([featRes.AUC]);
  %disp(mean([featRes.AUC]));

  clear origParams

      switch params.contClsAttMat
        case true
          conceptmatrix=cont_attributematrix;
        otherwise
          conceptmatrix=attributematrix;
      end


      try % assign featKnownScore
        if params.useConf
          featKnownScore=attrConf;
          normalize_scores=true;
          if normalize_scores
            ul=max(featKnownScore);
            ll=min(featKnownScore);
            range=ul-ll;
            numSamples=size(featKnownScore,1);
            tmp=ones(1,numSamples);
            featKnownScore=(featKnownScore-ll(tmp,:))./range(tmp,:);
          end
        else
          featKnownScore=attrPred;
        end
      catch
        warning('Could not assign featKnownScore correctly.');
      end

      % fixing name to save to
      matfilename=sprintf('%s/%s_set%d_%d_%d_CP0',params.OPfolder,params.filenameHeader, params.subSplitNo, params.cluster, params.process);
      numNews=0;
      if ~params.overWrite
        while exist([matfilename '.mat'],'var')
          matfilename=[matfilename '(new)'];
          numNews=numNews+1;
        end
      end
      matfilename=[matfilename '.mat'];

  %% set aside some data from test classes to be used as training data  (few-shot)
      if params.repeatable, rng(7732621); end
      ord=randperm(numel(testingInd));
      rng('shuffle');
      setAsideFrac=0.1; % imposes a limit of ~600 data points on the size of the few-shot training set. More than enough for 10 classes.
      ord=ord(1:setAsideFrac*numel(testingInd));
      setAside=testingInd(ord);
      testingInd=setdiff(testingInd,setAside);
      fsInd=setAside(1:params.numShots); % limiting the portion of set-aside data that is used in few-shot setting


      %% test zero-shot recognition with DAP
      loop_methodList=intersect({'dap'}, params.selMethodNames);
      if ~isempty(loop_methodList)
        for methodNo=1:length(loop_methodList)
          methodName=loop_methodList{methodNo};

          makeLine(methodName);
          switch methodName
            case 'dap'
              DAPfunc=@DAP_logsum;
              clsPrior=ones(1,numClasses)/numClasses;
              attrPrior=mean(class_attrib_mat.annot(unique(classes(trainingInd)),:));
              attrPrior(attrPrior<eps)=0.5;
              attrPrior(attrPrior>1-eps)=0.5;
            otherwise
              error('Unsupported method name!');
          end

          %confmat=cell2mat(arrayfun(@(x) getfield(x, 'conf'), featRes, 'UniformOutput', false));
          confmat=attrConf(testingInd,:);
          numTestCls=length(unique(classes(testingInd)));
          [zsRes, zsMultiRes] = DAPfunc(struct('attributes',confmat,'class',classes(testingInd)), (class_attrib_mat.annot), attrPrior, clsPrior);

          fprintf('\t# of testclasses: %d\n', numTestCls);
          fprintf('\tMean AP: %f\n', mean([zsRes.AP]));
          fprintf('\tMean AUC: %f\n', mean([zsRes.AUC]));
          fprintf('\tMean Fscore: %f\n', mean([zsRes.Fscore]));
          fprintf('\tMean normalized AP: %f\n', mean([zsRes.normAP]));
          disp(zsMultiRes.overallAcc);

          eval(sprintf('%sRes=zsRes;', methodName));
          eval(sprintf('%sMultiRes=zsMultiRes;', methodName));
          clear('zsRes', 'zsMultiRes');

          if params.saveCP
              varList={sprintf('%sRes', methodName), sprintf('%sMultiRes', methodName)};
              if exist(matfilename)
                varList{length(varList)+1}='-append';
              end
              makeLine(sprintf('Saving to %s', matfilename),'|',100);
              save(matfilename, varList{:});
          end
        end
     end

      loop_methodList=intersect({'felix'}, params.selMethodNames);
      if ~isempty(loop_methodList)
        for methodNo=1:length(loop_methodList)
          methodName=loop_methodList{methodNo};

          makeLine(methodName);
          switch methodName
            case 'felix'
            otherwise
              error('Unsupported method name!');
          end

          %confmat=cell2mat(arrayfun(@(x) getfield(x, 'conf'), featRes, 'UniformOutput', false));% confidences
          confmat=attrConf(testingInd,:);% confidences
          confmat=2*confmat-1;
          testClasses=unique(classes(testingInd));
          A_test=2*class_attrib_mat.mean(testClasses,:)-1;
          %A_test=double(2*class_attrib_mat.mean(testClasses,:)-1>0.5);
          for i = 1:size(A_test,1)
              A_test(i,:) = A_test(i,:)./norm(A_test(i,:),2);
          end

          [predict_class_label, error_att] = att_decoding(confmat, A_test, testClasses, classes(testingInd), 'loss');
          %[zsRes, zsMultiRes] = DAPfunc(struct('attributes',confmat,'class',classes(testingInd)), (class_attrib_mat.annot), attrPrior, clsPrior);
          zsRes(length(testClasses))=binClassRes;
          for i=1:length(testClasses)
            zsRes(i).AP=0;
            zsRes(i).AUC=0;
            zsRes(i).Fscore=0;
          end

          zsMultiRes.overallAcc=sum(predict_class_label'==classes(testingInd))/length(predict_class_label)*100;
          zsMultiRes.confusion=[];
          %fprintf('\tMean AP: %f\n', mean([zsRes.AP]));
          %fprintf('\tMean AUC: %f\n', mean([zsRes.AUC]));
          %fprintf('\tMean Fscore: %f\n', mean([zsRes.Fscore]));
          %fprintf('\tMean normalized AP: %f\n', mean([zsRes.normAP]));
          disp(zsMultiRes.overallAcc);

          eval(sprintf('%sRes=zsRes;', methodName));
          eval(sprintf('%sMultiRes=zsMultiRes;', methodName));
          clear('zsRes', 'zsMultiRes');

          if params.saveCP
              varList={sprintf('%sRes', methodName), sprintf('%sMultiRes', methodName)};
              if exist(matfilename)
                varList{length(varList)+1}='-append';
              end
              makeLine(sprintf('Saving to %s', matfilename),'|',100);
              save(matfilename, varList{:});
          end
        end
      end


    %% test zero-shot recognition with decision tree-based methods

    if params.Dep.RF_train_args.depth==0 % code to automatically set depth
      numValInd=length(valInd);
      optDepth_est=round(log(numValInd)/log(2)-6);
      optDepth_est=max(optDepth_est,3);
      params.Dep.RF_train_args.depth=optDepth_est;
      fprintf('Will use trees of depth: %d\n', optDepth_est);
    end
    loop_methodList=intersect({'zsDT','zsRF', 'zsRFwSampling', 'zsRFwCK','zsRFwValProp', 'zsRFwCKnROC', 'zsRFwValPropnROC', 'fsRF','fsRFwCK','fsRFwValProp','fsRFwCKnROC','fsRFwValPropnROC','fsRFwoPrior','zsRFwValPropnROC_noUncert','fsRFwValPropnROC_noUncert'}, params.selMethodNames);
    if ~isempty(loop_methodList)
      for methodNo=1:length(loop_methodList)
        methodName=loop_methodList{methodNo};

        testClasses=unique(classes(testingInd));
        numSamples=500*length(testClasses);
        clsPrior=ones(1,length(testClasses))/length(testClasses);

        Xtr=zeros(numSamples,numAttr); clsVec=zeros(numSamples,1);
        Ytr=cell(1,numAttr);
        count=0;
        for i=1:length(testClasses)
          numClsSamples=clsPrior(i)*numSamples;
          relInd=count+(1:numClsSamples);
          switch params.contClsAttMat
            case true
              Xtr(relInd,:)=repmat(class_attrib_mat.mean(testClasses(i),:), numClsSamples, 1);
            otherwise
             Xtr(relInd,:)=repmat(class_attrib_mat.annot(testClasses(i),:), numClsSamples, 1);
          end
          clsVec(relInd)=i*ones(numClsSamples,1); % numbering test classes separately for this task
          count=count+numClsSamples;
          Ytr{i}=double(clsVec==i);
        end

        Xtr_old=Xtr;
        % flip some bits of training data - signature uncertainty modeling
        for i=1:numAttr
          % flipping positive bits
          tmp=find(Xtr(:,i)==1); % instances with positive class-level annotation
          if params.repeatable, rng(12551); end
          tmp=randperm(length(tmp)); % jumble
          rng('shuffle');
          tmp=tmp(1:params.flipFrac*length(tmp)); % randomly select annotations to flip
          Xtr(tmp,i)=~Xtr(tmp,i); % flip to better represent instance-level (true) labels

          % flipping negative bits
          tmp=find(Xtr(:,i)==0); % instances with positive class-level annotation
          tmp=randperm(length(tmp)); % jumble
          tmp=tmp(1:0.0*length(tmp)); % select a fraction to flip
          Xtr(tmp,i)=~Xtr(tmp,i); % flip to better represent instance-level (true) labels
        end


        makeLine(methodName);
        fewShots=false;
        switch methodName
          case 'zsDT'
            args=params.Dep.RF_train_args;
            args.numTrees=1; % training single tree
            args.depth=10;
            args.numSplitsPerVar=1; % since binary attributes
            args.numVarsPerNode=300; % pretty much guaranteed to give optimal results
          case {'fsRFwoPrior'} % plain random forest using pre-trained attributes as "classeme"-like features
            args=params.Dep.RF_train_args;
            args.numSplitsPerVar=1; % since binary attributes
            args.srcDsWt=0;
            Xtr=[];
            for i=1:length(testClasses)
              Ytr{i}=[];
            end
            fewShots=true;
          case {'zsRF','fsRF'}
            args=params.Dep.RF_train_args;
            args.numbins=2; % meaningless otherwise
            args.numSplitsPerVar=1; % since binary attributes
            if strcmp(methodName,'fsRF')
              fewShots=true;
            end
          case {'zsRFwSampling','fsRFwSampling'}
            args=params.Dep.RF_train_args;
            args.numSplitsPerVar=1; % since binary attributes
            args.samplingProb=[val_featRes.AUC];
            args.numbins=2;
            if strcmp(methodName,'fsRFwSampling')
              fewShots=true;
            end
          case {'zsRFwCK','fsRFwCK'}
            args=params.Dep.RF_train_args;
            args.numSplitsPerVar=1; % since binary attributes
            tpr=[val_featRes.TPR];
            fpr=[val_featRes.FPR];
            tpr(isnan(tpr))=0.5;
            fpr(isnan(fpr))=0.5;
            for attrno=1:numAttr
              args.classifierKnowledge{attrno}=[0,0,0; fpr(attrno), tpr(attrno),0.5;fpr(attrno),tpr(attrno),1];
            end
            if strcmp(methodName,'fsRFwCK')
              fewShots=true;
            end
          case {'zsRFwValProp','fsRFwValProp'}
            args=params.Dep.RF_train_args;
            args.numSplitsPerVar=1; % since binary attributes
            args.valProp=true;
            args.valData.X=attrConf(valInd,basicConcepts);
            args.valData.annot=conceptmatrix(valInd,basicConcepts);
            if strcmp(methodName,'fsRFwValProp')
              fewShots=true;
            end
          case {'zsRFwCKnROC','fsRFwCKnROC'}
            args=params.Dep.RF_train_args;
            args.ROC=true;
            %args.classifierKnowledge=[0*ones(numAttr,1) 1*ones(numAttr,1)];% perfect classifiers
            %args.classifierKnowledge=[0.5*ones(numAttr,1) 0.5*ones(numAttr,1)];% random classifiers
            %args.classifierKnowledge=[0.2*ones(numAttr,1) 0.8*ones(numAttr,1)];% with some uncertainty - great results in terms of AP and AUC, but F-score drops. (BUG)

              for attrno=1:numAttr
                fprintf('Extracting classifier knowledge for attrno %d\n', attrno);
                try
                  [tmp1, tmp2, tmp3]=perfcurve(conceptmatrix(valInd,attrno), attrConf(valInd,attrno), 1, 'xCrit', 'FPR', 'yCrit', 'TPR');
                catch err
                  if strcmp(err.identifier,'stats:perfcurve:NotEnoughClasses')
                    args.classifierKnowledge{attrno}=[0 1 1];
                  else
                    rethrow(err);
                  end
                end
                args.classifierKnowledge{attrno}=[tmp1 tmp2 tmp3];
              end
            clear('tmp1','tmp2','tmp3');
            %tpr(isnan(tpr))=0.5;
            %fpr(isnan(fpr))=0.5;
            %args.classifierKnowledge=abort;
            if strcmp(methodName,'fsRFwCKnROC')
              fewShots=true;
            end
          case {'zsRFwValPropnROC','fsRFwValPropnROC'}
            args=params.Dep.RF_train_args;
            args.valProp=true;
            args.ROC=true;
            args.valData.X=attrConf(valInd,basicConcepts);
            args.valData.annot=conceptmatrix(valInd,basicConcepts);
            %args.valData.annot=attr_perimage(valInd,basicConcepts); % should get instance level labels
            %args.valData.annot=attr_perclass(valInd,basicConcepts); % should get instance level labels
            if strcmp(methodName,'fsRFwValPropnROC')
              fewShots=true;
            end
          case {'zsRFwValPropnROC_noUncert','fsRFwValPropnROC_noUncert'}
            args=params.Dep.RF_train_args;
            args.valProp=true;
            args.ROC=true;
            args.noUncert=true;
            args.valData.annot=conceptmatrix(testingInd,basicConcepts);
            args.valData.X=attrConf(valInd,basicConcepts);
            %args.valData.annot=attr_perclass(valInd,basicConcepts); % should get class level labels
            if strcmp(methodName,'fsRFwValPropnROC')
              fewShots=true;
            end
          otherwise
            error('Unknown methodName');
        end

        % Training decision trees with Xtr and clsVec
        zsRes(length(testClasses))=binClassRes;
        if ~isfield(args, 'noUncert')
          args.noUncert=false;
        end
        if args.noUncert
           Xtr=Xtr_old; % return to state before flipping
        end

        if fewShots, Xtr={attrConf(fsInd, basicConcepts), Xtr}; end
        if ~isfield(args, 'ROC')
          args.ROC=false;
        end
        if params.repeatable, rng(3562462); end
        for i=1:length(testClasses)
          fprintf('Learning %s (%d/%d)\n', classnames{testClasses(i)}, i, length(testClasses));
          if fewShots
            Ytr{i}={double(classes(fsInd)==testClasses(i)), Ytr{i}};
          end

          zsModel(i)=forestTrain(Xtr, Ytr{i}, args);
          if args.ROC
            Xts=attrConf(testingInd,basicConcepts);
          else
            Xts=attrPred(testingInd,basicConcepts);
          end
          YtsMat=double(classes(testingInd)==testClasses(i));
          zsRes(i) = classifyData(zsModel(i), data(Xts,YtsMat), 1, @forestTest_wrap);
        end
        rng('shuffle');

        fprintf('\tMean AP: %f\n', mean([zsRes.AP]));
        fprintf('\tMean AUC: %f\n', mean([zsRes.AUC]));
        fprintf('\tMean Fscore: %f\n', mean([zsRes.Fscore]));
        eval(sprintf('%sModel=zsModel;%sRes=zsRes;', methodName, methodName));

        % accounting for label exclusivity
        methodNameX=[methodName '_X'];
        params.selMethodNames=union(params.selMethodNames, methodNameX);
        % evaluate with the mutual exclusivity constraint accounted for (similar to DAP)
        Confidence= cell2mat(arrayfun(@(x) x.conf, zsRes, 'UniformOutput', false));
        % normalize rows of Confidence matrix
        rowsum = sum(Confidence,2);
        Confidence = bsxfun(@rdivide, Confidence, rowsum);
        [~,~,newClsVec]=unique(classes(testingInd));
        [zsxRes, zsxMultiRes] =evalmultiPreds(Confidence, newClsVec);
        fprintf('After including exclusivity constraint:\n ===== \n');
        fprintf('\tMean AP: %f\n', mean([zsxRes.AP]));
        fprintf('\tMean AUC: %f\n', mean([zsxRes.AUC]));
        fprintf('\tMean Fscore: %f\n', mean([zsxRes.Fscore]));
        fprintf('\tOverall accuracy: %f\n', zsxMultiRes.overallAcc);

        eval(sprintf('%s_XRes=zsxRes;', methodName, methodName));
        eval(sprintf('%s_XMultiRes=zsxMultiRes;', methodName, methodName));
        eval(sprintf('%sMultiRes=zsxMultiRes;', methodName, methodName)); % multiRes is the same for both methods

        clear('zsModel','zsRes', 'zsxRes', 'zsxMultiRes');
        if params.saveCP
            varList={sprintf('%sRes', methodName), sprintf('%sMultiRes', methodName), sprintf('%sModel', methodName)};
            varList=union(varList, {sprintf('%s_XRes', methodName), sprintf('%s_XMultiRes', methodName)});
            if exist(matfilename)
              varList{length(varList)+1}='-append';
            end
            makeLine(sprintf('Saving to %s', matfilename),'|',100);
            save(matfilename, varList{:});
        end
        rng('shuffle');
      end
   end
  end
      %% Presenting results

      allMethodNames={params.selMethodNames{:}};
      %allMethodNames={params.selMethodNames{:}};
      tmp=strcat(allMethodNames, 'Res''');
      allRes=eval(['[' sprintf('%s ', tmp{:}) ']']);
      tmp=strcat(allMethodNames, 'MultiRes''');
      allMultiRes=eval(['[' sprintf('%s ', tmp{:}) ']']);
      testClasses=unique(classes(testingInd));

      APfig=drawFigure();
      set(APfig, 'Position', [2619 145 812 800]);
      suptitle(sprintf('Zero-shot object recognition'));
      perfMats={};
      for i=1:length(params.perfMeasure)
        subplot(length(params.perfMeasure),1,i);
        perfMeasure=params.perfMeasure{i};
        title(perfMeasure);
        switch perfMeasure
          case {'AP','AUC','Fscore'}
            scores=arrayfun(@(x) eval(sprintf('x.%s', perfMeasure)), allRes);
            xlabels=classnames(testClasses);
          case {'overallAcc'}
            scores=arrayfun(@(x) eval(sprintf('x.%s', perfMeasure)), allMultiRes);
            if isrow(scores), scores=scores'; end
            scores=[scores NaN(size(scores,1),1)]';
            xlabels={'Overall'};
          otherwise
            error('Unknown performance measure');
        end
        bar(scores);
        try
          xlim([0.5,length(xlabels)+0.5]);
          set(gca,'XTick', 1:length(xlabels),'XTickLabel', xlabels);
          rotateXLabels(gca, 90);
        catch err
          getReport(err)
        end

        ylabel(perfMeasure);
        LEG=strcat(allMethodNames', ':', cellstr(num2str(nanmean(scores,1)')));
        legend(LEG{:}, 'Location', 'NorthEastOutside');
        eval(sprintf('%sMat=scores'';', perfMeasure));
        perfMats{end+1}=sprintf('%sMat', perfMeasure);
      end


      % % precision-recall curves
      % testClasses=unique(classes(testingInd));
      % yaxis=arrayfun(@(x) eval(sprintf('x.misc.%s', 'prec')), allRes, 'UniformOutput', false); % stores in a cell array
      % xaxis=arrayfun(@(x) eval(sprintf('x.misc.%s', 'reca')), allRes, 'UniformOutput', false); % stores in a cell array
      % ap_mat=arrayfun(@(x) eval(sprintf('x.%s', 'AP')), allRes);
      % cc=hsv(size(yaxis,2)); % as many colors as there are methods
      % for i=1:size(yaxis,1) % length(testClasses)
      %   figure,
      %   for j=1:size(yaxis,2) % number of methods
      %     plot(xaxis{i,j},yaxis{i,j},'color',cc(j,:)); hold on;
      %   end
      %   LEG=strcat(allMethodNames', ':', cellstr(num2str(ap_mat(i,:)')));
      %   legend(LEG{:}, 'Location', 'NorthEastOutside');
      %   title(sprintf('Class:%s',classnames{testClasses(i)}));
      %   %pause
      % end

    % save final matfile
    if params.saveCP
      matfilename=sprintf('%s/%s_set%d_%d_%d%s.mat', params.OPfolder, params.filenameHeader, params.subSplitNo, params.cluster, params.process, repmat('(new)',1,params.numNews));
      save(matfilename);
    end

    %results alone (to allow to generate plots without storing *everything*)
    matfilename=sprintf('%s/res/RES_%s_set%d_%d_%d%s.mat', params.OPfolder, params.filenameHeader, params.subSplitNo, params.cluster, params.process, repmat('(new)',1,params.numNews));
    fprintf('Saving result variables to %s\n', matfilename);
    save(matfilename, perfMats{:}, 'perfMats', 'params');
    fprintf('Finished\n');
  catch err
    getReport(err)
  end
end


function params=parseArgs(args)
  params.flipFrac=0;
  params.startCP=0;
  params.figSave=true;
  params.figFormat='.png';
  params.OPfolder=pwd;
  params.titleFieldNames={'RFsplitsPerVar',  'RFvarsPerNode', 'RFdepth', 'RFtrees', 'RFleafFrac',};
  params.discoverStages=true; % can be disabled to only use one stage
  params.useConf=true;
  params.perfMeasure={'AP','AUC','Fscore','overallAcc'};
  params.overWrite=true;
  params.numNews=0;
  params.numSelComp=25; % number of composites selected per round
  defaultFileNameHeader=true;
  %params.selMethodNames={'feat','RF_DNF','RF_plain'};
  params.selMethodNames={'zsRF', 'zsRFwValPropnROC'};
  params.kernel='linear';
  params.contClsAttMat=false;
  params.pretrain_data='';
  params.moreData=+1;
  params.trackTrainPerf=true;
  params.RFpriorMethod='varSel';
  params.repeatable=true;
  params.variableSelection=true;
  params.perclassAttr=false;
  params.cluster=0;
  params.process=0;

  % default values for some dependent parameters
  params.Dep.selectedMethods=true;

  % parameters that can either be set or left to other parameter-dependent values
  defaultSaveCP=true;

  % default values of all list parameters - also used in matching parameters across matfiles
  params.List.RFdepth=7;
  params.List.RFtrees=10;
  params.List.RFvarsPerNode=5;
  params.List.RFsplitsPerVar=5;
  params.List.RFpriorFrac=1;
  params.List.RFleafFrac=0.05;
  %params.List.treeClassifier=1;
  %params.List.classifierCommitFirst=true;

  params.List.svm_c=10^(-4.5); % optimal for AwA_PCA
  %params.List.svm_c=1e-5; % optimal for SUN
  params.List.svm_d=3;
  params.List.svm_g=1/500;
  params.List.svm_r=0;

  params.List.PCA=1;
  params.List.TrnTstSplit=3;
  params.List.valFrac=0.1;
  params.List.srcDsWt=0;
  params.List.numbins=2;
  defaultTrnTstSplit=true;

  params.List.combineGens=false;
  params.List.subSplitNo=2; % allowing the same classes to exist in both training and test data, relevant only to AwA
  params.List.addClasses=true;
  params.List.allTrain=1;% using all seen class data as training data by default i.e. reserving data

  params.List.numShots=0; % over all classes



  numarg = length(args);
  if numarg>=2
    for i=1:2:numarg
      switch args{i}
        case 'perclassAttr'
          params.perclassAttr=convert2num(args{i+1});
        case 'contClsAttMat'
          params.contClsAttMat=convert2num(args{i+1});
        case 'selMethodNames'
          params.Dep.selectedMethods=true;
          %defaultSelMethodNames=false;
          params.selMethodNames=args{i+1};%to load a cell array of strings
          if ~iscell(params.selMethodNames)
            params.selMethodNames={params.selMethodNames};
          end
          if ismember('all', params.selMethodNames)
            params.Dep.selectedMethods=false;
            %defaultSelMethodNames=true;
          end
        case 'trackTrainPerf'
          params.trackTrainPerf=convert2num(args{i+1});
        case 'saveCP'
          params.saveCP=convert2num(args{i+1});
          defaultSaveCP=false;
        case 'pretrain_data'
          params.pretrain_data=args{i+1};
        case 'repeatable' % splits are always repeatable. Only the randomized methods such as RFs are controlled by this parameter
          params.repeatable=convert2num(args{i+1});
        case 'titleFieldNames'
           params.titleFieldNames=args{i+1};%to load a cell array of strings
       case 'figFormat'
          params.figSave=true;
          params.figFormat=args{i+1};
        case 'figSave'
          params.figSave=convert2num(args{i+1});
        case 'OPfolder'
          params.OPfolder=args{i+1};
        case 'useConf'
          params.useConf=convert2num(args{i+1});
        case 'overWrite'
          params.overWrite=convert2num(args{i+1});
        case 'numSelComp'
          params.numSelComp=convert2num(args{i+1});
        case 'numNews'
          params.numNews=convert2num(args{i+1});
        case 'discoverStages'
          params.discoverStages=convert2num(args{i+1});
        case 'filenameHeader'
          params.filenameHeader=args{i+1};
          if ~strcmpi(params.filenameHeader, 'default')
            defaultFileNameHeader=false;
          end
       case 'kernel'
          params.kernel=args{i+1};
        case 'moreData'
          params.moreData=convert2num(args{i+1});
        case 'RFpriorMethod'
          params.RFpriorMethod=args{i+1};
        case 'variableSelection'
          params.variableSelection=convert2num(args{i+1});
        case 'perfMeasure'
           params.perfMeasure=args{i+1};%to load a cell array of strings
      % all "List" parameters (meant to be easily varied over a list through condor)
        case {'PCA','PCAList'}
          params.List.PCA=convert2num(args{i+1});
        case {'clauseLength', 'clauseLengthList'}
          params.List.clauseLength=convert2num(args{i+1});
        case {'RFdepth', 'RFdepthList'}
          params.List.RFdepth=convert2num(args{i+1});
        case {'RFtrees', 'RFtreesList'}
          params.List.RFtrees=convert2num(args{i+1});
        case {'RFleafFrac'}
          params.List.RFleafFrac=convert2num(args{i+1});
        %case {'RFsplits', 'RFsplitsList'}
        %  params.List.RFsplits=convert2num(args{i+1});
        case {'RFvarsPerNode', 'RFvarsPerNodeList'}
           params.List.RFvarsPerNode=convert2num(args{i+1})
        case {'RFsplitsPerVar', 'RFsplitsPerVarList'}
          params.List.RFsplitsPerVar=convert2num(args{i+1});
        case {'RFpriorFrac', 'RFpriorFracList'}
          params.List.RFpriorFrac=convert2num(args{i+1});
        case {'svm_c', 'svm_cList'}
          params.List.svm_c=convert2num(args{i+1});
        case {'svm_d', 'svm_dList'}
          params.List.svm_d=convert2num(args{i+1});
        case {'svm_g', 'svm_gList'}
          params.List.svm_g=convert2num(args{i+1});
        case {'svm_r', 'svm_rList'}
          params.List.svm_r=convert2num(args{i+1});
        case {'addComposites', 'addCompositesList'}
          params.List.addComposites=convert2num(args{i+1});
        case {'combineGens', 'combineGensList'}
          params.List.combineGens=convert2num(args{i+1});
        case {'addClasses', 'addClassesList'}
          params.List.addClasses=convert2num(args{i+1});
        case 'TrnTstSplit'
          params.List.TrnTstSplit=convert2num(args{i+1});
          defaultTrnTstSplit=false;
        case 'valFrac'
          params.List.valFrac=convert2num(args{i+1});
        case {'subSplitNo', 'subSplitNoList'} % relevant only to AwA
          params.List.subSplitNo=convert2num(args{i+1});
        case {'numRevisions', 'numRevisionsList'}
          params.List.numRevisions=convert2num(args{i+1});
        case {'allTrain', 'allTrainList'}
          params.List.allTrain=convert2num(args{i+1});
        case 'flipFrac'
          params.List.flipFrac=convert2num(args{i+1});
        case 'srcDsWt' % weights on each dataset (for the RF_adapt method)
          params.List.srcDsWt=convert2num(args{i+1});
        case 'numbins'
          params.List.numbins=convert2num(args{i+1});
        case 'numShots' % weights on each dataset (for the RF_adapt method)
          params.List.numShots=convert2num(args{i+1});
        otherwise
          error(sprintf('invalid parameter name %s', args{i}));
      end
    end
  end
  mkdir(params.OPfolder);
  mkdir([params.OPfolder '/res/']);

  % selecting item from list
  fprintf('\nCombinations');
  paramNames=fieldnames(params.List);
  for i = 1:length(paramNames)
    tmp{i}=eval(sprintf('params.List.%s',paramNames{i}));
  end
  combinations = allcomb(tmp{:});
  disp([(1:size(combinations,1))' combinations]);
  fprintf('\n Selecting parameter combination #');
  params.Dep.index = params.process;
  fprintf('%d(+1) of %d\n\n', params.Dep.index, size(combinations,1));
  assert(params.Dep.index+1<=size(combinations,1) && params.Dep.index>=0);
  for i=1:length(paramNames)
    eval(sprintf('params.%s=combinations(params.Dep.index+1,%d);', paramNames{i}, i));
  end

  params.Dep.RF_train_args=struct(...
    'depth', params.RFdepth,...
    'numTrees', params.RFtrees,...
    'leafFrac', params.RFleafFrac,...
    'numVarsPerNode', params.RFvarsPerNode,...
    'numSplitsPerVar', params.RFsplitsPerVar,...
    'priorFrac', params.RFpriorFrac,...
    'priorMethod', params.RFpriorMethod,...
    'select', ~params.variableSelection,...
    'classifierID', 1,...
    'classifierCommitFirst', true, ...
    'dsWts', [1-params.srcDsWt, params.srcDsWt],...
    'numbins', params.numbins...
    );

  if defaultSaveCP
    params.saveCP=true;
  end

  if defaultFileNameHeader
    %params.filenameHeader=sprintf('%s(%s_TrTs%d_subSpl%d_allTr%d_moreData%d)',mfilename, params.Dep.datasetName, params.TrnTstSplit, params.subSplitNo,params.allTrain,params.moreData);
    params.filenameHeader='trial';
  end

end

function arg = convert2num(arg)
  arg=arg;
end
