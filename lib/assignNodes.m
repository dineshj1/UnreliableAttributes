function X=assignNodes(distMat,priorFrac,select)
  warning off;
  % assigns nodes to clauses by solving a binary integer linear program using matlab's bintprog routine
  if nargin<2
    priorFrac=1;
  end
  if nargin<3
    select=true;
  end

  numNodes=size(distMat,1);
  numClauses=size(distMat,2);
  numSel=round(priorFrac*numNodes);

  if numSel==0 % just return all zeros
    X=false(numNodes,numClauses);
  elseif numClauses==1 % no need to solve the binary program. Just sort the nodes
    X=false(numNodes,1);
    if select, [~,ord]=sort(distMat, 'ascend');
    else ord=randperm(numNodes);
    end
    X(ord(1:numSel))=true;
  else
    if numSel<numNodes % first preselect the nodes among which to select for prioritization, then assign individual
      if select, [~,ord]=sort(sum(distMat,2), 'ascend'); % sorted by average distance to all clauses
      else ord=randperm(numNodes);
      end
      preselNodes=ord(1:numSel);
    else
      preselNodes=1:numNodes;
    end

    if select, distMat_trunc=distMat(preselNodes, :);
    else distMat_trunc=rand(numSel, numClauses); % random distances
    end

    numVars=numSel*numClauses;
    tmp=zeros(1,numSel); tmp(1)=1;
    vec=repmat(tmp, 1, numClauses); % first row
    assert(length(vec)==numVars);
    step=1; % how much to advance by at each row
    A=vec(mod(bsxfun(@plus,(0:-step:(-step*(numSel-1)))',0:1:numVars-1),numVars)+1); % inequality constraints: each node must be assigned to at most 1 clause
    b=max(ceil(numClauses/numSel),1)*ones(numSel,1);

    tmp=zeros(numClauses,1); tmp(1)=1;
    vec=repmat(tmp, 1, numSel)'; vec=vec(:)';% first row
    assert(length(vec)==numVars);
    step=numSel; % how much to advance by at each row
    Aeq=vec(mod(bsxfun(@plus,(0:-step:(-step*(numClauses-1)))',0:1:numVars-1),numVars)+1); % equality constraints: each clause must be assigned to exactly floor(numSel/numClauses) nodes.
    beq=max(floor(numSel/numClauses),1)*ones(numClauses,1);

    opts=optimset('Display','off');
    f=distMat_trunc(:); X0=false(numSel*numClauses,1);
    [tmp,~,exitflag]=bintprog(f,A,b,Aeq,beq,X0,opts);
    tmp=logical(reshape(tmp, numSel, numClauses));
    X=false(numNodes,numClauses);
    if exitflag<0
      warning(sprintf('Optimization failure in bintprog with exit flag %d!', exitflag));
      X(preselNodes,:)=true;
    else
      X(preselNodes,:)=tmp;
    end
  end
  warning on;
end
