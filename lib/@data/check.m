function val=check(this)
  try
    if isempty(this.X)
      warning('JD: X is empty');
      this.lenX=0;
    else
      assert(size(this.X,1)==this.numInstances, 'JD: X has %d(\neq numInstances=%d) rows', size(this.X,1), this.numInstances);
      assert(size(this.X,2)==this.lenX);
    end

    if isempty(this.Y)
      warning('JD: Y is empty');
      this.lenY=0;
    else
      assert(size(this.Y,1)==this.numInstances', 'JD: Y has %d(\neq numInstances=%d) rows', size(this.Y,1), this.numInstances);
      assert(size(this.Y,2)==this.lenY);
    end 

    if isempty(this.Z)
      %warning('JD: Z is empty');
      this.lenZ=0;
    else
      assert(size(this.Z,1)==this.numInstances', 'JD: Z has %d(\neq numInstances=%d) rows', size(this.Z,1), this.numInstances);
      assert(size(this.Z,2)==this.lenZ);
    end 
  catch err
    if strcmp(err.msg(1:4),'JD: ')
      val=false;
      return;
    else
      getReport(err);
    end
  end
  val=true;
end
