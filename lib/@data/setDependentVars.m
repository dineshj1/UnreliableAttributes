function this=setDependentVars(this)
  % set private variables
  this.numInstances=size(this.X,1);
  this.lenX=size(this.X,2);
  this.lenY=size(this.Y,2);
  this.lenZ=size(this.Z,2);   
end
