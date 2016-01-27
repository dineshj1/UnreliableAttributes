function drawTree(treeModel)
 for i=1:length(treeModel.weakModels)
   currNode=treeModel.weakModels{i};
   try
     r=currNode.r;
     t=currNode.t;
     ig=currNode.Igain_net;
     %if isnan(ig)
     %    5;
     %end
   catch
     r=0;t=0;ig=0;
   end

   text(currNode.xpos, currNode.ypos, sprintf('x%d>%f:IG:%g',r, t,ig)); %TODO fix properly if this is greater or lesser

   % drawing lines to children nodes
   line([currNode.xpos currNode.xpos-currNode.gaps*0.25], [currNode.ypos, currNode.ypos-1]);
   line([currNode.xpos currNode.xpos+currNode.gaps*0.25], [currNode.ypos, currNode.ypos-1]);

   % % makeshift code that doesn't look like tree
   %text(currNode.levelPos, currNode.ypos, sprintf('x%d>%f',r, t));

   %% drawing lines to children nodes
   %line([currNode.levelPos 2*currNode.levelPos-1], [currNode.ypos, currNode.ypos-1]);
   %line([currNode.levelPos  2*currNode.levelPos], [currNode.ypos, currNode.ypos-1]);
 end
 % TODO text at leaf nodes - class probabilities using leafdist
end
