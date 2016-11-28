local GetAlpha, parent = torch.class('nn.Get_Alpha', 'nn.Module')

function GetAlpha:__init()
   parent.__init(self)
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function GetAlpha:updateOutput(input)
   local theta = input
   if (theta:nDimension() == 2) then
      theta = addOuterDim(theta)
   end
   assert ((theta:nDimension() == 3) and (theta:size(2) == 2) and (theta:size(3) == 3), 'Please make sure theta is 2x3 matrix')
   local alpha = torch.CudaTensor(theta:size()):zero()
   local batchsize = alpha:size(1)
   for i=1,batchsize do
      alpha:select(1,i)[{{},{1,2}}]:copy(torch.inverse(theta:select(1,i)[{{},{1,2}}]))
--      alpha:select(1,i)[{{},3}] = torch.mv(alpha:select(1,i)[{{},{1,2}}],torch.mul(theta:select(1,i):select(2,3), -1))
      alpha:select(1,i)[{{},3}] = torch.mv(alpha:select(1,i)[{{},{1,2}}],theta:select(1,i):select(2,3))
   end
   self.output = alpha
   return alpha
end

function GetAlpha:updateGradInput(_input,gradOutput)
   self.gradInput = torch.CudaTensor(_input:size()):fill(0)
   return self.gradInput
end
