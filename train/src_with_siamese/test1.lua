require 'torch'
require 'stn'
require 'nn'
require 'nngraph'
require 'paths'
paths.dofile('ref.lua')
paths.dofile('models/hg-stacked.lua')
paths.dofile('models/spatial_transformer_with_theta.lua')--models/layers/stnbhwd/demo/spatial_transformer.lua
paths.dofile('models/map_back_stn.lua')
paths.dofile('models/Get_Alpha.lua')


local model1 = torch.load('umich-stacked-hourglass_1.t7')
local model2 = createMapBackStn(64,64)
local model3 = createMapBackStn(64,64)
local inp = nn.Identity()()
local img_stn,_theta = spanet(inp):split(2)
local out1_stn,out2_stn = model1(img_stn):split(2)
local alpha = nn.Get_Alpha()(_theta)
local out1 = model2({out1_stn,alpha})
local out2 = model3({out2_stn,alpha})
model = nn.gModule({inp}, {out1,out2})

--indata = torch.rand()
--indata1 = torch.rand(1,3,256,256)
for i, m in ipairs(model.modules) do
--   print(torch.type(m))
   print(m)
   if m.weight then
     print(#m.weight)
   end
end

--model:forward(indata1)
--g:backward(indata, gdata)

--graph.dot(m.fg, 'Forward Graph')
--graph.dot(c.bg, 'Backward Graph')
