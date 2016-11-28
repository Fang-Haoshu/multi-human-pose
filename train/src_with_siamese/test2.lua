require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

opt = {}
opt.netType = 'hg-stacked'
opt.loadModel = 'umich-stacked-hourglass.t7'
paths.dofile('models/' .. opt.netType .. '.lua')
paths.dofile('models/spatial_transformer_with_theta.lua')--models/layers/stnbhwd/demo/spatial_transformer.lua
paths.dofile('models/map_back_stn.lua')
paths.dofile('models/Get_Alpha.lua')

nngraph.setDebug(true)


local model1 = torch.load(opt.loadModel)
local model2 = createMapBackStn(64,64)
local model3 = createMapBackStn(64,64)
local inp = nn.Identity()()
local img_stn,theta = spanet(inp):split(2)
local out1_stn,out2_stn = model1(img_stn):split(2)
local alpha = nn.Get_Alpha()(theta)
local out1 = model2({out1_stn,alpha})
local out2 = model3({out2_stn,alpha})
model = nn.gModule({inp}, {out1,out2})

gdata = torch.Tensor(1,16,64,64)
pcall(function() model:backward(input) end)
