--- Load up network model or initialize from scratch
require 'stn'
paths.dofile('models/' .. opt.netType .. '.lua')
paths.dofile('models/spatial_transformer_with_theta.lua')
paths.dofile('models/map_back_stn.lua')
paths.dofile('models/Get_Alpha.lua')

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

elseif opt.addParallelSPPE ==  true and opt.addSTN == false then
   assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
   print('==> Loading model from: ' .. opt.loadModel..' with parallel SPPE')
   model = torch.load(opt.loadModel)

elseif opt.addParallelSPPE ==  true then
   assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
   print('==> Loading model from: ' .. opt.loadModel..' and add parallel SPPE')
   local model1 = torch.load(opt.loadModel)
   local model2 = createMapBackStn(64,64)
   local model3 = createMapBackStn(64,64)
   local model4 = torch.load(opt.loadModel)
   local inp = nn.Identity()()
   local img_stn,_theta = spanet(inp):split(2)
   local out1_stn,out2_stn = model1(img_stn):split(2)
   local out3_stn,out4_stn = model4(img_stn):split(2)
   local alpha = nn.Get_Alpha()(_theta)
   local out1 = model2({out1_stn,alpha})
   local out2 = model3({out2_stn,alpha})
   model = nn.gModule({inp}, {out1,out2,out3_stn,out4_stn})
   
-- Or we add a STN to a trained model 
elseif opt.addSTN ~= false and opt.loadModel ~= 'none' then
   assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
   print('==> Loading model from: ' .. opt.loadModel .. ' and add SSTN')
   local model1 = torch.load(opt.loadModel)
   local model2 = createMapBackStn(64,64)
   local model3 = createMapBackStn(64,64)
   local inp = nn.Identity()()
   local img_stn,_theta = spanet(inp):split(2)
   local out1_stn,out2_stn = model1(img_stn):split(2)
   local alpha = nn.Get_Alpha()(_theta)
   local out1 = model2({out1_stn,alpha})
   local out2 = model3({out2_stn,alpha})
   model = nn.gModule({inp}, {out1,out2})
    

-- Or we add a STN to a new model 
elseif opt.addSTN ~= false and opt.loadModel == 'none' then
    print('==> Creating model from file: models/' .. opt.netType .. '.lua add SSTN')
    local model1 = createModel(modelArgs)
    local model2 = createMapBackStn(64,64)
    local model3 = createMapBackStn(64,64)
    local inp = nn.Identity()()
    local img_stn,_theta = spanet(inp):split(2)
    local out1_stn,out2_stn = model1(img_stn):split(2)
    local alpha = nn.Get_Alpha()(_theta)
    local out1 = model2({out1_stn,alpha})
    local out2 = model3({out2_stn,alpha})
    model = nn.gModule({inp}, {out1,out2})


-- Or a path to previously trained model is provided
elseif opt.addSTN == false and opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh without STN
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
end


-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
end
