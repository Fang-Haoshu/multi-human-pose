require 'paths'
require 'stn'
require 'nn'
nnlib = nn
--paths.dofile('models/spatial_transformer_with_theta.lua')
paths.dofile('models/Get_Alpha.lua')
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
--paths.dofile('model.lua')   -- Read in network model
paths.dofile('util.lua')   -- Load up training/testing functions

a = loadAnnotations('MPII_valid')


--m = torch.load('ftStn_nobias_0.81.t7')   -- Load pre-trained model
m = torch.load('loadModel/2STN+patch.t7')   -- Load pre-trained model
--m = torch.load('final_model.t7')
for i,module in ipairs(m:listModules()) do
--for i,module in ipairs(m:modules) do
  print(torch.type(module))
   if (i == 5) then 
--   if (i == 3) then
      spatial_transformer = module
      break
   end
end

--for i, modules in ipairs(m.modules) do
--   print(torch.type(modules))
--   if (i == 100) then
--      spatial_transformer = modules
--      break end
--end

idxs = torch.range(1,a.nsamples)
nsamples = idxs:nElement() 
xlua.progress(0,nsamples)

index = 3
for i = 1, 10 do
--for i = index,index+1 do
--for i = 1,nsamples do
   local im = image.load('data/images/' .. a['images'][idxs[i]])
    local pt1= torch.Tensor(2)
    local pt2= torch.Tensor(2)
    pt1[1] = a['xmin'][idxs[i]]
    pt1[2] = a['ymin'][idxs[i]]
    pt2[1] = a['xmax'][idxs[i]]
    pt2[2] = a['ymax'][idxs[i]]
    local ht = pt2[2]-pt1[2]
    local width = pt2[1]-pt1[1]
    local scaleRate = 0.3
    local bias=0
    local rand = torch.rand(1)
    pt1[1] = pt1[1] - width*scaleRate/2 - rand*width*bias
    pt1[2] = pt1[2] - ht*scaleRate/2 - rand*ht*bias
    pt2[1] = pt2[1] + width*scaleRate/2 + (1-rand)*width*bias
    pt2[2] = pt2[2] + ht*scaleRate/2 + (1-rand)*ht*bias
    local inputRes = 256

    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputRes)
    -- Get network output
--    local out = m:forward(inp:view(1,3,inputRes,inputRes):cuda())
   local out = spatial_transformer:forward(inp:view(1,3,inputRes,inputRes):cuda())

--   theta = spatial_transformer:get(1):get(2):get(11).output
--   print(theta)

--   dispImg = out[1]:view(out[1]:size(2),out[1]:size(3),out[1]:size(4))
--   print(out[2])
   print(out[2]:size())
   dispImg = out[1]:view(out[1]:size(2),out[1]:size(3),out[1]:size(4))
   print(dispImg:size())
   xlua.progress(i,nsamples)
   w = image.display{image=dispImg,win=w}
   image.save('preds/full-stn-1/' .. i .. '.jpg' ,dispImg)
   image.save('preds/full-stn-1/' .. i .. '_1.jpg' ,inp:view(3,inputRes,inputRes))
--   sys.sleep(5)
   
   collectgarbage()
end

w.window:close()
