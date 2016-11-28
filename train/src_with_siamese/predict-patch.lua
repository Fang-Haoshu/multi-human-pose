require 'torch'
require 'hdf5'
require 'nn'
require 'gnuplot'
require 'image'
require 'paths'
paths.dofile('util/siamese_input.lua')
-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-model","siamese-model/final_model.t7","Pretrained model")
cmd:option("-out", "out.png", "out file (.png)")
cmd:option("-gpu", true, "use gpu")
cmd:option("-log", "", "output log file")
cmd:option("-inpRes", 256, "input resolution")
cmd:option("-dist_thres", 0.5, "distance threshold")

params = cmd:parse(arg)

if params.log ~= "" then
    cmd:log(params.log, params)
end
torch.setdefaulttensortype('torch.FloatTensor')
dist_thres = params.dist_thres
-------------------------------------------------------------------------------
-- Load in annotations
-------------------------------------------------------------------------------

-- i is the i th valid document
i = 1

opt = {}
opt.dataDir = '/home/fanghaoshu/git/multi-human-pose/train/data/mpii'

annotLabels = {'train', 'valid'}
annot,ref = {},{}
for _,l in ipairs(annotLabels) do
    local a, namesFile
    if opt.dataset == 'mpii' and l == 'valid' and opt.finalPredictions == 1 then
        a = hdf5.open(opt.dataDir .. '/annot/valid.h5')
        namesFile = io.open(opt.dataDir .. '/annot/valid_images.txt')
    else
        a = hdf5.open(opt.dataDir .. '/annot/' .. l .. '.h5')
        namesFile = io.open(opt.dataDir .. '/annot/' .. l .. '_images.txt')
    end
    annot[l] = {}

    -- Read in annotation information
    local tags = {'part', 'xmax', 'ymin', 'xmin', 'ymax'}
    for _,tag in ipairs(tags) do annot[l][tag] = a:read(tag):all() end
    annot[l]['nsamples'] = annot[l]['part']:size()[1]

    -- Load in image file names (reading strings wasn't working from hdf5)
    annot[l]['images'] = {}
    local toIdxs = {}
    local idx = 1
    for line in namesFile:lines() do
        annot[l]['images'][idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- This allows us to reference multiple people who are in the same image
    annot[l]['imageToIdxs'] = toIdxs

end

ref.predict = {}
ref.predict.nsamples = annot.valid.nsamples
ref.predict.iters = annot.valid.nsamples
ref.predict.batchsize = 1

-- Default input is assumed to be an image and output is assumed to be a heatmap
-- This can change if an hdf5 file is used, or if opt.task specifies something different
--nParts = annot['train']['part']:size(2)
--dataDim = {3, opt.inputRes, opt.inputRes}
--labelDim = {nParts, opt.outputRes, opt.outputRes}


-----------------------------------------------------------------------------
------------Function that generate siamese network's input ------------------
-----------------------------------------------------------------------------

function generateSample(set, idx)
    local pts = annot[set]['part'][idx]
    local upLeft = torch.Tensor(2)
    local bottomRight = torch.Tensor(2)
    upLeft[1] = annot[set]['xmin'][idx]
    upLeft[2] = annot[set]['ymin'][idx]
    bottomRight[1] = annot[set]['xmax'][idx]
    bottomRight[2] = annot[set]['ymax'][idx]
    local ht = bottomRight[2]-upLeft[2]
    local width = bottomRight[1]-upLeft[1]
    local scaleRate = 0.25
    local bias = 0.1
    local rand = torch.rand(1)
    upLeft[1] = upLeft[1] - width*scaleRate/2 - rand*width*bias
    upLeft[2] = upLeft[2] - ht*scaleRate/2 - rand*ht*bias
    bottomRight[1] = bottomRight[1] + width*scaleRate/2 + (1-rand)*width*bias
    bottomRight[2] = bottomRight[2] + ht*scaleRate/2 + (1-rand)*ht*bias
	
    
    -- use patch augmentation 
    local heightScale = torch.rand(1)
    local widthScale = torch.rand(1)
    --PatchSacle = 0.5+0.5*PatchScale
    width = bottomRight[1]-upLeft[1]
    ht = bottomRight[2]-upLeft[2]
        -- pick the short size and multiple it with PatchScale
            patchWidth = widthScale*width
            patchHt = heightScale*ht
        upLeft[1] = upLeft[1]+torch.rand(1)*(width-patchWidth)
        upLeft[2] = upLeft[2]+torch.rand(1)*(ht-patchHt)
        bottomRight[1] = upLeft[1]+patchWidth
        bottomRight[2] = upLeft[2]+patchHt
      
    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])


    -- generate Part Input and Patch input 
    local Part_input = drawPartImage(img:size(),pts, 10, 0.5);
    --local Patch_input = torch.ones(img:size())
    Patch_input = drawPatchImage(img:size(), {upLeft:int(), bottomRight:int()})

    return Part_input, Patch_input
end

------------------------------------------------------------------------------
-------------------------- Load model ----------------------------------------
------------------------------------------------------------------------------

if params.model ~= "" then
    if params.gpu then
	require 'cutorch'
	require 'cunn'
	require 'cunnx'
	require 'cudnn'
        model = torch.load(params.model):cuda()
    else
	model = torch.load(params.model)
    end
end

for i,module in ipairs(model:listModules()) do
--for i,module in ipairs(model.modules) do
  print(torch.type(module))
   if (i == 2) then 
      paralleTable = module
      break
   end
end

for i, module in ipairs(paralleTable.modules) do
   print(torch.type(module))
   if (i == 1) then
      deploy_model = module
      break
   end
end

-----------------------------------------------------------------------------
------------------- Doing cluster -------------------------------------------
-----------------------------------------------------------------------------

--annot.valid.nsamples = 1

local distance = torch.Tensor(annot.valid.nsamples)
local pred_labels = torch.ones(annot.valid.nsamples)

print("generating embeddings ")
plot_results = {}
plot_coordi = {}
plot_coordi_1 = {}
plot_dist = {}

for sample = 1, annot.valid.nsamples do
   idx = torch.random(annot.valid.nsamples)
   local part, patch = generateSample('valid', idx)
   part = image.scale(part, params.inpRes, params.inpRes):cuda()
   patch = image.scale(patch, params.inpRes, params.inpRes):cuda()
   local input = {part,patch} -- the tensor containing two images 
   dist = model:forward(input)[1]
   distance[sample] = dist
   if (dist < dist_thres) then 
	pred_labels[sample] = 1
w1 = image.display{image=part,win=w1}
w2 = image.display{image=patch,win=w2}
--	image.save('patch result/positive patch/part/' .. sample .. '.jpg', part)
--	image.save('patch result/positive patch/patch/' .. sample .. '.jpg', patch)
   else 
	pred_labels[sample] = -1
--	image.save('patch result/negative patch/part/' .. sample .. '.jpg', part)
--	image.save('patch result/negative patch/patch/' .. sample .. '.jpg', patch)
   end
   xlua.progress(sample, annot.valid.nsamples)
end


-------------------------------------------------------------
------------- print results ---------------------------------
-------------------------------------------------------------

print(torch.sum(torch.eq(pred_labels, -1)))
print(torch.sum(torch.eq(pred_labels, 1)))


print("done.") 
---------------------------------------------------------------------------------------------------------------




