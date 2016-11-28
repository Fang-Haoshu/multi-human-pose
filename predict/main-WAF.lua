require 'paths'
require 'stn'
require 'nn'
nnlib = nn
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('spatial_transformer_with_theta.lua')
paths.dofile('Get_Alpha.lua')


--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1]  == 'demo' or arg[1] == 'predict-test' then
    -- Test set annotations do not have ground truth part locations, but provide
    -- information about the location and scale of people in each image.
    --a = loadAnnotations('MPII_multi_valid')
    a = loadAnnotations('WAF_test')

elseif arg[1] == 'predict-valid' or arg[1] == 'eval' then
    -- Validation set annotations on the other hand, provide part locations,
    -- visibility information, normalization factors for final evaluation, etc.
    a = loadAnnotations('MPII_valid_gt')

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on a few demo images")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    print("    eval - Run basic evaluation on predictions from the validation set")
    return
end

--m = torch.load('../train/exp/mpii/parallel-grad-0.1e-4/model_7.t7')   -- Load pre-trained model
m = torch.load('../train/exp/mpii/parallel+correctstn+fulldata-c/model_40.t7')
--print(m.forwardnodes[3].data.module.forwardnodes[6])
--m = m.forwardnodes[3].data.module.forwardnodes[6].data.module
--m = torch.load('/home/fred/Git/human-pose-prediction/train/exp/mpii/new2-ccc/model_20.t7')   -- Load pre-trained model
if arg[1] == 'demo' then
    --idxs = torch.Tensor({695, 3611, 2486, 7424, 10032, 5, 4829})
    -- If all the MPII images are available, use the following line to see a random sampling of images
    --idxs = torch.randperm(a.nsamples):sub(1,50)
    idxs = torch.range(1,a.nsamples)
else
    idxs = torch.range(1,a.nsamples)
end

if arg[1] == 'eval' then
    nsamples = 0
else
    nsamples = idxs:nElement() 
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
    scores = torch.Tensor(nsamples,16,1)
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image

    --local im = image.load('data/OutdoorPoseDataset/' .. a['images'][idxs[i]])
  local im = image.load('/data/fanghaoshu/WAF/images/' .. a['images'][idxs[i]],3)
  if(im:size()[1] == 1) then
    print('black white image '..a['images'][idxs[i]])
  else      
    local imght = im:size()[2]
    local imgwidth = im:size()[3]
    local pt1= torch.Tensor(2)
    local pt2= torch.Tensor(2)
    pt1[1] = a['xmin'][idxs[i]]
    pt1[2] = a['ymin'][idxs[i]]
    pt2[1] = a['xmax'][idxs[i]]
    pt2[2] = a['ymax'][idxs[i]]
    local ht = pt2[2]-pt1[2]
    local width = pt2[1]-pt1[1]
    local scaleRate = 0.2
    local bias=0
    local rand = torch.rand(1)
    local _pt1 = pt1[1] - width*0.3/2 - rand*width*bias
    local _pt2 = pt1[2] - ht*scaleRate/2 - rand*ht*bias
    pt1[1] = math.max(0,_pt1[1])
    pt1[2] = math.max(0,_pt2[1])
    pt2[1] = math.min(imgwidth,(pt2[1] + width*0.3/2 + (1-rand)*width*bias)[1])
    pt2[2] = math.min(imght,(pt2[2] + ht*scaleRate/2 + (1-rand)*ht*bias)[1])
    --pt1[1] = _pt1[1]
    --pt1[2] = _pt2[1]
    --pt2[1] = pt2[1] + width*scaleRate/2 + (1-rand)*width*bias
    --pt2[2] = pt2[2] + ht*scaleRate/2 + (1-rand)*ht*bias
    local inputRes = 256
    
    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputRes)
    -- Get network output
    local out = m:forward(inp:view(1,3,inputRes,inputRes):cuda())
    cutorch.synchronize()
    local hm = out[2][1]:float()
    hm[hm:lt(0)] = 0
    --print (hm)
    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img, pred_scores = getPreds(hm, pt1:int(), pt2:int())


    --To see if the bounding box crops a person
    --if isHuman(hm) then
        preds[i]:copy(preds_img)
        scores[i]:copy(pred_scores)

        xlua.progress(i,nsamples)

        -- Display the result
        if arg[1] == 'demo' then
            preds_hm:mul(inputRes/64) -- Change to input scale
            local dispImg = drawOutput(inp, hm, preds_hm[1])
            w = image.display{image=dispImg,win=w}
            image.save('preds/images/' .. tostring(i) .. '.jpg',dispImg)
            sys.sleep(3)
        end
    --end

    collectgarbage()
  end
end

-- Save predictions
if arg[1] == 'predict-valid' then
    local predFile = hdf5.open('preds/valid.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif arg[1] == 'predict-test' then
    local predFile = hdf5.open('preds/test.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif arg[1] == 'demo' then
    w.window:close()
end

--------------------------------------------------------------------------------
-- Evaluation code
--------------------------------------------------------------------------------

if arg[1] == 'eval' then
    -- Calculate distances given each set of predictions
    local labels = {'valid-example','valid-ours'}
    local dists = {}
    for i = 1,#labels do
        local predFile = hdf5.open('preds/' .. labels[i] .. '.h5','r')
        local preds = predFile:read('preds'):all()
        table.insert(dists,calcDists(preds, a.part, a.normalize))
    end

    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')    
    gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')
    displayPCK(dists, {9,10}, labels, 'Head')
    displayPCK(dists, {2,5}, labels, 'Knee')
    displayPCK(dists, {1,6}, labels, 'Ankle')
    gnuplot.raw('set tmargin 2.5')
    gnuplot.raw('set bmargin 1.5')
    displayPCK(dists, {13,14}, labels, 'Shoulder')
    displayPCK(dists, {12,15}, labels, 'Elbow')
    displayPCK(dists, {11,16}, labels, 'Wrist', true)
    gnuplot.raw('unset multiplot')
end
