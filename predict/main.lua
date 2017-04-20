require 'paths'
require 'stn'
require 'nn'
nnlib = nn
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('Get_Alpha.lua')


--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1]  == 'demo' or arg[1] == 'predict-test' then
    a = loadAnnotations('mpii-test0.09/test-bbox')

elseif arg[1] == 'predict-valid' then
    a = loadAnnotations('MPII_multi_valid')--valid set during our exp

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on the test set")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    return
end

m = torch.load('./final_model.t7')

idxs = torch.range(1,a.nsamples)

nsamples = idxs:nElement() 
-- Displays a convenient progress bar
xlua.progress(0,nsamples)
preds = torch.Tensor(nsamples,16,2)
scores = torch.Tensor(nsamples,16,1)


--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image
    local im = image.load('data/images/' .. a['images'][idxs[i]])
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
    local scaleRate = 0.3
    pt1[1] = math.max(0,(pt1[1] - width*scaleRate/2))
    pt1[2] = math.max(0,(pt1[2] - ht*scaleRate/2))
    pt2[1] = math.min(imgwidth,(pt2[1] + width*scaleRate/2))
    pt2[2] = math.min(imght,(pt2[2] + ht*scaleRate/2))

    local inputRes = 256
    
    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputRes)
    -- Get network output
    local out = m:forward(inp:view(1,3,inputRes,inputRes):cuda())
    cutorch.synchronize()
    local hm = out[8][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img, pred_scores = getPreds(hm, pt1:int(), pt2:int())


    preds[i]:copy(preds_img)
    scores[i]:copy(pred_scores)

    xlua.progress(i,nsamples)

    -- Display the result
    if arg[1] == 'demo' then
        preds_hm:mul(inputRes/64) -- Change to input scale
        local dispImg = drawOutput(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        --image.save('preds/images/' .. tostring(i) .. '.jpg',dispImg)
        sys.sleep(3)
    end


    collectgarbage()
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
