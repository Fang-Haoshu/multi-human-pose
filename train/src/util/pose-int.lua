-- Get prediction coordinates
predDim = {nParts,2}

criterion = nn.ParallelCriterion()
          :add(nn.MSECriterion())
          :add(nn.MSECriterion())

-- Code to generate training samples from raw images.
function generateSample(set, idx)
    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])
    local imght = img:size()[2]
    local imgwidth = img:size()[3]    
    local pts = annot[set]['part'][idx]
    local upLeft = torch.Tensor(2)
    local bottomRight = torch.Tensor(2)
    local bndbox = annot[set]['bndbox'][idx]
    upLeft[1] = bndbox[1][1]
    upLeft[2] = bndbox[1][2]
    bottomRight[1] = bndbox[1][3]
    bottomRight[2] = bndbox[1][4]
    local ht = bottomRight[2]-upLeft[2]
    local width = bottomRight[1]-upLeft[1]
    local scaleRate = 0.3
    local bias = 0
    local rand = torch.rand(1)
    local usePatch = 1
    upLeft[1] = math.max(0,(upLeft[1] - width*scaleRate/2 - rand*width*bias)[1])
    upLeft[2] = math.max(0,(upLeft[2] - ht*0.2/2 - rand*ht*bias)[1])
    bottomRight[1] = math.min(imgwidth,(bottomRight[1] + width*scaleRate/2 + (1-rand)*width*bias)[1])
    bottomRight[2] = math.min(imght,(bottomRight[2] + ht*0.2/2 + (1-rand)*ht*bias)[1])

    local out_center = torch.zeros(nParts, opt.outputRes, opt.outputRes)

    if usePatch==1 then
        -- use patch augmentation 
        local PatchScale = torch.rand(1)
        --print (PatchScale)
        --print(upLeft,bottomRight)
        width = bottomRight[1]-upLeft[1]
        ht = bottomRight[2]-upLeft[2]
        if PatchScale[1] > 0.5 then
            ratio = ht/width
            -- pick the short size and multiple it with PatchScale
            if (width < ht) then
                patchWidth = PatchScale*width
                patchHt = patchWidth*ratio
            else
                patchHt = PatchScale*ht
                patchWidth = patchHt/ratio
            end
            upLeft[1] = upLeft[1]+torch.rand(1)*(width-patchWidth)
            upLeft[2] = upLeft[2]+torch.rand(1)*(ht-patchHt)
            bottomRight[1] = upLeft[1]+patchWidth+1
            bottomRight[2] = upLeft[2]+patchHt+1
            if opt.addParallelSPPE == true then
                
                for i = 1,nParts do
                    if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 2)
                    end
                end
            end

        else
            if opt.addParallelSPPE == true then
                for i = 1,nParts do
                    if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 2)
                    end
                end
            end
            upLeft[1] = upLeft[1]-torch.normal(-0.0142,0.1158)*width
            upLeft[2] = upLeft[2]-torch.normal(0.0043,0.068)*ht
            bottomRight[1] = bottomRight[1]+torch.normal(0.0154,0.1337)*width
            bottomRight[2] = bottomRight[2]+torch.normal(-0.0013,0.0711)*ht
        end
        --print(upLeft,bottomRight)
    end  

    -- For single-person pose estimation with a centered/scaled figure
    local inp = cropBox(img, upLeft:int(),bottomRight:int(), 0, opt.inputRes)
    local out = torch.zeros(nParts, opt.outputRes, opt.outputRes)
    for i = 1,nParts do
        if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 2)
        end
    end


    
    if opt.addParallelSPPE == true then
        return inp,out,out_center
    else
        return inp,out
    end
end

function preprocess(input, label)
    return input, {label,label}
end

function postprocess(set, idx, output)
    local preds = getPreds(output[#output])
    return preds
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[opt.dataset])
end
