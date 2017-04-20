-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nJoints,5}
ref.outputDim = {}
criterion = nn.ParallelCriterion()

if opt.addParallelSPPE then
  for i = 1,opt.nStack+opt.nParaStack do
      ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
      criterion:add(nn[opt.crit .. 'Criterion']())
  end
else
  for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
    criterion:add(nn[opt.crit .. 'Criterion']())
  end
end

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images, center scale format
local function generateSampleCS(set, idx)
    local img = dataset:loadImage(idx)
    local pts, c, s = dataset:getPartInfo(idx)
    local r = 0

    if set == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
    end

    local inp = crop(img, c, s, r, opt.inputRes)
    local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
    for i = 1,dataset.nJoints do
        if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(pts[i], c, s, r, opt.outputRes), opt.hmGauss)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    return inp,out
end

-- Code to generate training samples from raw images, bbox format
local function generateSampleBox(set, idx)
    local img = dataset:loadImage(idx)
    local pts, bndbox = dataset:getPartInfo(idx)
    local mu, std = dataset:getNormInfo(idx)
    local upLeft = torch.Tensor({bndbox[1][1],bndbox[1][2]})
    local bottomRight = torch.Tensor({bndbox[1][3],bndbox[1][4]})
    local ht = bottomRight[2]-upLeft[2]
    local width = bottomRight[1]-upLeft[1]
    local imght = img:size()[2]
    local imgwidth = img:size()[3]
    local scaleRate = 0.3
    upLeft[1] = math.max(0,(upLeft[1] - width*scaleRate/2))
    upLeft[2] = math.max(0,(upLeft[2] - ht*scaleRate/2))
    bottomRight[1] = math.min(imgwidth,(bottomRight[1] + width*scaleRate/2))
    bottomRight[2] = math.min(imght,(bottomRight[2] + ht*scaleRate/2))

    local out_center = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
    local xmin,ymin,xmax,ymax

    if (opt.usePGPG == true) then 
    -----------------------------------------------
    ------------- Doing random samples ------------
    -----------------------------------------------
        local PatchScale = torch.uniform()
        if PatchScale > 0.7 then
            ratio = ht/width
            -- Cut a patch
            if (width < ht) then
                patchWidth = PatchScale*width
                patchHt = patchWidth*ratio
            else
                patchHt = PatchScale*ht
                patchWidth = patchHt/ratio
            end
            xmin = upLeft[1]+torch.uniform()*(width-patchWidth)
            ymin = upLeft[2]+torch.uniform()*(ht-patchHt)
            xmax = xmin+patchWidth+1
            ymax = ymin+patchHt+1
        else
            --Gaussian distribution
            xmin = math.max(0, upLeft[1]+torch.normal(mu[1],std[1])*width)
            ymin = math.max(0, upLeft[2]+torch.normal(mu[2],std[2])*ht)
            xmax = math.max(math.min(imgwidth, bottomRight[1]+torch.normal(mu[3],std[3])*width), xmin+5)
            ymax = math.max(math.min(imght, bottomRight[2]+torch.normal(mu[4],std[4])*ht), ymin+5)
        end
 
         if opt.addParallelSPPE == true then
            if PatchScale > 0.7 then -- For patch, we 'disable' the parallel SPPE because STN do not need to focus in this case
                for i = 1,dataset.nJoints do
                    if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],torch.Tensor({xmin,ymin}):int(),torch.Tensor({xmax,ymax}):int(),opt.outputRes), opt.hmGauss)
                    end
                end
            else
                for i = 1,dataset.nJoints do
                    if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), opt.hmGauss)
                    end
                end
            end
        end
        upLeft[1] = xmin; upLeft[2] = ymin;
        bottomRight[1] = xmax; bottomRight[2] = ymax;
    end
    
    local inp = cropBox(img, upLeft:int(),bottomRight:int(), 0, opt.inputRes)
    local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
    for i = 1,dataset.nJoints do
        if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), opt.hmGauss)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
            if opt.addParallelSPPE == true then
            	out_center = shuffleLR(flip(out_center))
            end
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    if opt.addParallelSPPE == true then
        return inp,out,out_center
    else
        return inp,out
    end
end

-- Load in a mini-batch of data
function loadData(set, idxs)
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    if opt.dataset == 'mpii-cs' then 
        generateSample = generateSampleCS
    else
        generateSample = generateSampleBox
    end
    local nsamples = idxs:size(1)
    local input,label

    if opt.addParallelSPPE == true then
        for i = 1,nsamples do
            local tmpInput,tmpLabel,tmpLabelPara
            tmpInput,tmpLabel,tmpLabelPara = generateSample(set, idxs[i])
            tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
            tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
            tmpLabelPara = tmpLabelPara:view(1,unpack(tmpLabelPara:size():totable()))
            if not input then
                input = tmpInput
                label = tmpLabel
                labelPara = tmpLabelPara
            else
                input = input:cat(tmpInput,1)
                label = label:cat(tmpLabel,1)
                labelPara = labelPara:cat(tmpLabelPara,1)
            end
        end

        if opt.nStack > 1 then
            -- Set up label for intermediate supervision
            local newLabel = {}
            for i = 1,opt.nStack do newLabel[i] = label end
            for i = 1,opt.nStack do table.insert(newLabel,labelPara) end
            return input,newLabel
        else
            return input,{label,labelPara}
        end
    else
        for i = 1,nsamples do
        local tmpInput,tmpLabel
        tmpInput,tmpLabel = generateSample(set, idxs[i])
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        if not input then
            input = tmpInput
            label = tmpLabel
        else
            input = input:cat(tmpInput,1)
            label = label:cat(tmpLabel,1)
        end
        end

        if opt.nStack > 1 then
            -- Set up label for intermediate supervision
            local newLabel = {}
            for i = 1,opt.nStack do newLabel[i] = label end
            return input,newLabel
        else
            return input,label
        end
    end
end

function postprocess(set, idx, output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    local p_tf = torch.zeros(p:size())
    if opt.dataset == 'mpii-cs' then 
        for i = 1,p:size(1) do
            _,c,s = dataset:getPartInfo(idx[i])
            p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
        end
    else
        for i = 1,p:size(1) do
            local bndbox
            _, bndbox = dataset:getPartInfo(idx[i])
            local upLeft = torch.Tensor({bndbox[1][1],bndbox[1][2]})
            local bottomRight = torch.Tensor({bndbox[1][3],bndbox[1][4]})
            p_tf[i]:copy(transformBoxPreds(p[i], upLeft:int(), bottomRight:int(), opt.outputRes))
        end
    end
    
    return p_tf:cat(p,3):cat(scores,3)
end

function accuracy(output,label)
    if type(output) == 'table' and opt.addParallelSPPE == true then
        return heatmapAccuracy(output[opt.nStack],label[opt.nStack],nil,dataset.accIdxs)
    elseif type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs)
    end
end
