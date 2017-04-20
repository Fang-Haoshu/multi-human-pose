local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 17
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
    self.flipRef = {{2,3},   {4,5},   {6,7},
                    {8,9}, {10,11}, {12,13},
                    {14,15}, {16,17}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,6,1},      {1,7,1},
                        {6,8,2},    {7,9,2},    {8,10,2},      {9,11,2},
                        {1,12,3},      {1,13,3},
                        {12,14,4},     {14,16,4},   {13,15,4},     {15,17,4}}

    local annot = {}
    local tags = {'imgname','part','bndbox'}
    local a = hdf5.open('../data/'..opt.dataset..'/annot.h5','r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        opt.idxRef = {}
        -- Set up training/validation split
        opt.idxRef.train = torch.randperm(annot.part:size(1)-5887)
        opt.idxRef.valid = torch.range(annot.part:size(1)-5887,annot.part:size(1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()):sub(1,31))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx),3)
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local bndbox = self.annot.bndbox[idx]:clone()

    return pts, bndbox
end


return M.Dataset

