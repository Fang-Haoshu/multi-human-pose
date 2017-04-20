local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    local annot = {}
    local tags = {'imgname','part','bndbox','mu','std'}
    local a = hdf5.open('../data/'..opt.dataset..'/annot.h5','r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        opt.idxRef = {}
        -- Set up training/validation split
        opt.idxRef.train = torch.randperm(annot.part:size(1)-1358)
        opt.idxRef.valid = torch.range(annot.part:size(1)-1358,annot.part:size(1))

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
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx),3)
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local bndbox = self.annot.bndbox[idx]:clone()

    return pts, bndbox
end

function Dataset:getNormInfo(idx)
    local mu = self.annot.mu[idx]:clone()
    local std = self.annot.std[idx]:clone()

    return mu, std
end

return M.Dataset

