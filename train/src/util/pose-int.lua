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
	local rand = torch.rand(1)
	upLeft[1] = math.max(0,(upLeft[1] - width*scaleRate/2))
	upLeft[2] = math.max(0,(upLeft[2] - ht*scaleRate/2))
	bottomRight[1] = math.min(imgwidth,(bottomRight[1] + width*scaleRate/2))
	bottomRight[2] = math.min(imght,(bottomRight[2] + ht*scaleRate/2))

	local out_center = torch.zeros(nParts, opt.outputRes, opt.outputRes)
----------------------------------------------------------------------------------
------------- Doing random samples -----------------------------------------------
----------------------------------------------------------------------------------
	if (opt.addDPG == true) then 
		local Part_input = image.scale(drawPartImage(img:size(),pts, 10, 2.5), opt.inputRes, opt.inputRes):cuda();
		local patch_num = 10
		local labels = torch.ones(patch_num)
		local xmin = torch.ones(patch_num)
		local xmax = torch.ones(patch_num)
		local ymin = torch.ones(patch_num)
		local ymax = torch.ones(patch_num)
		local PatchScale = torch.ones(patch_num)
	
		for i = 1, patch_num do
			PatchScale[i] = torch.rand(1)
			if PatchScale[i] > 0.5 then
				ratio = ht/width
				-- Cut a patch
				if (width < ht) then
					patchWidth = PatchScale[i]*width
					patchHt = patchWidth*ratio
				else
					patchHt = PatchScale[i]*ht
					patchWidth = patchHt/ratio
				end
				xmin[i] = upLeft[1]+torch.rand(1)*(width-patchWidth)
				ymin[i] = upLeft[2]+torch.rand(1)*(ht-patchHt)
				xmax[i] = xmin[i]+patchWidth+1
				ymax[i] = ymin[i]+patchHt+1
			else
				--Gaussian distribution
				xmin[i] = upLeft[1]+torch.normal(-0.0142,0.1158)*width
				ymin[i] = upLeft[2]+torch.normal(0.0043,0.068)*ht
 				xmax[i] = bottomRight[1]+torch.normal(0.0154,0.1337)*width
				ymax[i] = bottomRight[2]+torch.normal(-0.0013,0.0711)*ht
			end
			local Patch_input = image.scale(drawPatchImage(img:size(), {{xmin[i],ymin[i]}, {xmax[i], ymax[i]}}), opt.inputRes, opt.inputRes):cuda()
			local dist = Siamese_model:forward({Part_input, Patch_input})
			if (dist[1] > 0.8) then labels[i] = 0 end
		end
	
		-- pick labels that are 1's 
		xmin = xmin[torch.eq(labels,1)]:int(); ymin = ymin[torch.eq(labels,1)]:int();
		xmax = xmax[torch.eq(labels,1)]:int(); ymax = ymax[torch.eq(labels,1)]:int();
		PatchScale = PatchScale[torch.eq(labels,1)]:int();
		--print(xmin:nElement())	
		if (xmin:nElement() > 0) then
			pick_id = torch.random(xmin:nElement())
			if (xmax[pick_id] > 1 and ymax[pick_id] > 1) and (xmin[pick_id] < img:size(3) and ymin[pick_id] < img:size(2))and (xmax[pick_id] ~= xmin[pick_id]) and (ymax[pick_id] ~= ymin[pick_id]) then
				-------------------------------------------------
				----------- 'Perfect' located human labels-------
				-------------------------------------------------
			     if opt.addParallelSPPE == true then
					if PatchScale[pick_id] > 0.5 then -- For patch, we 'disable' the parallel SPPE because STN do not need to focus in this case
						for i = 1,nParts do
							if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
								drawGaussian(out_center[i], transformBox(pts[i],({xmin[pick_id],ymin[pick_id]}):int(),({xmax[pick_id],ymax[pick_id]}):int(),opt.outputRes), 1)
							end
						end
					else
						for i = 1,nParts do
							if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
								drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 1)
							end
						end
					end
				end
            upLeft[1] = xmin[pick_id]; upLeft[2] = ymin[pick_id];
			bottomRight[1] = xmax[pick_id]; bottomRight[2] = ymax[pick_id];
			upLeft = upLeft:int();bottomRight=bottomRight:int()
			end
		elseif opt.addParallelSPPE == true then
			for i = 1,nParts do
				if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
					drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 1)
				end
			end
		end
		
		if(upLeft[1] == bottomRight[1]) then
			bottomRight[1]=upLeft[1]+1
		end
		
		if(upLeft[2] == bottomRight[2]) then
			bottomRight[2]=upLeft[2]+1
		end
	end
	
	local inp = cropBox(img, upLeft:int(),bottomRight:int(), 0, opt.inputRes)
	local out = torch.zeros(nParts, opt.outputRes, opt.outputRes)
	for i = 1,nParts do
		if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
			drawGaussian(out[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputRes), 1)
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
