require 'torch'
require 'nn'
require 'image'
require 'hdf5'


--define different colors for different parts
colors = {{0.3,0.6,0.7}, {0.2,0.1,0.9}, {0.8,0.4,0.2}, {0.2,0.6,0.1},
        {0.5,0.33,0.55}, {0.144, 0.668, 0.42}, {0.35, 0.35, 0.6}, {0.7,0.5,0.2},
        {0.42, 0.72, 0.72}, {0.85, 0.63, 0.5}, {0.1, 0.9, 0.2}, {0.9, 0.6, 0.3},
        {0.5, 0.7, 0.32}, {0.342, 0.455, 0.763}, {0.66, 0.11, 0.8}, {0.44, 0.22, 0.4}
        };

function drawPartImage(size, part, sigma, delta)
   local img = torch.ones(size)
   for i = 1,16 do
	if part[i][1] ~= 0 or part[i][2] ~= 0 then 
   	img = drawPart(img, part[i], sigma, delta, colors[i])
	end
   end
   return img
end

function drawPatchImage(size, coordi)
   local img = torch.ones(size)
   local ul = coordi[1]
   local br = coordi[2]
   -- make sure the upper left and bottom right are correct
   if (ul[1] > img:size(3) or ul[2] > img:size(2) or br[1] < 1 or br[2] < 1) then return img end
   -- Image range
   local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(3))}
   local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(2))}
   -- generate black patchs
   img:sub(1,img:size(1), img_y[1], img_y[2], img_x[1], img_x[2]):fill(0)
   return img
end


function drawPart(img,pt, sigma, delta, color)  -- delta is not used 
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    local red = color[1]
    local green = color[2]
    local blue = color[3]
    -- If not, return the image as is
    if (ul[1] > img:size(3) or ul[2] > img:size(2) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    g:div(torch.max(g))
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(3)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(2)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(3))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(2))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img[1]:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(red-1, g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[2]:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(green-1, g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[3]:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(blue-1, g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))

--    img[img:lt(0)] = 0
    return img
end
