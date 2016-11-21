Crop3DFun = {}

function Crop3DFun:crop(input, target, cropSize, center)
        -- default crop size if 30
	local size = cropSize or 30
        -- random crop or crop to the middle
	center = center or false
	local input = input:cuda():squeeze()
	local target = target:cuda():squeeze()
        -- if crop to the middel,then the size is side/sqrt(3)
	if center == true then
		size = torch.floor( input:size(1) / torch.sqrt(3) )
	end
	local x0 = 0
	local y0 = 0
	local z0 = 0
        -- x0 y0 z0 upper left corner vertex of the cube
	if center == true then
		local offset = torch.floor((input:size(1) - size)/2);
		x0 = offset + 1;
		y0 = offset + 1;
		z0 = offset + 1;
	else
		x0 = torch.random(1, input:size(1) - size + 1)
		y0 = torch.random(1, input:size(1) - size + 1)
		z0 = torch.random(1, input:size(1) - size + 1)
	end
	local _input = input[{{x0, x0 + size - 1},{y0, y0 + size - 1},{z0, z0 + size - 1}}]
	local _target = target[{{x0, x0 + size - 1},{y0, y0 + size - 1},{z0, z0 + size - 1}}]
	local input_tensor = torch.ones(1, size, size, size):cuda();
	local target_tensor = torch.ones(1, size, size, size):cuda();
	input_tensor[{{1},{},{},{}}] = _input
	target_tensor[{{1},{},{},{}}] = _target
	return input_tensor, target_tensor
end

