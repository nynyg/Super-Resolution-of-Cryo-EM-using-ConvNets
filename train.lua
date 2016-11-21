--[[

Name: Nan Yang
Email: nan.yang@tum.de
Project: Super Resolution of Cryo-Em
Tutor: Vladimir Golkov
Data: 08.08.2016

This file is the main trainig file

]]--

-- load the libraries needed
-- nn cunn cudnn are for constructing CNNs
-- optim is used for Adam optimizer
-- Rot3DFun and Crop3DFun are used for random rotation and cropping the tensor
-- gnuplot for ploting figures
-- matio is used for load and save matlab file

require 'nn';
require 'cunn';
require 'cudnn';
require 'optim';
require 'utils/Rot3DFun'
require 'utils/Crop3DFun'
require 'gnuplot'
matio = require 'matio';
logger = optim.Logger('loss_log_3layers.txt')
gnuplot.setterm("png")

-- parse the cmd paramers
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Super Resolution of EM')
cmd:text()
cmd:text('Options:')
cmd:option('-epoch', 1000, 'total training epochs')
cmd:option('-learning_rate', 1e-3, 'learning rate')
cmd:option('-optim', 'adam', 'which optimizer to use')
cmd:option('-layer_num', 3, 'the number of layers')
cmd:option('-neuron_num', '64/128/256', 'feature maps for each layer')
cmd:option('-kernel_size', 3, 'size of convolution kernel')
cmd:option('-border', 'valid', 'border method')
cmd:option('-input', 'filtered50k', 'training input')
cmd:option('-target', 'filtered688k', 'training target')
cmd:text()
local opt = cmd:parse(arg)
print(opt)

-- prepare data
-- load mat file
--input_file_name = '../data/train/2/' .. opt.input .. '.mat';
input_file_name = './' .. opt.input .. '.mat';
--target_file_name = '../data/train/2/' .. opt.target .. '.mat';
target_file_name = './' .. opt.target .. '.mat';
input = matio.load(input_file_name);
target = matio.load(target_file_name);
input = input.filtered50k; 
target = target.filtered688k; 
-- store the data into 4D tensor
input_tensor = torch.zeros(1, 180, 180, 180);
target_tensor = torch.zeros(1, 180, 180, 180);
input_tensor[{{1}, {}, {}, {}}] = input; 
target_tensor[{{1}, {}, {}, {}}] = target;

-- construct the CNNs
kernelSize = opt.kernel_size;
local border = 0;
if opt.border == 'valid' then
    border = 0;
else
    border = (kernelSize - 1)/2
end
local epoch = opt.epoch;
local maps = string.split(opt.neuron_num, '/');

-- construct the network
-- VolumetricConvolution is 3D convolution
-- VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH, padT, padW, padH])
-- @ parameters:
-- @ nInputPlane: The number of expected input planes in the image given into forward(); planes means channels
-- @ nOutputPlane: The number of output planes the convolution layer will produce.
-- @ kT: The kernel size of the convolution in time
-- @ kW: The kernel width of the convolution
-- @ kH: The kernel height of the convolution
-- @ dT: The step of the convolution in the time dimension. Default is 1.
-- @ dW: The step of the convolution in the width dimension. Default is 1.
-- @ dH: The step of the convolution in the height dimension. Default is 1.
-- @ padT: The additional zeros added per time to the input planes. Default is 0, a good number is (kT-1)/2.
-- @ padW: The additional zeros added per width to the input planes. Default is 0, a good number is (kW-1)/2.
-- @padH: The additional zeros added per height to the input planes. Default is 0, a good number is (kH-1)/2.

net = nn.Sequential()
net:add(cudnn.VolumetricConvolution(1, tonumber(maps[1]), kernelSize, kernelSize, kernelSize, 1, 1, 1, border, border, border))
net:add(cudnn.ReLU())
for i = 1, opt.layer_num - 1 do 
    net:add(cudnn.VolumetricConvolution(tonumber(maps[i]), tonumber(maps[i+1]), kernelSize, kernelSize, kernelSize, 1, 1, 1, border, border, border))
    net:add(cudnn.ReLU())
end
net:add(cudnn.VolumetricConvolution(tonumber(maps[opt.layer_num]), 1, kernelSize, kernelSize, kernelSize, 1, 1, 1, border, border, border))
--net:add(cudnn.ReLU())

-- training process
local params = {
   learningRate = opt.learning_rate;
}
-- define loss function: mean square
criterion = nn.MSECriterion()
-- change everthing into cuda type
net = net:cuda()
input_tensor = input_tensor:cuda();
target_tensor = target_tensor:cuda();
criterion = criterion:cuda()
-- crop to the middle first
input_tensor, target_tensor = Crop3DFun:crop(input_tensor, target_tensor, 1, true)
print("params:");
print(params);
print('Input File: ' .. input_file_name);
print('Target File: ' .. target_file_name);
print('Epoch: ' .. epoch);
print('My Net\n' .. net:__tostring());

x, dl_dx = net:getParameters();
feval = function(x_new)
    -- learnable parameters x and dloss/dx
    -- for torch, we need to initialize the gradients to 0 everytime
    if x ~= x_new then
        x:copy(x_new)
    end
    dl_dx:zero()
    -- random crop
    _input_tensor, _target_tensor = Crop3DFun:crop(input_tensor, target_tensor)
    -- generate random rotation matrix
    Q = Rot3DFun:genRotMat()
    -- rotation
    _input_tensor = Rot3DFun:rot(_input_tensor, Q);
    _target_tensor = Rot3DFun:rot(_target_tensor, Q);
    -- crop to the middle again
    _input_tensor, _target_tensor = Crop3DFun:crop(_input_tensor, _target_tensor, 1, true)
    local margin = 0;
    if opt.border == 'valid' then
        margin = opt.layer_num + 1 + 1;
        _target_tensor = _target_tensor[{{1},{margin,-margin},{margin,-margin},{margin,-margin}}]
    end
    --net:backward will modify the dl_dx
    loss_x = criterion:forward(net:forward(_input_tensor), _target_tensor)
    net:backward(_input_tensor, criterion:backward(net.output, _target_tensor))
    -- return loss(x) and dloss/dx
    return loss_x, dl_dx
end
 
for i = 1,opt.epoch  do
    -- closure function use for optim
    current_loss = 0
    if opt.optim == 'adam' then
    	_,fs = optim.adam(feval, x, params);
    else
	    _,fs = optim.sgd(feval, x, params);
    end
    -- Functions in optim all return two things:
    --   + the new x, found by the optimization method (here Adam)
    --   + the value of the loss functions at all points that were used by
    --     the algorithm.
    current_loss = fs[1]
    -- report error on epoch
    print('current loss = ' .. current_loss)
    -- add loss info to the log 
    logger:add{['training error'] = current_loss}
    logger:style{['training error'] = '-'}
    logger:plot()  
end

-- save output
final_output = net.output:double();
final_output = final_output:squeeze();
saveTime = os.date("%m") .. "-" .. os.date("%d") .. "_" .. os.date("%X");
output_file_name = './output/' .. saveTime .. '_train_output.mat';
print('Final output has been saved to ' .. output_file_name);
matio.save(output_file_name, final_output);
-- save model
model_file_name = './model/' .. saveTime .. '_model.t7';
-- claerState makes the model file smaller
net:clearState();
print('Final model has been saved to ' .. model_file_name);
torch.save(model_file_name, net);
