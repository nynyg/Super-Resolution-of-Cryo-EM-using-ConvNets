--[[
 
  Name: Nan Yang
  Email: nan.yang@tum.de
  Project: Super Resolution of Cryo-Em
  Tutor: Vladimir Golkov
  Data: 08.08.2016
  
  This file is used for testing new data
]]--
require 'torch'
require 'cunn'
require 'cudnn'
local matio = require 'matio'

input_tensor = torch.zeros(1,180,180,180);

--input = matio.load('../data/train/1/un3000.mat')
input = matio.load('./un3000.mat')
--input = matio.load('../data/train/2/filtered50k.mat')
--input = matio.load('./filtered50k.mat')
--input = matio.load('../data/train/1/3k.mat')
--input = input.f_map3
input = input.un3000
--input = input.filtered50k
-- input = input.filtered-3k
input_tensor[{{1},{},{},{}}] = input
input_tensor = input_tensor:cuda()

-- net = torch.load('./model/07-14_04:49:07_model.t7')
-- net = torch.load('./model/64-1024.t7')
net = torch.load('./model/final.t7')
--net = torch.load('./model/3layers.t7')

local batch_size = 30;
local stride = 0;
local test = net:forward(input_tensor[{{1},{1,batch_size},{1,batch_size},{1,batch_size}}]);
stride = test:size()[2];
test = nil;
local margin = batch_size - stride;
local output_size = 180 - margin;
output_tensor = torch.zeros(1, output_size, output_size, output_size):cuda();
local it = torch.ceil((180 - margin) / stride)

-- commented for loop is used for invalid border mode
--[[
for i = 1, 180, 45 do
    for j = 1, 180, 45 do
        for k = 1, 180, 45 do
            tmp = net:forward(input_tensor[{{1},{i,i+44},{j,j+44},{k,k+44}}])
	    output_tensor[{{1},{(i+44)/45,(i+44)/45+test_size-1},{(j+44)/45,(j+44)/45+test_size-1},{(k+44)/45,(k+44)/45+test_size-1}}] = tmp
	end
    end
end
]]--

-- valid border mode testing
for i = 1, it do
    i_begin = ( i - 1 ) * stride + 1;
    for j = 1, it do
        j_begin = ( j - 1) * stride + 1;
        for k = 1, it do 
            k_begin = ( k - 1 ) * stride + 1;
            tmp = net:forward(input_tensor[{{1},{i_begin, math.min(i_begin + batch_size - 1, 180)},{j_begin, math.min(j_begin + batch_size - 1, 180)},{k_begin, math.min(k_begin + batch_size - 1, 180)}}]) 
            output_tensor[{{1},{i_begin, math.min(i_begin + stride - 1, output_size)},{j_begin, math.min(j_begin + stride - 1, output_size)},{k_begin, math.min(k_begin + stride - 1, output_size)}}] = tmp
        end
    end
end

output_tensor = output_tensor:float():squeeze();

matio.save('./output/result.mat', output_tensor);
