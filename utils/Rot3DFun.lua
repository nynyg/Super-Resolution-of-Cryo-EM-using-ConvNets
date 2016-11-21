--[[
  
  Name: Nan Yang
  Email: nan.yang@tum.de
  Project: Super Resolution of Cryo-Em
  Tutor: Vladimir Golkov
  Data: 08.08.2016
 
  This file is the used for 3D rotation

]]--

Rot3DFun = {}

-- i j k : x y z
-- S size of the square tensor
function Rot3DFun:sub2int(i,j,k,S)
    return (i - 1)*S*S + (j - 1)*S +k;
end

-- generate random rotation matrix
function Rot3DFun:genRotMat()
    local randn_matrix = torch.randn(3,3);
    local Q,R
    Q, R = torch.qr(randn_matrix);
    -- TODO: sign of diagonal of R
    local a = Q[1][1];
    local b = Q[1][2];
    local c = Q[1][3];
    local d = Q[2][1];
    local e = Q[2][2];
    local f = Q[2][3];
    local g = Q[3][1];
    local h = Q[3][2];
    local i = Q[3][3];
    local determinant = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    Q = Q * determinant;
    Q = Q:cuda();
    return Q
end

function Rot3DFun:mgrid(input_size)
    -- roatation center
    local endx = (input_size + 1) / 2;
    -- generate the range of coordinates centered at endX
    local coordinatesX = torch.range(1-endx, input_size - endx, 1):cuda();
    -- create the container of the coordinates
    local grid_coords = torch.Tensor(3, input_size, input_size, input_size):cuda();
    -- subTensor:
    -- grid_coords[{{1},{},{},{}}] = subTensor1
    -- grid_coords[{{2},{},{},{}}] = subTensor2
    -- grid_coords[{{3},{},{},{}}] = subTensor3
    local subTensor1 = torch.Tensor(input_size, input_size, input_size):fill(1):cuda();
    -- subTensor1:
    -- the first layer of this 3D subTensor is an input*input square matrix containing coordinate1( 1-endx in this case )
    -- the second layer of this 3D subTensor is an input*input square matrix containing coordinate2( 1-endx+1 )
    -- ...
    -- the last layer of this 3D subTensor is an input*input square matrix containing the last coordinate( input_size - endx here ) 
    for i = 1, input_size do
        subTensor1[{{i},{},{}}] = subTensor1[{{i},{},{}}] * coordinatesX[i];
    end
    grid_coords[{{1},{},{},{}}] = subTensor1;
    -- the coordinatesX before is 1*input_size. Here change it to input_size*1
    coordinatesX:resize(input_size, 1);
    -- expand is used to repliacte the tensor columns/rows to for a new tensor 
    local coordinatesXExpanded = torch.expand(coordinatesX,input_size, input_size);
    -- transpose
    local coordinatesXExpandedT = coordinatesXExpanded:t();
    subTensor2 = torch.Tensor(1, input_size, input_size);
    subTensor2 = subTensor2:double();
    coordinatesXExpanded = coordinatesXExpanded:double();
    subTensor2[{{1},{},{}}] = coordinatesXExpanded;
    subTensor2 = subTensor2:cuda();
    subTensor2 = torch.expand(subTensor2,input_size, input_size, input_size);
    subTensor3 = torch.Tensor(1, input_size, input_size);
    subTensor3 = subTensor3:double();
    coordinatesXExpandedT = coordinatesXExpandedT:double();
    subTensor3[{{1},{},{}}] = coordinatesXExpandedT;
    subTensor3 = subTensor3:cuda();
    subTensor3 = torch.expand(subTensor3,input_size, input_size, input_size);
    grid_coords[{{1},{},{},{}}] = subTensor1;
    grid_coords[{{2},{},{},{}}] = subTensor2;
    grid_coords[{{3},{},{},{}}] = subTensor3;
    -- clear memory
    subTensor1 = nil;
    subTensor2 = nil;
    subTensor3 = nil;
    coordinatesX = nil;
    coordinatesXExpanded = nil;
    coordinatesXExpandedT = nil;
    return grid_coords
end


function Rot3DFun:rot(input, Q)
    local input = input:cuda();
    local input = input:squeeze();
    local input_size = input:size(1);
    -- roatation center
    local endx = (input_size + 1) / 2;
    -- TODO: make mgrid called only once 
    -- get coordinates
    local grid_coords = self:mgrid(input_size);
    -- rotation
    local rot_coords = Q * grid_coords:resize(3,input_size*input_size*input_size):cuda();
    rot_coords:resize(3, input_size, input_size, input_size);
    grid_coords:resize(3, input_size, input_size, input_size);
    rot_coords = rot_coords + endx;
    -- do trilinear interpolation
    local thex = rot_coords[{{1},{},{},{}}]:squeeze();
    local they = rot_coords[{{2},{},{},{}}]:squeeze();
    local thez = rot_coords[{{3},{},{},{}}]:squeeze();
    local x = torch.floor(thex):cuda();
    local y = torch.floor(they):cuda();
    local z = torch.floor(thez):cuda();
    local X = x + 1;
    local Y = y + 1;
    local Z = z + 1;
    local deltax = thex-x;
    local deltay = they-y;
    local deltaz = thez-z;
    local deltaX = 1-deltax;
    local deltaY = 1-deltay;
    local deltaZ = 1-deltaz;

    -- clip
    x[x:gt(input_size)] = input_size;
    y[y:gt(input_size)] = input_size;
    z[z:gt(input_size)] = input_size;
    X[X:gt(input_size)] = input_size;
    Y[Y:gt(input_size)] = input_size;
    Z[Z:gt(input_size)] = input_size;

    x[x:lt(1)] = 1;
    y[y:lt(1)] = 1;
    z[z:lt(1)] = 1;
    X[X:lt(1)] = 1;
    Y[Y:lt(1)] = 1;
    Z[Z:lt(1)] = 1;

    x:resize(input_size*input_size*input_size, 1);
    y:resize(input_size*input_size*input_size, 1);
    z:resize(input_size*input_size*input_size, 1);
    X:resize(input_size*input_size*input_size, 1);
    Y:resize(input_size*input_size*input_size, 1);
    Z:resize(input_size*input_size*input_size, 1);

    -- torch cannot indexing by array, so vectorizing the tensor
    -- and using gather to implement indexing by array 
    -- https://github.com/torch/torch7/blob/master/doc/tensor.md
    input:resize(input_size*input_size*input_size, 1);
    local xyz = input:gather(1, self:sub2int(x, y, z, input_size));
    local xyZ = input:gather(1, self:sub2int(x, y, Z, input_size));
    local xYz = input:gather(1, self:sub2int(x, Y, z, input_size));
    local xYZ = input:gather(1, self:sub2int(x, Y, Z, input_size));
    local Xyz = input:gather(1, self:sub2int(X, y, z, input_size));
    local XyZ = input:gather(1, self:sub2int(X, y, Z, input_size));
    local XYz = input:gather(1, self:sub2int(X, Y, z, input_size));
    local XYZ = input:gather(1, self:sub2int(X, Y, Z, input_size));

   
    xyz:resize(input_size,input_size,input_size):cuda();   
    xyZ:resize(input_size,input_size,input_size):cuda();   
    xYz:resize(input_size,input_size,input_size):cuda();   
    xYZ:resize(input_size,input_size,input_size):cuda();   
    Xyz:resize(input_size,input_size,input_size):cuda(); 
    XyZ:resize(input_size,input_size,input_size):cuda();
    XYz:resize(input_size,input_size,input_size):cuda();
    XYZ:resize(input_size,input_size,input_size):cuda();

    local deltaXYZ = torch.cmul(torch.cmul(deltaX, deltaY),deltaZ):cuda(); 
    local deltaXYz = torch.cmul(torch.cmul(deltaX, deltaY),deltaz):cuda();
    local deltaXyZ = torch.cmul(torch.cmul(deltaX, deltay),deltaZ):cuda();
    local deltaXyz = torch.cmul(torch.cmul(deltaX, deltay),deltaz):cuda();
    local deltaxYZ = torch.cmul(torch.cmul(deltax, deltaY),deltaZ):cuda();
    local deltaxYz = torch.cmul(torch.cmul(deltax, deltaY),deltaz):cuda();
    local deltaxyZ = torch.cmul(torch.cmul(deltax, deltay),deltaZ):cuda();
    local deltaxyz = torch.cmul(torch.cmul(deltax, deltay),deltaz):cuda();

    local result = torch.cmul(deltaXYZ, xyz)+torch.cmul(deltaXYz,xyZ)+
             torch.cmul(deltaXyZ,xYz)+torch.cmul(deltaXyz,xYZ)+
             torch.cmul(deltaxYZ,Xyz)+torch.cmul(deltaxYz,XyZ)+
             torch.cmul(deltaxyZ,XYz)+torch.cmul(deltaxyz,XYZ);
    _result = torch.zeros(1,input_size,input_size,input_size):cuda()
    _result[{{1},{},{},{}}] = result;
    return _result
end

