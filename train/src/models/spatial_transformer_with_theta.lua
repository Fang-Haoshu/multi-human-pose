require 'stn'

   local inp1 = nn.Identity()()
   
   -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
   local tranet=nn.Transpose({2,3},{3,4})(inp1)
   
   -- second branch is the localization network
   local conv1 = nnlib.SpatialConvolution(3,64,3,3)(inp1)
   local conv1_bn = nn.SpatialBatchNormalization(64)(conv1)
   local conv1_relu = nnlib.ReLU(true)(conv1_bn)  
   local pool1 = nnlib.SpatialMaxPooling(3,3,2,2):ceil()(conv1_relu)
   local fire2_squeeze1x1 = nnlib.SpatialConvolution(64,16,1,1)(pool1)
   local fire2_squeeze1x1_bn = nnlib.SpatialBatchNormalization(16)(fire2_squeeze1x1)
   local fire2_squeeze1x1_relu = nnlib.ReLU(true)(fire2_squeeze1x1_bn)

   local fire2_expand1x1 = nnlib.SpatialConvolution(16,64,1,1)(fire2_squeeze1x1_relu)
   local fire2_expand1x1_bn = nnlib.SpatialBatchNormalization(64)(fire2_expand1x1)
   local fire2_expand1x1_relu = nnlib.ReLU(true)(fire2_expand1x1_bn)

   local fire2_expand3x3 = nnlib.SpatialConvolution(16,64,3,3,1,1,1,1)(fire2_squeeze1x1_relu)
   local fire2_expand3x3_bn = nnlib.SpatialBatchNormalization(64)(fire2_expand3x3)
   local fire2_expand3x3_relu = nnlib.ReLU(true)(fire2_expand3x3_bn)

   local fire2_concat = nn.JoinTable(2)({fire2_expand1x1_relu,fire2_expand3x3_relu})

   local fire3_squeeze1x1 = nnlib.SpatialConvolution(128,16,1,1)(fire2_concat)
   local fire3_squeeze1x1_bn = nnlib.SpatialBatchNormalization(16)(fire3_squeeze1x1)
   local fire3_squeeze1x1_relu = nnlib.ReLU(true)(fire3_squeeze1x1_bn)

   local fire3_expand1x1 = nnlib.SpatialConvolution(16,64,1,1)(fire3_squeeze1x1_relu)
   local fire3_expand1x1_bn = nnlib.SpatialBatchNormalization(64)(fire3_expand1x1)
   local fire3_expand1x1_relu = nnlib.ReLU(true)(fire3_expand1x1_bn)

   local fire3_expand3x3 = nnlib.SpatialConvolution(16,64,3,3,1,1,1,1)(fire3_squeeze1x1_relu)
   local fire3_expand3x3_bn = nnlib.SpatialBatchNormalization(64)(fire3_expand3x3)
   local fire3_expand3x3_relu = nnlib.ReLU(true)(fire3_expand3x3_bn)

   local fire3_concat = nn.JoinTable(2)({fire3_expand1x1_relu,fire3_expand3x3_relu})
   
   local pool3 = nnlib.SpatialMaxPooling(3,3,2,2):ceil()(fire3_concat)

   local fire4_squeeze1x1 = nnlib.SpatialConvolution(128,32,1,1)(pool3)
   local fire4_squeeze1x1_bn = nnlib.SpatialBatchNormalization(32)(fire4_squeeze1x1)
   local fire4_squeeze1x1_relu = nnlib.ReLU(true)(fire4_squeeze1x1_bn)

   local fire4_expand1x1 = nnlib.SpatialConvolution(32,128,1,1)(fire4_squeeze1x1_relu)
   local fire4_expand1x1_bn = nnlib.SpatialBatchNormalization(128)(fire4_expand1x1)
   local fire4_expand1x1_relu = nnlib.ReLU(true)(fire4_expand1x1_bn)

   local fire4_expand3x3 = nnlib.SpatialConvolution(32,128,3,3,1,1,1,1)(fire4_squeeze1x1_relu)
   local fire4_expand3x3_bn = nnlib.SpatialBatchNormalization(128)(fire4_expand3x3)
   local fire4_expand3x3_relu = nnlib.ReLU(true)(fire4_expand3x3_bn)

   local fire4_concat = nn.JoinTable(2)({fire4_expand1x1_relu,fire4_expand3x3_relu})

   local fire5_squeeze1x1 = nnlib.SpatialConvolution(256,32,1,1)(fire4_concat)
   local fire5_squeeze1x1_bn = nnlib.SpatialBatchNormalization(32)(fire5_squeeze1x1)
   local fire5_squeeze1x1_relu = nnlib.ReLU(true)(fire5_squeeze1x1_bn)

   local fire5_expand1x1 = nnlib.SpatialConvolution(32,128,1,1)(fire5_squeeze1x1_relu)
   local fire5_expand1x1_bn = nnlib.SpatialBatchNormalization(128)(fire5_expand1x1)
   local fire5_expand1x1_relu = nnlib.ReLU(true)(fire5_expand1x1_bn)

   local fire5_expand3x3 = nnlib.SpatialConvolution(32,128,3,3,1,1,1,1)(fire5_squeeze1x1_relu)
   local fire5_expand3x3_bn = nnlib.SpatialBatchNormalization(128)(fire5_expand3x3)
   local fire5_expand3x3_relu = nnlib.ReLU(true)(fire5_expand3x3_bn)

   local fire5_concat = nn.JoinTable(2)({fire5_expand1x1_relu,fire5_expand3x3_relu})
   
   local pool5 = nnlib.SpatialMaxPooling(3,3,2,2):ceil()(fire5_concat)

   local fire6_squeeze1x1 = nnlib.SpatialConvolution(256,48,1,1)(pool5)
   local fire6_squeeze1x1_bn = nnlib.SpatialBatchNormalization(48)(fire6_squeeze1x1)
   local fire6_squeeze1x1_relu = nnlib.ReLU(true)(fire6_squeeze1x1_bn)

   local fire6_expand1x1 = nnlib.SpatialConvolution(48,192,1,1)(fire6_squeeze1x1_relu)
   local fire6_expand1x1_bn = nnlib.SpatialBatchNormalization(192)(fire6_expand1x1)
   local fire6_expand1x1_relu = nnlib.ReLU(true)(fire6_expand1x1_bn)

   local fire6_expand3x3 = nnlib.SpatialConvolution(48,192,3,3,1,1,1,1)(fire6_squeeze1x1_relu)
   local fire6_expand3x3_bn = nnlib.SpatialBatchNormalization(192)(fire6_expand3x3)
   local fire6_expand3x3_relu = nnlib.ReLU(true)(fire6_expand3x3_bn)

   local fire6_concat = nn.JoinTable(2)({fire6_expand1x1_relu,fire6_expand3x3_relu})

   local fire7_squeeze1x1 = nnlib.SpatialConvolution(384,48,1,1)(fire6_concat)
   local fire7_squeeze1x1_bn = nnlib.SpatialBatchNormalization(48)(fire7_squeeze1x1)
   local fire7_squeeze1x1_relu = nnlib.ReLU(true)(fire7_squeeze1x1_bn)

   local fire7_expand1x1 = nnlib.SpatialConvolution(48,192,1,1)(fire7_squeeze1x1_relu)
   local fire7_expand1x1_bn = nnlib.SpatialBatchNormalization(192)(fire7_expand1x1)
   local fire7_expand1x1_relu = nnlib.ReLU(true)(fire7_expand1x1_bn)

   local fire7_expand3x3 = nnlib.SpatialConvolution(48,192,3,3,1,1,1,1)(fire7_squeeze1x1_relu)
   local fire7_expand3x3_bn = nnlib.SpatialBatchNormalization(192)(fire7_expand3x3)
   local fire7_expand3x3_relu = nnlib.ReLU(true)(fire7_expand3x3_bn)

   local fire7_concat = nn.JoinTable(2)({fire7_expand1x1_relu,fire7_expand3x3_relu})
   
   local fire8_squeeze1x1 = nnlib.SpatialConvolution(384,64,1,1)(fire7_concat)
   local fire8_squeeze1x1_bn = nnlib.SpatialBatchNormalization(64)(fire8_squeeze1x1)
   local fire8_squeeze1x1_relu = nnlib.ReLU(true)(fire8_squeeze1x1_bn)

   local fire8_expand1x1 = nnlib.SpatialConvolution(64,256,1,1)(fire8_squeeze1x1_relu)
   local fire8_expand1x1_bn = nnlib.SpatialBatchNormalization(256)(fire8_expand1x1)
   local fire8_expand1x1_relu = nnlib.ReLU(true)(fire8_expand1x1_bn)

   local fire8_expand3x3 = nnlib.SpatialConvolution(64,256,3,3,1,1,1,1)(fire8_squeeze1x1_relu)
   local fire8_expand3x3_bn = nnlib.SpatialBatchNormalization(256)(fire8_expand3x3)
   local fire8_expand3x3_relu = nnlib.ReLU(true)(fire8_expand3x3_bn)

   local fire8_concat = nn.JoinTable(2)({fire8_expand1x1_relu,fire8_expand3x3_relu})

   local fire9_squeeze1x1 = nnlib.SpatialConvolution(512,64,1,1)(fire8_concat)
   local fire9_squeeze1x1_bn = nnlib.SpatialBatchNormalization(64)(fire9_squeeze1x1)
   local fire9_squeeze1x1_relu = nnlib.ReLU(true)(fire9_squeeze1x1_bn)

   local fire9_expand1x1 = nnlib.SpatialConvolution(64,256,1,1)(fire9_squeeze1x1_relu)
   local fire9_expand1x1_bn = nnlib.SpatialBatchNormalization(256)(fire9_expand1x1)
   local fire9_expand1x1_relu = nnlib.ReLU(true)(fire9_expand1x1_bn)

   local fire9_expand3x3 = nnlib.SpatialConvolution(64,256,3,3,1,1,1,1)(fire9_squeeze1x1_relu)
   local fire9_expand3x3_bn = nnlib.SpatialBatchNormalization(256)(fire9_expand3x3)
   local fire9_expand3x3_relu = nnlib.ReLU(true)(fire9_expand3x3_bn)

   local fire9_concat = nn.JoinTable(2)({fire9_expand1x1_relu,fire9_expand3x3_relu})
   local fire9_dropout = nn.SpatialDropout(0.5)(fire9_concat)

   local conv10 = nnlib.SpatialConvolution(512,6,1,1)(fire9_dropout)
   local conv10_bn = nnlib.SpatialBatchNormalization(6)(conv10)
   local conv10_relu = nnlib.ReLU(true)(conv10_bn)  

   local reshape1 = nn.View(6*31*31)(conv10_relu)

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(6*31*31,6)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(6):fill(0)
bias[1]=1
bias[5]=1
outLayer.bias:copy(bias)

   -- there we generate the grids
   
   --locnet = locnet(inp)
   local theta = nn.View(2,3)(outLayer(reshape1))
   local grids = nn.AffineGridGeneratorBHWD(256,256)(theta)
   
   -- we need a table input for the bilinear sampler, so we use concattable
   
   local Sampling = nn.BilinearSamplerBHWD()({tranet,grids})
   local img = nn.Transpose({3,4},{2,3})(Sampling)
   
   -- and we transpose back to standard BDHW format for subsequent processing by nn modules
   spanet=nn.gModule({inp1},{img,theta})
