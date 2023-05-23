import torch
from torch import nn
from model.mrf import MRF

'''
class Generator(nn.Module):

def __init__(self, input_channel=80, hu=512, ku=[16, 16, 4, 4], kr=[3, 7, 11], Dr=[1, 3, 5]):
    super(Generator, self).__init__()
    generator = []
    generator += [
        nn.ReflectionPad1d(3),
        nn.utils.weight_norm(nn.Conv1d(input_channel, hu, kernel_size=7))
    ]

    
    
    for k in ku:
        inp = hu
        out = int(inp/2)
        generator += [
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(inp, out, k, k//2),
            MRF(kr, out, Dr)
        ]
        hu = out

    generator += [
        nn.LeakyReLU(0.2),
        nn.ReflectionPad1d(3),
        nn.utils.weight_norm(nn.Conv1d(hu, 1, kernel_size=7, stride=1)),
        nn.Tanh()
    ]
    self.generator = nn.Sequential(*generator)

    

def forward(self, x):
    x = self.generator(x)
    return x
 
'''

class Generator(nn.Module):
    
    def __init__(self, input_channel=80, hu=512, ku=[16, 16, 4, 4], kr=[3, 7, 11], Dr=[1, 3, 5]):
        super(Generator, self).__init__()
        self.input = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(input_channel, hu, kernel_size=7))
        )

        generator = []
        
        for k in ku:
            inp = hu
            out = int(inp/2)
            generator += [
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.ConvTranspose1d(inp, out, k, k//2)),
                MRF(kr, out, Dr)
            ]
            hu = out
        self.generator = nn.Sequential(*generator)

        self.output = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(hu, 1, kernel_size=7, stride=1)),
            nn.Tanh()

        )
    
    def forward(self, x):
        x1 = self.input(x)
        x2 = self.generator(x1)
        out = self.output(x2)
        return out

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    # def remove_weight_norm(self):
    #     for idx, layer in enumerate(self.generator):
    #         if len(layer.state_dict()) != 0:
    #             try:
    #                 nn.utils.remove_weight_norm(layer)
    #             except:
    #                 layer.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.input):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

        for idx, layer in enumerate(self.output):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def inference(self, mel):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.input_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        return audio


'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
   ReflectionPad1d-1              [-1, 80, 506]               0
            Conv1d-2             [-1, 512, 500]         287,232
         LeakyReLU-3             [-1, 512, 500]               0
   ConvTranspose1d-4            [-1, 256, 4008]       2,097,408
            Conv1d-5            [-1, 256, 4008]          65,792
         LeakyReLU-6            [-1, 256, 4008]               0
   ReflectionPad1d-7            [-1, 256, 4010]               0
            Conv1d-8            [-1, 256, 4008]         196,864
         LeakyReLU-9            [-1, 256, 4008]               0
  ReflectionPad1d-10            [-1, 256, 4008]               0
           Conv1d-11            [-1, 256, 4008]          65,792
        LeakyReLU-12            [-1, 256, 4008]               0
  ReflectionPad1d-13            [-1, 256, 4014]               0
           Conv1d-14            [-1, 256, 4008]         196,864
        LeakyReLU-15            [-1, 256, 4008]               0
  ReflectionPad1d-16            [-1, 256, 4008]               0
           Conv1d-17            [-1, 256, 4008]          65,792
        LeakyReLU-18            [-1, 256, 4008]               0
  ReflectionPad1d-19            [-1, 256, 4018]               0
           Conv1d-20            [-1, 256, 4008]         196,864
        LeakyReLU-21            [-1, 256, 4008]               0
  ReflectionPad1d-22            [-1, 256, 4008]               0
           Conv1d-23            [-1, 256, 4008]          65,792
         ResStack-24            [-1, 256, 4008]               0
           Conv1d-25            [-1, 256, 4008]          65,792
        LeakyReLU-26            [-1, 256, 4008]               0
  ReflectionPad1d-27            [-1, 256, 4010]               0
           Conv1d-28            [-1, 256, 4004]         459,008
        LeakyReLU-29            [-1, 256, 4004]               0
  ReflectionPad1d-30            [-1, 256, 4016]               0
           Conv1d-31            [-1, 256, 4016]          65,792
        LeakyReLU-32            [-1, 256, 4016]               0
  ReflectionPad1d-33            [-1, 256, 4022]               0
           Conv1d-34            [-1, 256, 4004]         459,008
        LeakyReLU-35            [-1, 256, 4004]               0
  ReflectionPad1d-36            [-1, 256, 4016]               0
           Conv1d-37            [-1, 256, 4016]          65,792
        LeakyReLU-38            [-1, 256, 4016]               0
  ReflectionPad1d-39            [-1, 256, 4026]               0
           Conv1d-40            [-1, 256, 3996]         459,008
        LeakyReLU-41            [-1, 256, 3996]               0
  ReflectionPad1d-42            [-1, 256, 4008]               0
           Conv1d-43            [-1, 256, 4008]          65,792
         ResStack-44            [-1, 256, 4008]               0
           Conv1d-45            [-1, 256, 4008]          65,792
        LeakyReLU-46            [-1, 256, 4008]               0
  ReflectionPad1d-47            [-1, 256, 4010]               0
           Conv1d-48            [-1, 256, 4000]         721,152
        LeakyReLU-49            [-1, 256, 4000]               0
  ReflectionPad1d-50            [-1, 256, 4024]               0
           Conv1d-51            [-1, 256, 4024]          65,792
        LeakyReLU-52            [-1, 256, 4024]               0
  ReflectionPad1d-53            [-1, 256, 4030]               0
           Conv1d-54            [-1, 256, 4000]         721,152
        LeakyReLU-55            [-1, 256, 4000]               0
  ReflectionPad1d-56            [-1, 256, 4024]               0
           Conv1d-57            [-1, 256, 4024]          65,792
        LeakyReLU-58            [-1, 256, 4024]               0
  ReflectionPad1d-59            [-1, 256, 4034]               0
           Conv1d-60            [-1, 256, 3984]         721,152
        LeakyReLU-61            [-1, 256, 3984]               0
  ReflectionPad1d-62            [-1, 256, 4008]               0
           Conv1d-63            [-1, 256, 4008]          65,792
         ResStack-64            [-1, 256, 4008]               0
              MRF-65            [-1, 256, 4008]               0
        LeakyReLU-66            [-1, 256, 4008]               0
  ConvTranspose1d-67           [-1, 128, 32072]         524,416
           Conv1d-68           [-1, 128, 32072]          16,512
        LeakyReLU-69           [-1, 128, 32072]               0
  ReflectionPad1d-70           [-1, 128, 32074]               0
           Conv1d-71           [-1, 128, 32072]          49,280
        LeakyReLU-72           [-1, 128, 32072]               0
  ReflectionPad1d-73           [-1, 128, 32072]               0
           Conv1d-74           [-1, 128, 32072]          16,512
        LeakyReLU-75           [-1, 128, 32072]               0
  ReflectionPad1d-76           [-1, 128, 32078]               0
           Conv1d-77           [-1, 128, 32072]          49,280
        LeakyReLU-78           [-1, 128, 32072]               0
  ReflectionPad1d-79           [-1, 128, 32072]               0
           Conv1d-80           [-1, 128, 32072]          16,512
        LeakyReLU-81           [-1, 128, 32072]               0
  ReflectionPad1d-82           [-1, 128, 32082]               0
           Conv1d-83           [-1, 128, 32072]          49,280
        LeakyReLU-84           [-1, 128, 32072]               0
  ReflectionPad1d-85           [-1, 128, 32072]               0
           Conv1d-86           [-1, 128, 32072]          16,512
         ResStack-87           [-1, 128, 32072]               0
           Conv1d-88           [-1, 128, 32072]          16,512
        LeakyReLU-89           [-1, 128, 32072]               0
  ReflectionPad1d-90           [-1, 128, 32074]               0
           Conv1d-91           [-1, 128, 32068]         114,816
        LeakyReLU-92           [-1, 128, 32068]               0
  ReflectionPad1d-93           [-1, 128, 32080]               0
           Conv1d-94           [-1, 128, 32080]          16,512
        LeakyReLU-95           [-1, 128, 32080]               0
  ReflectionPad1d-96           [-1, 128, 32086]               0
           Conv1d-97           [-1, 128, 32068]         114,816
        LeakyReLU-98           [-1, 128, 32068]               0
  ReflectionPad1d-99           [-1, 128, 32080]               0
          Conv1d-100           [-1, 128, 32080]          16,512
       LeakyReLU-101           [-1, 128, 32080]               0
 ReflectionPad1d-102           [-1, 128, 32090]               0
          Conv1d-103           [-1, 128, 32060]         114,816
       LeakyReLU-104           [-1, 128, 32060]               0
 ReflectionPad1d-105           [-1, 128, 32072]               0
          Conv1d-106           [-1, 128, 32072]          16,512
        ResStack-107           [-1, 128, 32072]               0
          Conv1d-108           [-1, 128, 32072]          16,512
       LeakyReLU-109           [-1, 128, 32072]               0
 ReflectionPad1d-110           [-1, 128, 32074]               0
          Conv1d-111           [-1, 128, 32064]         180,352
       LeakyReLU-112           [-1, 128, 32064]               0
 ReflectionPad1d-113           [-1, 128, 32088]               0
          Conv1d-114           [-1, 128, 32088]          16,512
       LeakyReLU-115           [-1, 128, 32088]               0
 ReflectionPad1d-116           [-1, 128, 32094]               0
          Conv1d-117           [-1, 128, 32064]         180,352
       LeakyReLU-118           [-1, 128, 32064]               0
 ReflectionPad1d-119           [-1, 128, 32088]               0
          Conv1d-120           [-1, 128, 32088]          16,512
       LeakyReLU-121           [-1, 128, 32088]               0
 ReflectionPad1d-122           [-1, 128, 32098]               0
          Conv1d-123           [-1, 128, 32048]         180,352
       LeakyReLU-124           [-1, 128, 32048]               0
 ReflectionPad1d-125           [-1, 128, 32072]               0
          Conv1d-126           [-1, 128, 32072]          16,512
        ResStack-127           [-1, 128, 32072]               0
             MRF-128           [-1, 128, 32072]               0
       LeakyReLU-129           [-1, 128, 32072]               0
 ConvTranspose1d-130            [-1, 64, 64146]          32,832
          Conv1d-131            [-1, 64, 64146]           4,160
       LeakyReLU-132            [-1, 64, 64146]               0
 ReflectionPad1d-133            [-1, 64, 64148]               0
          Conv1d-134            [-1, 64, 64146]          12,352
       LeakyReLU-135            [-1, 64, 64146]               0
 ReflectionPad1d-136            [-1, 64, 64146]               0
          Conv1d-137            [-1, 64, 64146]           4,160
       LeakyReLU-138            [-1, 64, 64146]               0
 ReflectionPad1d-139            [-1, 64, 64152]               0
          Conv1d-140            [-1, 64, 64146]          12,352
       LeakyReLU-141            [-1, 64, 64146]               0
 ReflectionPad1d-142            [-1, 64, 64146]               0
          Conv1d-143            [-1, 64, 64146]           4,160
       LeakyReLU-144            [-1, 64, 64146]               0
 ReflectionPad1d-145            [-1, 64, 64156]               0
          Conv1d-146            [-1, 64, 64146]          12,352
       LeakyReLU-147            [-1, 64, 64146]               0
 ReflectionPad1d-148            [-1, 64, 64146]               0
          Conv1d-149            [-1, 64, 64146]           4,160
        ResStack-150            [-1, 64, 64146]               0
          Conv1d-151            [-1, 64, 64146]           4,160
       LeakyReLU-152            [-1, 64, 64146]               0
 ReflectionPad1d-153            [-1, 64, 64148]               0
          Conv1d-154            [-1, 64, 64142]          28,736
       LeakyReLU-155            [-1, 64, 64142]               0
 ReflectionPad1d-156            [-1, 64, 64154]               0
          Conv1d-157            [-1, 64, 64154]           4,160
       LeakyReLU-158            [-1, 64, 64154]               0
 ReflectionPad1d-159            [-1, 64, 64160]               0
          Conv1d-160            [-1, 64, 64142]          28,736
       LeakyReLU-161            [-1, 64, 64142]               0
 ReflectionPad1d-162            [-1, 64, 64154]               0
          Conv1d-163            [-1, 64, 64154]           4,160
       LeakyReLU-164            [-1, 64, 64154]               0
 ReflectionPad1d-165            [-1, 64, 64164]               0
          Conv1d-166            [-1, 64, 64134]          28,736
       LeakyReLU-167            [-1, 64, 64134]               0
 ReflectionPad1d-168            [-1, 64, 64146]               0
          Conv1d-169            [-1, 64, 64146]           4,160
        ResStack-170            [-1, 64, 64146]               0
          Conv1d-171            [-1, 64, 64146]           4,160
       LeakyReLU-172            [-1, 64, 64146]               0
 ReflectionPad1d-173            [-1, 64, 64148]               0
          Conv1d-174            [-1, 64, 64138]          45,120
       LeakyReLU-175            [-1, 64, 64138]               0
 ReflectionPad1d-176            [-1, 64, 64162]               0
          Conv1d-177            [-1, 64, 64162]           4,160
       LeakyReLU-178            [-1, 64, 64162]               0
 ReflectionPad1d-179            [-1, 64, 64168]               0
          Conv1d-180            [-1, 64, 64138]          45,120
       LeakyReLU-181            [-1, 64, 64138]               0
 ReflectionPad1d-182            [-1, 64, 64162]               0
          Conv1d-183            [-1, 64, 64162]           4,160
       LeakyReLU-184            [-1, 64, 64162]               0
 ReflectionPad1d-185            [-1, 64, 64172]               0
          Conv1d-186            [-1, 64, 64122]          45,120
       LeakyReLU-187            [-1, 64, 64122]               0
 ReflectionPad1d-188            [-1, 64, 64146]               0
          Conv1d-189            [-1, 64, 64146]           4,160
        ResStack-190            [-1, 64, 64146]               0
             MRF-191            [-1, 64, 64146]               0
       LeakyReLU-192            [-1, 64, 64146]               0
 ConvTranspose1d-193           [-1, 32, 128294]           8,224
          Conv1d-194           [-1, 32, 128294]           1,056
       LeakyReLU-195           [-1, 32, 128294]               0
 ReflectionPad1d-196           [-1, 32, 128296]               0
          Conv1d-197           [-1, 32, 128294]           3,104
       LeakyReLU-198           [-1, 32, 128294]               0
 ReflectionPad1d-199           [-1, 32, 128294]               0
          Conv1d-200           [-1, 32, 128294]           1,056
       LeakyReLU-201           [-1, 32, 128294]               0
 ReflectionPad1d-202           [-1, 32, 128300]               0
          Conv1d-203           [-1, 32, 128294]           3,104
       LeakyReLU-204           [-1, 32, 128294]               0
 ReflectionPad1d-205           [-1, 32, 128294]               0
          Conv1d-206           [-1, 32, 128294]           1,056
       LeakyReLU-207           [-1, 32, 128294]               0
 ReflectionPad1d-208           [-1, 32, 128304]               0
          Conv1d-209           [-1, 32, 128294]           3,104
       LeakyReLU-210           [-1, 32, 128294]               0
 ReflectionPad1d-211           [-1, 32, 128294]               0
          Conv1d-212           [-1, 32, 128294]           1,056
        ResStack-213           [-1, 32, 128294]               0
          Conv1d-214           [-1, 32, 128294]           1,056
       LeakyReLU-215           [-1, 32, 128294]               0
 ReflectionPad1d-216           [-1, 32, 128296]               0
          Conv1d-217           [-1, 32, 128290]           7,200
       LeakyReLU-218           [-1, 32, 128290]               0
 ReflectionPad1d-219           [-1, 32, 128302]               0
          Conv1d-220           [-1, 32, 128302]           1,056
       LeakyReLU-221           [-1, 32, 128302]               0
 ReflectionPad1d-222           [-1, 32, 128308]               0
          Conv1d-223           [-1, 32, 128290]           7,200
       LeakyReLU-224           [-1, 32, 128290]               0
 ReflectionPad1d-225           [-1, 32, 128302]               0
          Conv1d-226           [-1, 32, 128302]           1,056
       LeakyReLU-227           [-1, 32, 128302]               0
 ReflectionPad1d-228           [-1, 32, 128312]               0
          Conv1d-229           [-1, 32, 128282]           7,200
       LeakyReLU-230           [-1, 32, 128282]               0
 ReflectionPad1d-231           [-1, 32, 128294]               0
          Conv1d-232           [-1, 32, 128294]           1,056
        ResStack-233           [-1, 32, 128294]               0
          Conv1d-234           [-1, 32, 128294]           1,056
       LeakyReLU-235           [-1, 32, 128294]               0
 ReflectionPad1d-236           [-1, 32, 128296]               0
          Conv1d-237           [-1, 32, 128286]          11,296
       LeakyReLU-238           [-1, 32, 128286]               0
 ReflectionPad1d-239           [-1, 32, 128310]               0
          Conv1d-240           [-1, 32, 128310]           1,056
       LeakyReLU-241           [-1, 32, 128310]               0
 ReflectionPad1d-242           [-1, 32, 128316]               0
          Conv1d-243           [-1, 32, 128286]          11,296
       LeakyReLU-244           [-1, 32, 128286]               0
 ReflectionPad1d-245           [-1, 32, 128310]               0
          Conv1d-246           [-1, 32, 128310]           1,056
       LeakyReLU-247           [-1, 32, 128310]               0
 ReflectionPad1d-248           [-1, 32, 128320]               0
          Conv1d-249           [-1, 32, 128270]          11,296
       LeakyReLU-250           [-1, 32, 128270]               0
 ReflectionPad1d-251           [-1, 32, 128294]               0
          Conv1d-252           [-1, 32, 128294]           1,056
        ResStack-253           [-1, 32, 128294]               0
             MRF-254           [-1, 32, 128294]               0
       LeakyReLU-255           [-1, 32, 128294]               0
 ReflectionPad1d-256           [-1, 32, 128300]               0
          Conv1d-257            [-1, 1, 128294]             225
            Tanh-258            [-1, 1, 128294]               0
================================================================
Total params: 9,488,417
Trainable params: 9,488,417
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.15
Forward/backward pass size (MB): 6450.82
Params size (MB): 36.20
Estimated Total Size (MB): 6487.17
----------------------------------------------------------------
'''
