import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--ganloss', type=float, default=4.0, help='weight for gan loss (A -> B -> A)')
            # parser.add_argument('--AEloss', type=float, default=4.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--real_AEloss_para', type=float, default=20, help='weight for cycle loss (A -> B -> A)')

        return parser



    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'GAN_A2B', 'cycle_A', 'D_B', 'GAN_B2A', 'cycle_B']#, 'idt_A2B', 'idt_B2A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B2A=G_A(B) ad idt_A2B=G_A(B)
            visual_names_A.append('idt_B2A')
            visual_names_B.append('idt_A2B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A2B', 'G_B2A', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A2B', 'G_B2A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A2B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B2A = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        # wqh
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images 
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.embeding_real_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images 
            self.embeding_real_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.embeding_fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images 
            self.embeding_fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images                        
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.ganloss = opt.ganloss
            self.real_AEloss_para = opt.real_AEloss_para
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.AE_loss = torch.nn.L1Loss()
            self.gradient_loss = torch.nn.L1Loss()
            self.L1_loss = torch.nn.L1Loss()

            # Omni-directional gradient kernel
            gradient_kernel = [[[[-1.0,  -1.0,  -1.0],
                        [-1.0,  8.0,  -1.0],
                        [-1.0, -1.0, -1.0]]]]  # out_channel, channels, h, w
            gradient_kernel = torch.FloatTensor(gradient_kernel).expand(1,3,3,3).cuda()
            self.gradient_kernel_weight = torch.nn.Parameter(data = gradient_kernel, requires_grad=False)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_D = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        return self.real_A, self.real_B


    def forward(self):
        # print(self.real_A.shape)        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.embeding_real_A = self.netG_A2B(self.real_A, "translation")   # G_A(A)
        self.rec_A,  self.embeding_fake_B = self.netG_B2A(self.fake_B, "translation")   # G_B(G_A(A))
        self.AE_real_A, _ = self.netG_B2A(self.embeding_real_A, "decoder")  
        self.AE_fake_B, _ = self.netG_A2B(self.embeding_fake_B, "decoder")  


        self.fake_A, self.embeding_real_B = self.netG_B2A(self.real_B, "translation")   # G_B(B)
        self.rec_B,  self.embeding_fake_A = self.netG_A2B(self.fake_A, "translation")   # G_A(G_B(B))
        self.AE_real_B, _ = self.netG_A2B(self.embeding_real_B, "decoder")
        self.AE_fake_A, _ = self.netG_B2A(self.embeding_fake_A, "decoder")




    # need to change
    def backward_D_basic(self, netD, embedding_real, embedding_fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(embedding_real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.ganloss
        # Fake
        pred_fake = netD(embedding_fake)    
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.ganloss
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D, loss_D_real, loss_D_fake


    # need to change
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_A"""

        self.loss_D_B, self.loss_DB_real, self.loss_DB_fake = self.backward_D_basic(self.netD_B, self.embeding_real_B, self.embeding_fake_B)


    # need to change
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_A, self.loss_DA_real, self.loss_DA_fake = self.backward_D_basic(self.netD_A, self.embeding_real_A, self.embeding_fake_A)


    # need to change
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B       

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # calculate L1 loss
        self.L1_loss_A = self.L1_loss(self.fake_A, self.real_A) * 8
        self.L1_loss_B = self.L1_loss(self.fake_B, self.real_B) * 8



        # calculate Omni-directional gradient loss
        self.gradient_fake_A = F.conv2d(self.fake_A, self.gradient_kernel_weight, bias=None, stride=1, padding=1)
        self.gradient_real_A = F.conv2d(self.real_A, self.gradient_kernel_weight, bias=None, stride=1, padding=1)
        self.gradient_fake_B = F.conv2d(self.fake_B, self.gradient_kernel_weight, bias=None, stride=1, padding=1)
        self.gradient_real_B = F.conv2d(self.real_B, self.gradient_kernel_weight, bias=None, stride=1, padding=1)

        self.gradient_loss_A = self.gradient_loss(self.gradient_fake_A, self.gradient_real_A) * 3.5           
        self.gradient_loss_B = self.gradient_loss(self.gradient_fake_B, self.gradient_real_B) * 2.5           



        # AE loss
        self.loss_AE_real_A = self.AE_loss(self.AE_real_A, self.real_A) * self.real_AEloss_para
        self.loss_AE_real_B = self.AE_loss(self.AE_real_B, self.real_B) * self.real_AEloss_para


        self.loss_G_backward_to_all = self.loss_cycle_A + self.loss_cycle_B + self.loss_AE_real_A + self.loss_AE_real_B + self.gradient_loss_A + self.gradient_loss_B + self.L1_loss_A + self.L1_loss_B
        self.loss_G_backward_to_all.backward(retain_graph=True)



        # GAN loss D_A(G_A(A)) 
        self.set_requires_grad([self.netG_B2A], False)
        self.loss_GAN_A2B = self.criterionGAN(self.netD_B(self.embeding_fake_B), True) * self.ganloss
        self.loss_GAN_A2B.backward(retain_graph=True)
        self.set_requires_grad([self.netG_B2A], True)


        # GAN loss D_B(G_B(B))
        self.set_requires_grad([self.netG_A2B], False)
        self.loss_GAN_B2A = self.criterionGAN(self.netD_A(self.embeding_fake_A), True) * self.ganloss
        self.loss_GAN_B2A.backward(retain_graph=True)
        self.set_requires_grad([self.netG_A2B], True)





    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_B, self.netD_A], False)  # Ds require no gradients when optimizing Gs


        self.optimizer_G_D.zero_grad()  # set G_A and G_B's gradients to zero

        self.backward_G()             # calculate gradients for G_A and G_B
    



        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)

        self.set_requires_grad([self.netG_A2B], False)
        self.backward_D_B()      # calculate gradients for D_A 
        self.set_requires_grad([self.netG_A2B], True)

        self.set_requires_grad([self.netG_B2A], False)
        self.backward_D_A()      # calculate graidents for D_B  
        self.set_requires_grad([self.netG_B2A], True)


        self.optimizer_G_D.step()       # update G_A and G_B's weights

        return (self.loss_GAN_A2B + self.loss_GAN_B2A), self.loss_GAN_A2B, self.loss_GAN_B2A, (self.L1_loss_A + self.L1_loss_B), self.L1_loss_A, self.L1_loss_B, (self.gradient_loss_A + self.gradient_loss_B), self.gradient_loss_A, self.gradient_loss_B, (self.loss_cycle_A + self.loss_cycle_B), self.loss_cycle_A, self.loss_cycle_B, (self.loss_AE_real_A + self.loss_AE_real_B), self.loss_AE_real_A, self.loss_AE_real_B, (self.loss_D_A + self.loss_D_B), self.loss_D_A, self.loss_DA_real, self.loss_DA_fake, self.loss_D_B, self.loss_DB_real, self.loss_DB_fake, self.fake_B, self.fake_A
