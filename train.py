"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import torch

from torch.utils.tensorboard import SummaryWriter
def data_norm_back(img):
    img_back = (img * 0.5 + 0.5) * 255
    img_back = torch.clamp(img_back, min=0, max=255)
    img_back = torch.tensor(img_back, dtype=torch.uint8)
    return img_back

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    writer = SummaryWriter('./results/' + str(opt.name) + "/log")

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            real_A, real_B = model.set_input(data)         # unpack data from dataset and apply preprocessing
            GAN_A2B_plus_B2A, loss_GAN_A2B, loss_GAN_B2A, L1_loss_A_B, L1_loss_A, L1_loss_B, gradient_loss_A_B, gradient_loss_A, gradient_loss_B, cycle_A_plus_B, loss_cycle_A, loss_cycle_B, AEloss_real_A_plus_B, loss_AE_real_A, loss_AE_real_B, D_A_plus_B, loss_D_A, loss_DA_real, loss_DA_fake, loss_D_B, loss_DB_real, loss_DB_fake, fake_B, fake_A = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        '''**** draw all of loss results in tensorboard  ****'''
        # writer.add_scalar('loss_all_G', loss_all_G, epoch)

        writer.add_scalar('G_GANloss/GAN_A2B_plus_B2A', GAN_A2B_plus_B2A, epoch)
        writer.add_scalar('G_GANloss/loss_GAN_A2B', loss_GAN_A2B, epoch)
        writer.add_scalar('G_GANloss/loss_GAN_B2A', loss_GAN_B2A, epoch)

        writer.add_scalar('cycle_loss/cycle_A_plus_B', cycle_A_plus_B, epoch)
        writer.add_scalar('cycle_loss/loss_cycle_A', loss_cycle_A, epoch)
        writer.add_scalar('cycle_loss/loss_cycle_B', loss_cycle_B, epoch)

        writer.add_scalar('gradient_loss/gradient_loss_A_B', gradient_loss_A_B, epoch)
        writer.add_scalar('gradient_loss/gradient_loss_A', gradient_loss_A, epoch)
        writer.add_scalar('gradient_loss/gradient_loss_B', gradient_loss_B, epoch)


        writer.add_scalar('AEloss/AEloss_real_A_plus_B', AEloss_real_A_plus_B, epoch)
        writer.add_scalar('AEloss/loss_AE_real_A', loss_AE_real_A, epoch)
        writer.add_scalar('AEloss/loss_AE_real_B', loss_AE_real_B, epoch)

        # writer.add_scalar('AEloss/AEloss_fake_A_plus_B', AEloss_fake_A_plus_B, epoch)
        # writer.add_scalar('AEloss/loss_AE_fake_A', loss_AE_fake_A, epoch)
        # writer.add_scalar('AEloss/loss_AE_fake_B', loss_AE_fake_B, epoch)


        writer.add_scalar('D_loss/D_A_plus_B', D_A_plus_B, epoch)
        writer.add_scalar('D_loss/loss_D_A', loss_D_A, epoch)    
        writer.add_scalar('D_loss/loss_DA_real', loss_D_A, epoch)    
        writer.add_scalar('D_loss/loss_DA_fake', loss_D_A, epoch)    
        writer.add_scalar('D_loss/loss_D_B', loss_D_B, epoch)    
        writer.add_scalar('D_loss/loss_DB_real', loss_D_B, epoch)    
        writer.add_scalar('D_loss/loss_DB_fake', loss_D_B, epoch)    
        
        writer.add_scalar('L1_loss/L1_loss_A_B', L1_loss_A_B, epoch)    
        writer.add_scalar('L1_loss/L1_loss_A', L1_loss_A, epoch)    
        writer.add_scalar('L1_loss/L1_loss_B', L1_loss_B, epoch)            


        tensor_show = torch.cat((data_norm_back(real_A), data_norm_back(fake_B), data_norm_back(real_B),  data_norm_back(fake_A)), dim=0)

        writer.add_images("real_fake_reconstruct_" + str(epoch), tensor_show, global_step = epoch, walltime=None, dataformats='NCHW')
