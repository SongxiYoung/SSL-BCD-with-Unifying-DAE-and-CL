#########################
# DAE for building object representation learning
# Pretext
# Both pre and post-disaster bldg objects loaded on one side, used as baseline similar to SimCLR
#########################

from torchvision import transforms, utils
import torch
import time
import logging
import dataproc_double as dp
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import os
# from dae import CNNDAE as Model
import dae
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='dual_1', help='version of the experiment')
parser.add_argument('-t', '--train_batch_size', type=int, default=32)
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-6, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default="", help="path to pre-trained model")
parser.add_argument('--data_path', type=str, default= "")
parser.add_argument('--csv_train', type=str, default='', help='train csv sub-path within data path')
parser.add_argument('--csv_eval', type=str, default='', help='train csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=10, help="print evaluation for every n iterations")

# Model parameters
parser.add_argument('--model', default='denoising_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
parser.add_argument('--patch_size', default=16, type=int,
                        help='images patch size')

parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

def main(args):

    #######################################################
    # Create necessary folders for logging results
    #######################################################
    # create logs folder
    if not os.path.isdir('./output_dir'):
        os.mkdir('./output_dir')

    log_dir = "./output_dir/{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "./output_dir/{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    print("-----------------------------------------------------------------------------")
    print("---------------Denoising Autoencoder of xBD Pre & Post Building Object Image Patches---------------")

    #######################################################
    # Log arguments
    #######################################################
    print('version: {}'.format(args.version))
    print('data_path: {}'.format(args.data_path))
    print('csv_train: {}'.format(args.csv_train))
    print('pretrained_model: {}'.format(args.pretrained_model))
    print('lr: {}'.format(args.lr))
    print('wd: {}'.format(args.wd))
    print('n_epochs: {}'.format(args.n_epochs))
    print('train batch size: {}'.format(args.train_batch_size))
    print('print_freq: {}'.format(args.print_freq))

    #######################################################
    # load computing device, cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Model().to(device=device)
    # net = dae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    net = dae.DualAutoencoderViT(img_size=args.input_size,
        patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_pix_loss=args.norm_pix_loss)

    net.to(device)
    print(device)
    print(net)

    #######################################################
    # Criterion, optimizer, learning rate scheduler
    # original SimCLR uses optimizer LARS with learning rate of 4.8
    # however, LARS is not implemented in PyTorch, so we use Adam
    # with learning rate of 1e-4 instead with weight decay 1e-6
    #######################################################
    #criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.BCELoss()
    #weight = torch.tensor([0.03087066637337608, 0.2944726031734042, 0.3531639351033918, 0.3214927953498281], device=device)
    #weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 45], gamma=0.1)
    #logging.info(criterion)
    print(optimizer)
    print(scheduler)

    #######################################################
    # train + valid dataset
    # Load both pre- & post- images to One Side, and copy to the other Side
    #######################################################
    # mean & std
    # same mean and std for pre and post images
    mean_pre = (0.39327543, 0.40631564, 0.32678495)
    std_pre = (0.16512179, 0.14379614, 0.15171282)
    mean_post = (0.39327543, 0.40631564, 0.32678495)
    std_post = (0.16512179, 0.14379614, 0.15171282)
    # train data
    transform_trn = transforms.Compose([
        dp.Resize((224, 224)),
        dp.RandomFlip(p=0.5),
        dp.ColorJitter_BCSH(0.8, 0.8, 0.8, 0.2, p=0.8),
        dp.RandomGrayScale(p=0.2),
        dp.GaussianBlur(kernel_size=9, sigma_range=(0.1, 2.0), p=0.5),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
        ])
    trainset = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=transform_trn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=10)
    print(trainset)

    #######################################################
    # resume from pre-trained model
    #######################################################
    min_loss = float('inf') # initialize valid loss to a large number
    start_epoch = 0
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        net.load_state_dict(checkpoint['net_state_dict'])
        print("resumed checkpoint at epoch {} with min loss {:.4f}".format(start_epoch, min_loss))

    #######################################################
    # start training + validation
    #######################################################
    t0 = time.time()
    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(start_epoch+ep + 1, start_epoch+args.n_epochs))

        # training
        t1 = time.time()
        loss_train = train(trainloader, net, optimizer, criterion, start_epoch+ep, writer, device, args.print_freq)
        t2 = time.time()
        print('Train [Time: {:.2f} hours] [Loss: {:.4f}]'.format((t2 - t1) / 3600.0, loss_train))
        writer.add_scalars('training/Loss', {"train": loss_train}, start_epoch+ep + 1)

        print('Time spent total at [{}/{}]: {:.2f}'.format(start_epoch + ep + 1, start_epoch + args.n_epochs, (t2 - t0) / 3600.0))

        # save the best model
        is_best = loss_train < min_loss
        min_loss = min(loss_train, min_loss)
        save_checkpoint({
            'epoch': start_epoch + ep + 1,
            'net_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min_loss,
        }, is_best, root_dir=model_dir, checkpoint_name='checkpoint_ep_{}'.format(start_epoch + ep + 1))

        # reschedule learning rate
        scheduler.step(loss_train) # if loss plateau
        #scheduler.step() # for multiple step LR change
        current_LR = optimizer.param_groups[0]['lr']
        logging.info('Current learning rate: {:.4e}'.format(current_LR))
        if current_LR < 1e-6:
            logging.info('**********Learning rate too small, training stopped at epoch {}**********'.format(start_epoch + ep + 1))
            break

    print('Training Done....')


def train(dataloader, net, optimizer, criterion, epoch, writer, device='cpu', print_freq=5):

    print('Training...')
    net.train()

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)
    #pdb.set_trace()

    for i, batch in enumerate(dataloader):

        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        batch_size = bldg_pre.shape[0]

        #######################
        # visualize input
        show_batch = 0
        if show_batch:
           print('visualizing input...')
           grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=16, normalize=True)
           show_tensor_img(grid, "input_pre")
           grid = utils.make_grid(vis_ms(bldg_post, 0, 1, 2), nrow=16, normalize=True)
           show_tensor_img(grid, "input_post")
           print('visualization done')
        ########################

        # feed forward
        data_noise = torch.randn(bldg_pre.shape).to(device)  # generate noise
        pre_noise = bldg_pre + data_noise  # add noise
        post_noise = bldg_post + data_noise

        loss, recon_pre, recon_post= net(pre_noise, post_noise)
        # loss = criterion(recon_batch, bldg_pre)
        
        # update loss
        epoch_loss.update(loss.item(), batch_size)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (n_batches // print_freq + 1) == 0:
            print('[{}][{}/{}], loss={:.4f}'.format(epoch+1, i, n_batches, epoch_loss.avg))

    return epoch_loss.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
