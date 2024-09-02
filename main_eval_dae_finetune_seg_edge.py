from torchvision import transforms, utils
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
import dataproc_double as dp
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, vis_single_channel, save_checkpoint, precision_recall_multi_class_pytorch, mean_iou
from torch.utils.tensorboard import SummaryWriter
import os
import dae

import argparse
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='seg_vit_edge', help='version of the experiment')
parser.add_argument('-t', '--train_batch_size', type=int, default=32)
parser.add_argument('-e', '--n_epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-6, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default="", help="path to pre-trained model")
parser.add_argument('--checkpoint', type=str, default="", help="path to resume training")
parser.add_argument('--data_path', type=str, default= "")
parser.add_argument('--csv_train', type=str, default='', help='train csv sub-path within data path')
parser.add_argument('--csv_eval', type=str, default='', help='train csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=100, help="print evaluation for every n iterations")
parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")

# Model parameters
parser.add_argument('--model', default='denoising_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
parser.add_argument('--patch_size', default=16, type=int,
                        help='images patch size')

parser.add_argument('--num_labels', default=4, type=int,
                        help='Number of labels to classify')

parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks""")

parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

def main(args):

    #######################################################
    # Create necessary folders for logging results
    #######################################################
    # create logs folder
    if not os.path.isdir('./output_dir/'):
        os.mkdir('./output_dir/')

    log_dir = "./output_dir/{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "./output_dir/{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

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
    print(device)

    # net = Model().to(device=device)
    net = dae.DualAutoencoderViT(img_size=args.input_size,
        patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_pix_loss=args.norm_pix_loss).to(device=device)
    
    # Load the pretrained model if specified
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']

        # net.load_state_dict(checkpoint['net_state_dict'])
        # Filter out the 'SuperRe' keys
        net_state_dict = {k: v for k, v in checkpoint['net_state_dict'].items() if not k.startswith('SuperRe')}
        net.load_state_dict(net_state_dict, strict=False)

        print("resumed DualAutoencoderViT checkpoint at epoch {} with min loss {:.4f}".format(start_epoch, min_loss))

    #######################################################
    # model
    model = ViT_FPN(net).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        model.load_state_dict(checkpoint['net_state_dict'])
        print("resumed downstream checkpoint at epoch {} with min loss {:.4f}".format(start_epoch, min_loss))
    else:
        min_loss = float('inf') 
        start_epoch = 0

    # loss function and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
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
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
        ])
    
    trainset = dp.xBD_Building_Mask_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=transform_trn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=10)
    print(trainset)

    transform_val = transforms.Compose([
        dp.Resize((224, 224)),
        dp.ToTensor(),
        dp.Normalize_Std(mean_pre, std_pre, mean_post, std_post)
        ])
    
    evalset = dp.xBD_Building_Mask_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_eval,
                                    transform=transform_val)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.train_batch_size, shuffle=True, num_workers=10)
    print(evalset)

    #######################################################
    # start training + validation
    #######################################################
    t0 = time.time()
    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(start_epoch+ep + 1, start_epoch+args.n_epochs))

        # training
        t1 = time.time()
        loss_train, iou_train = train_classifier(model, optimizer, criterion, trainloader, device, start_epoch+ep + 1, print_freq=5)
        t2 = time.time()
        print('Train [Time: {:.2f} hours] [Loss: {:.4f}] [IoU: {:.4f}]'.format((t2 - t1) / 3600.0, loss_train, iou_train))

        print('Time spent total at [{}/{}]: {:.2f}'.format(start_epoch + ep + 1, start_epoch + args.n_epochs, (t2 - t0) / 3600.0))

        if ep % args.val_freq == 0 or ep == args.epochs - 1:
            loss_eval, iou_eval = validate_network(criterion, evalloader, model, device, start_epoch+ep + 1)
            print('Validation [Loss: {:.4f}] [IoU: {:.4f}]'.format(loss_eval, iou_eval))

        # save the best model
        is_best = loss_train < min_loss
        min_loss = min(loss_train, min_loss)
        save_checkpoint({
            'epoch': start_epoch + ep + 1,
            'net_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min_loss,
        }, is_best, root_dir=model_dir, checkpoint_name='fpn_checkpoint_ep_{}'.format(start_epoch + ep + 1))

        # reschedule learning rate
        scheduler.step(loss_train) # if loss plateau
        #scheduler.step() # for multiple step LR change
        current_LR = optimizer.param_groups[0]['lr']
        print('Current learning rate: {:.4e}'.format(current_LR))
        if current_LR < 1e-6:
            print('**********Learning rate too small, training stopped at epoch {}**********'.format(start_epoch + ep + 1))
            break
        
    print("Training of the supervised linear classifier on frozen features completed.\n")
    

def train_classifier(model, optimizer, criterion, dataloader, device, epoch, print_freq=5):
    model.train()
    epoch_loss = AverageMeter()
    epoch_iou = AverageMeter()
    n_batches = len(dataloader)

    print('Training...')

    for i, batch in enumerate(dataloader):
        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        bldg_pre_mask = batch['pre_mask'].float().to(device=device).unsqueeze(1)
        bldg_post_mask = batch['post_mask'].float().to(device=device).unsqueeze(1)

        # 
        bldg_post_mask = torch.where(bldg_post_mask > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        bldg_pre_mask = torch.where(bldg_pre_mask > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

        post_edges = sobel_edge_detection(bldg_post_mask, device)
        post_edges = (post_edges > 0.1).float().to(device=device, dtype=torch.float32)
        pre_edges = sobel_edge_detection(bldg_pre_mask, device)
        pre_edges = (pre_edges > 0.1).float().to(device=device, dtype=torch.float32)
    
        batch_size = bldg_pre.shape[0] 

        # forward
        pre_edge, pre_mask, post_edge, post_mask = model(bldg_pre, bldg_post)
        
        # compute loss
        loss = criterion(pre_edge, pre_edges, pre_mask, bldg_pre_mask) + criterion(post_edge, post_edges, post_mask, bldg_post_mask)
        epoch_loss.update(loss.item(), batch_size)

        # Calculate IoU
        batch_iou = 0.5*mean_iou(torch.sigmoid(pre_mask), bldg_post_mask) + 0.5*mean_iou(torch.sigmoid(post_mask), bldg_post_mask)
        epoch_iou.update(batch_iou, batch_size)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss, batch_size)
        epoch_iou.update(batch_iou, batch_size)


        #######################
        # visualize input
        show_batch = 0
        if i == 0 and show_batch:
            print('visualizing output...')
            grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_original")
            grid = utils.make_grid(vis_single_channel(pre_mask, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_pred")
            grid = utils.make_grid(vis_single_channel(pre_edges, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_edges")
            print('visualization done')
        ########################

        if i % (n_batches // print_freq + 1) == 0:
            print('[{}][{}/{}], loss={:.4f}, IoU: {:.4f}'.format(epoch+1, i, n_batches, epoch_loss.avg, epoch_iou.avg))
        
    return epoch_loss.avg, epoch_iou.avg

@torch.no_grad()
def validate_network(criterion, dataloader, model, device, epoch, print_freq=5):
    model.eval()

    epoch_loss = AverageMeter()
    epoch_iou = AverageMeter()

    n_batches = len(dataloader)

    print('Validating...')

    for i, batch in enumerate(dataloader):
 
        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        bldg_pre_mask = batch['pre_mask'].float().to(device=device).unsqueeze(1)
        bldg_post_mask = batch['post_mask'].float().to(device=device).unsqueeze(1)
        # 
        bldg_post_mask = torch.where(bldg_post_mask > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        bldg_pre_mask = torch.where(bldg_pre_mask > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

        post_edges = sobel_edge_detection(bldg_post_mask, device)
        post_edges = (post_edges > 0.1).float().to(device=device, dtype=torch.float32)
        pre_edges = sobel_edge_detection(bldg_pre_mask, device)
        pre_edges = (pre_edges > 0.1).float().to(device=device, dtype=torch.float32)
    
        batch_size = bldg_pre.shape[0] 
 
        # forward
        pre_edge, pre_mask, post_edge, post_mask = model(bldg_pre, bldg_post)
        
        # compute loss
        loss = criterion(pre_edge, pre_edges, pre_mask, bldg_pre_mask) + criterion(post_edge, post_edges, post_mask, bldg_post_mask)
        epoch_loss.update(loss.item(), batch_size)

        # Calculate IoU
        batch_iou = 0.5*mean_iou(torch.sigmoid(pre_mask), bldg_post_mask) + 0.5*mean_iou(torch.sigmoid(post_mask), bldg_post_mask)
        epoch_iou.update(batch_iou, batch_size)

        #######################
        # visualize input
        show_batch = 1
        if i == 0 and show_batch:
            print('visualizing output...')
            grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_original")
            grid = utils.make_grid(vis_single_channel(pre_mask, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_pred")
            grid = utils.make_grid(vis_single_channel(pre_edges, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_pre_edges")
            grid = utils.make_grid(vis_ms(bldg_post, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_post_original")
            grid = utils.make_grid(vis_single_channel(post_mask, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_post_pred")
            grid = utils.make_grid(vis_single_channel(post_edges, 0), nrow=16, normalize=True)
            show_tensor_img(grid, "edge_post_edges")
            print('visualization done')
        ########################

        if i % (n_batches // print_freq + 1) == 0:
            print('[{}][{}/{}], loss={:.4f}, IoU: {:.4f}'.format(epoch+1, i, n_batches, epoch_loss.avg, epoch_iou.avg))
        
    return epoch_loss.avg, epoch_iou.avg

class FPN(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1):
        super(FPN, self).__init__()

        # Load a pretrained backbone (ResNet)
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        # Replace the first convolutional layer
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract layers from the backbone
        self.layer1 = nn.Sequential(self.backbone.conv1,
                                    self.backbone.bn1,
                                    self.backbone.relu,
                                    self.backbone.maxpool,
                                    self.backbone.layer1)  # Output stride 4
        self.layer2 = self.backbone.layer2  # Output stride 8
        self.layer3 = self.backbone.layer3  # Output stride 16
        self.layer4 = self.backbone.layer4  # Output stride 32

        # Lateral layers
        self.lateral4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Top-down pathway
        p4 = self.lateral4(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p1 = self._upsample_add(p2, self.lateral1(c1))

        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        out = p1
        # out = self.head(p1)
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # Upsample to original size

        return out

class EdgeGuidanceModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeGuidanceModule, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Initialization (optional)
        nn.init.kaiming_normal_(self.conv1x1_1.weight)
        nn.init.kaiming_normal_(self.conv1x1_2.weight)

    def forward(self, x):
        # Apply the first 1x1 convolution and softmax
        edge_logits = self.conv1x1_1(x)
        edge_prob_map = F.softmax(edge_logits, dim=1)
        
        # Apply the second 1x1 convolution and softmax
        change_logits = self.conv1x1_2(self.relu(edge_logits + x))
        change_prob_map = F.softmax(change_logits, dim=1)
        
        return edge_logits, change_logits

class ViT_FPN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.pos_embed = model.pos_embed
        self.cls_token = model.cls_token
        self.blocks = model.blocks
        self.norm = model.norm
        self.get_intermediate_layers = model.get_intermediate_layers

        self.vit = model
        self.fpn_pre = FPN()
        self.fpn_post = FPN()

        self.egm = EdgeGuidanceModule(in_channels=256) 

        # Define the Dual Attention Block
        self.pam = PAM_Module(in_dim=3)
        self.cam = CAM_Module(in_dim=3)
        # self.conv_last = nn.Conv2d(6, 2, kernel_size=1)

        for param in self.vit.parameters():
            param.requires_grad = False
        
    def forward(self, pre_img, post_img):
        # Extract features from both images
        pre_vit_features = self.vit.get_encoder_layers(pre_img)
        post_vit_features = self.vit.get_encoder_layers(post_img)

        # Remove the class token
        pre_vit_features = pre_vit_features[:, 1:]
        post_vit_features = post_vit_features[:, 1:]

        # Unpatchify the features
        pre_latent = self.vit.unpatchify(pre_vit_features) # 3, 224, 224
        post_latent = self.vit.unpatchify(post_vit_features) # 3, 224, 224
        # print(pre_latent.shape, post_latent.shape)

        # Concatenate the features along the channel dimension
        latent = torch.cat([pre_latent, post_latent], dim=1) # 6, 224, 224

        # Pass through FPN
        # output = self.conv_last(latent)
        fpn_pre = self.fpn_pre(pre_latent)
        fpn_post = self.fpn_post(post_latent)

        pre_edge, pre_mask = self.egm(fpn_pre)
        post_edge, post_mask = self.egm(fpn_post)

        return pre_edge, pre_mask, post_edge, post_mask


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, logits=True)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, edge_prob, edges, building_prob, buildings, dice_weight=None):
        if dice_weight is not None:
            self.dice_weight = dice_weight

        fl = self.focal_loss(edge_prob, edges)
        dl = self.dice_loss(building_prob, buildings)
        return (1 - self.dice_weight) * fl + self.dice_weight * dl

def sobel_edge_detection(image, device):
    sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Move kernels to the same device as image
    sobel_kernel_x = sobel_kernel_x.to(device)
    sobel_kernel_y = sobel_kernel_y.to(device)

    # Apply the Sobel kernels
    edge_x = F.conv2d(image, sobel_kernel_x, padding=1)
    edge_y = F.conv2d(image, sobel_kernel_y, padding=1)

    # Combine the edges
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edges

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
