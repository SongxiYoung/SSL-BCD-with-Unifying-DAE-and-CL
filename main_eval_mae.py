#########################
#########################

from torchvision import transforms, utils
import torch
import time
import logging
import dataproc_double as dp
from utils import AverageMeter, show_tensor_img, set_logger, vis_ms, save_checkpoint, batch_similarity, harmonic_mean
from torch.utils.tensorboard import SummaryWriter
import os
import models_mae
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='maevit', help='version of the experiment')
parser.add_argument('-t', '--train_batch_size', type=int, default=32)
parser.add_argument('-e', '--n_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--wd', type=float, default=1e-6, help="weight decay")
parser.add_argument('--pretrained_model', type=str, default="output_dir/maevit/model_best.pth.tar", help="path to pre-trained model")
parser.add_argument('--data_path', type=str, default= "/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors")
parser.add_argument('--csv_train', type=str, default='csvs_buffer/sub_valid_wo_unclassified_prepost.csv', help='train csv sub-path within data path')
parser.add_argument('--csv_eval', type=str, default='csvs_buffer/sub_train_wo_unclassified_prepost.csv', help='train csv sub-path within data path')
parser.add_argument('--print_freq', type=int, default=10, help="print evaluation for every n iterations")

# Model parameters
parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

def main(args):

    print("-----------------------------------------------------------------------------")
    print("---------------Masked Autoencoder of xBD Pre & Post Building Object Image Patches---------------")

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
    net = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss).to(device=device)
    # Load the pretrained model if specified
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        net.load_state_dict(checkpoint['net_state_dict'])
        print("resumed checkpoint at epoch {} with min loss {:.4f}".format(start_epoch, min_loss))
    net.eval()
    print(device)
    print(net)

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
    
    trainset = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_train,
                                    transform=transform_trn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    

    evalset = dp.xBD_Building_Polygon_TwoSides_PrePost(data_path = args.data_path,
                                    csv_file=args.csv_eval,
                                    transform=transform_trn)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print(evalset)

    fea_label_train = extract_features(net, trainloader, device)
    # fea_label_eval = extract_features(net, evalloader, device)
    fea_label_eval = fea_label_train
    print('features extracted.')

    # Train the classifier
    classifier = train_classifier(fea_label_train)

    eval_similarity = [x['similarity'] for x in fea_label_eval]
    eval_label = [x['bldg_damage'] for x in fea_label_eval]
    eval_similarity = np.array(eval_similarity).reshape(-1, 1)
    predicted_label = classifier.predict(eval_similarity)

    # Calculate the confusion matrix
    cm = confusion_matrix(eval_label, predicted_label)
    print('Confusion Matrix:')
    print(cm)

    # Calculate the F1 score
    f1 = f1_score(eval_label, predicted_label, average=None)
    for i, score in enumerate(f1):
        print(f'F1 Score for class {i}: {score}')

    # Calculate the overall F1 score
    overall_f1 = harmonic_mean(f1)
    print(f'Overall F1 Score: {overall_f1}')

    # Calculate the accuracy
    accuracy = accuracy_score(eval_label, predicted_label)
    print('Accuracy: ', accuracy)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(classifier, np.array(eval_similarity).reshape(-1, 1), eval_label, cv=5)

    # Calculate the mean and standard deviation of the accuracy scores
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    print(f'Cross-validated Accuracy: {mean_accuracy * 100:.1f}±{std_accuracy * 100:.1f}')

def train_classifier(feature_label, n_samples_per_class=8000):
    # Separate feature_label by class
    classes = set(x['bldg_damage'] for x in feature_label)
    feature_label_by_class = {cls: [x for x in feature_label if x['bldg_damage'] == cls] for cls in classes}

    # Resample each class to have n_samples_per_class
    resampled_feature_label = []
    for cls, fl in feature_label_by_class.items():
        if len(fl) > n_samples_per_class:
            fl = resample(fl, replace=False, n_samples=n_samples_per_class, random_state=0)
        resampled_feature_label.extend(fl)

    # Extract features and labels
    similarity = [x['similarity'] for x in resampled_feature_label]
    damage_label = [x['bldg_damage'] for x in resampled_feature_label]

    # Reshape similarity to a 2D array
    similarity = np.array(similarity).reshape(-1, 1)

    # Create a Logistic Regression classifier
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(similarity, damage_label)

    return classifier

@torch.no_grad()
def extract_features(net, data_loader, device):
    net.eval()
    similarity_code = []

    for i, batch in enumerate(data_loader):

        bldg_pre = batch['bldg_pre'].to(device=device, dtype=torch.float32)
        bldg_post = batch['bldg_post'].to(device=device, dtype=torch.float32)
        bldg_damage = batch['bldg_damage'].to(device=device, dtype=torch.float32)
 
        # recon_batch = net(bldg_post)
        loss, recon_batch, mask = net(bldg_post)
        recon_batch = net.unpatchify(recon_batch)

        similarities = batch_similarity(bldg_pre, recon_batch)
        # similarities = compute_gradient_similarity(bldg_pre, recon_batch)

        for j, sim in enumerate(similarities):
            # print('Eval [{}/{}]\tImage {}\tSimilarity: {:.4f}\tDamage: {:.4f}'.format(
            #     i, len(data_loader), j, sim.item(), bldg_damage[j].item()))
            similarity_code.append({
                'similarity': sim.item(),
                'bldg_damage': bldg_damage[j].item()
            })
 
        #######################
        # visualize input
        show_batch = 1
        with open('batch_bldg_damage.txt', 'w') as f:
            pass
        if i == 2 and show_batch:
            print('visualizing output...')
            grid = utils.make_grid(vis_ms(bldg_pre, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "output_pre")
            grid = utils.make_grid(vis_ms(bldg_post, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "output_post")
            grid = utils.make_grid(vis_ms(recon_batch, 0, 1, 2), nrow=16, normalize=True)
            show_tensor_img(grid, "output_recon")

            # Save bldg_damage to a text file
            with open('batch_bldg_damage.txt', 'a') as f:
                for j in range(bldg_damage.shape[0]):
                    f.write(str(bldg_damage[j].item()) + '\n')

            print('visualization done')
        ########################
        
    return similarity_code

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)