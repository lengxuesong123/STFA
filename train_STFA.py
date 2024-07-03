import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.ST_loss import compute_loss
from dataloaders import utils


from dataloaders.acdc import ACDCDataset
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/home/data/lxs/SSL1/SSL4MIS-master/SSL4MIS-master/data/ACDC", help="Name of Experiment")
parser.add_argument("--save_path", type=str, default="/home/data/lxs/SSL1/SSL4MIS-master/SSL4MIS-master/SAM_model", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="ACDC/STFA", help="experiment_name")
parser.add_argument("--model", type=str, default="unet_drop", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=12, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=int, default=256, help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)

parser.add_argument("--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu")
parser.add_argument("--labeled_num", type=int, default=1, help="labeled data")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--labeled-id-path', default='/home/data/lxs/SSL1/SSL4MIS-master/SSL4MIS-master/down/splits/acdc/1 /labeled.txt',type=str)
parser.add_argument('--unlabeled-id-path',default='/home/data/lxs/SSL1/SSL4MIS-master/SSL4MIS-master/down/splits/acdc/1 /unlabeled.txt', type=str)

args = parser.parse_args()


class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        x = x.unsqueeze(1)
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patch_list, 1)
        y = np.fft.fftn(y.detach().cpu().numpy())
        freq = torch.tensor(y).cuda()
        freq = torch.stack([freq.real, freq.imag], -1)
        
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "1": 16,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model



    model = create_model()


    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # if restoring previous models:
    snapshot_path = args.save_path
    if args.load:
        try:
            # check if there is previous progress to be restored:
            logging.info(f"Snapshot path: {snapshot_path}")
            iter_num = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, extension = os.path.splitext(filename)
                    iter_num.append(int(basename.split("_")[2]))
            iter_num = max(iter_num)
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename and str(iter_num) in filename:
                    model_checkpoint = filename
        except Exception as e:
            logging.warning(f"Error finding previous checkpoints: {e}")

        try:
            logging.info(f"Restoring model checkpoint: {model_checkpoint}")
            model, optimizer, start_epoch, performance = util.load_checkpoint(
                snapshot_path + "/" + model_checkpoint, model, optimizer
            )
            logging.info(f"Models restored from iteration {iter_num}")
        except Exception as e:
            logging.warning(f"Unable to restore model checkpoint: {e}, using new model")


    trainset_u = ACDCDataset('acdc', args.root_path, 'train_u',
                             args.patch_size, args.unlabeled_id_path)
    trainset_l = ACDCDataset('acdc', args.root_path, 'train_l',
                             args.patch_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = ACDCDataset('acdc', args.root_path, 'test')

    trainsampler_l = torch.utils.data.RandomSampler(trainset_l, replacement=False, num_samples=None)
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_u)

    trainsampler_u_mix = torch.utils.data.RandomSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=args.batch_size,
                                   pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_u_mix)

    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0,
                           drop_last=False, sampler=valsampler)
    # set to train
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss_U(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader_u)))

    max_epoch = max_iterations // len(trainloader_u) + 1
    previous_best = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    
    for epoch_num in iterator:
        total_loss = util.AverageMeter()
        total_loss_x = util.AverageMeter()
        total_loss_s = util.AverageMeter()
        total_loss_w_fp = util.AverageMeter()
        total_mask_ratio = util.AverageMeter()
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
                
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            conf_u_w_fp = pred_u_w_fp.softmax(dim=1).max(dim=1)[0]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach() 
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1) #伪标签

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            #st_loss = compute_loss(pred_u_w,pred_u_s1)
            loss_x = (ce_loss(pred_x, mask_x) + dice_loss(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1 = dice_loss(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < args.conf_thresh).float())
            
            loss_u_s2 = dice_loss(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < args.conf_thresh).float())
            
            loss_u_w_fp = dice_loss(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < args.conf_thresh).float())
       
            ema_un = pred_u_w.softmax(dim=1) #[args.labeled_bs:]
          
            entropy = -torch.sum(ema_un * torch.log(ema_un + 1e-10), dim=1)
            entropy /= np.log(4)
            confidence = 1.0 - entropy
            confidence = confidence * conf_u_w
            confidence = confidence.mean(dim=[1,2])  # 1*C
            confidence = confidence.cpu().numpy().tolist()
            fft = FocalFrequencyLoss(1.0,1.0,1,True,True,True).cuda()
            fft_loss = fft(conf_u_w,conf_u_w_fp)
            st_loss = compute_loss(pred_u_s1,pred_u_s2,confidence=confidence)
            
           
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0 + 0.01*fft_loss + 0.03 * (st_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
          

            mask_ratio = (conf_u_w >= args.conf_thresh).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch_num * len(trainloader_u) + i
            lr = args.base_lr * (1 - iters / max_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
          
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
         
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))
            if (i % (len(trainloader_u) // 8) == 0):
                logging.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))
        model.eval()
        dice_class = [0] * 3
        
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (args.patch_size, args.patch_size), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)
                
                pred = model(img)
                
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, args.num_classes):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        
        
        for (cls_idx, dice) in enumerate(dice_class):
                logging.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, 'acdc', dice))
        logging.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            
        writer.add_scalar('eval/MeanDice', mean_dice, iter_num)
        CLASSES = {'acdc': ['Right Ventricle', 'Myocardium', 'Left Ventricle']}
        for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES['acdc'][i]), dice, iter_num)


        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if is_best:
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(iter_num, round(previous_best, 4)),
                    )
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)
        

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save_path = "/home/data/lxs/SSL1/SSL4MIS-master/SSL4MIS-master/SAM_model/{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logging.basicConfig(
        filename=args.save_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args)
