from model_debug import *
import torch
import torch.nn as nn
from win_dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import psnr, ssim, accuracy

class Args(object):
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 0
        self.test_orig_images_path = '../data/merge_test/clear/'
        self.test_degen_images_path = '../data/merge_test/degen/'

class Debuger(object):
    def __init__(self):
        self.attns_dict = {
            'res1_1':[],
            'res6_1':[]
        }
        self.INs_dict = {}
    def __call__(self,model):
        self.attns_dict['res1_1'].append(model.decoder.res1.task_attn1.attn)
        self.attns_dict['res6_1'].append(model.decoder.res6.task_attn1.attn)

def test(args, model, mode):
    model.eval()
    print('testing '+mode)
    debuger = Debuger()
    with torch.no_grad():
        val_list = populate_test_list(args.test_orig_images_path, args.test_degen_images_path, mode)
        val_dataset = merge_dataset(val_list, train=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
        reg_criterion = nn.L1Loss()
        cls_criterion = nn.CrossEntropyLoss()
        val_reg_loss, val_cls_loss1, val_cls_loss2,  val_psnr, val_ssim, val_acc1, val_acc2 = 0, 0, 0, 0, 0, 0, 0
        l = len(val_dataloader)
        for step, batch in enumerate(tqdm(val_dataloader)):
            data_orig, data_degen, label = [i.cuda() for i in batch]
            
            pred, logit1, logit2, prob1, prob2 = model(data_degen)

            debuger(model)
            reg_loss = reg_criterion(pred, data_orig)
            cls_loss1 = cls_criterion(logit1, label)
            cls_loss2 = cls_criterion(logit2, label)
            
            val_reg_loss += reg_loss.item()
            val_cls_loss1 += cls_loss1.item()
            val_cls_loss2 += cls_loss2.item()
            val_psnr += psnr(pred, data_orig).item()
            val_ssim += ssim(pred, data_orig).item()
            val_acc1 += accuracy(prob1, label).item()
            val_acc2 += accuracy(prob2, label).item()
        
        print('val_reg_loss: %.6f, val_cls_loss1: %.6f, val_cls_loss2: %.6f,val_psnr: %.6f, val_ssim: %.6f, val_acc1: %.6f, val_acc2: %.6f'%
                      (val_reg_loss/l,val_cls_loss1/l,val_cls_loss2/l,val_psnr/l,val_ssim/l,val_acc1/l, val_acc2/l))
        
        model.train()
    return debuger

model = GCANet()
model.load_state_dict(torch.load('./model_20.pth'))
model = model.cuda()

args = Args()
debuger = test(args,model,'hazy')