from dataset import *
from model_abla1 import GCANet,merge_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import psnr, ssim, accuracy
import argparse
from tqdm import tqdm
import torchvision
import pytorch_warmup as warmup
import pandas as pd

def w_schedule_cosdecay(t,T,init_w=0.4):
    w=0.5*(1+math.cos(t*math.pi/T))*init_w
    return w  

def train(args, model):
    global epoch_cnt
    model.train()
    train_list = populate_train_list(args.train_orig_images_path, args.train_degen_images_path)
    train_dataset = merge_dataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    reg_criterion = merge_loss()
    cls_criterion = F.cross_entropy
    
    num_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=num_steps//10)
    for epoch in range(1,args.num_epochs+1):
        train_reg_loss, train_cls_loss,  train_psnr, train_ssim, train_acc = 0, 0, 0, 0, 0
        print(f'epoch {epoch}')
        epoch_cnt=epoch
        for step, batch in enumerate(tqdm(train_dataloader)):
            
            data_orig, data_degen, label = [i.cuda() for i in batch]
            
            pred, logit, prob = model(data_degen)
            
            
            optimizer.zero_grad()
            reg_loss = reg_criterion(pred, data_orig)
            cls_loss = cls_criterion(logit, label)
            loss = reg_loss + w_schedule_cosdecay(epoch,args.num_epochs+1)*cls_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step(lr_scheduler.last_epoch+1)
            warmup_scheduler.dampen()
            
            train_reg_loss += reg_loss.item()
            train_cls_loss += cls_loss.item()
            train_psnr += psnr(pred, data_orig).item()
            train_ssim += ssim(pred, data_orig).item()
            train_acc += accuracy(prob, label).item()
            if (step+1) % args.log_step == 0:
                print('lr: %.8f, train_reg_loss: %.6f, train_cls_loss: %.6f, train_psnr: %.6f, train_ssim: %.6f, train_acc: %.6f'%
                      (optimizer.param_groups[0]['lr'],train_reg_loss/args.log_step,train_cls_loss/args.log_step,train_psnr/args.log_step,train_ssim/args.log_step,train_acc/args.log_step))
                
                train_reg_loss, train_cls_loss,  train_psnr, train_ssim, train_acc = 0, 0, 0, 0, 0
        
        torch.save(model.state_dict(), '/openbayes/home/snapshots_abla1/model_%d.pth' % int(epoch))
        for m in ['rain','hazy','rain_drop_test_a','rain_drop_test_b']:
            test(args,model,m)
            

def test(args, model, mode):
    global epoch_cnt
    global table
    model.eval()
    print('testing '+mode)
    with torch.no_grad():
        val_list = populate_test_list(args.test_orig_images_path, args.test_degen_images_path, mode)
        val_dataset = merge_dataset(val_list, train=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
        reg_criterion = merge_loss()
        cls_criterion = nn.CrossEntropyLoss()
        val_reg_loss, val_cls_loss, val_psnr, val_ssim, val_acc = 0, 0, 0, 0, 0
        l = len(val_dataloader)
        for step, batch in enumerate(tqdm(val_dataloader)):
            
            data_orig, data_degen, label = [i.cuda() for i in batch]
            
            pred, logit, prob = model(data_degen)
            reg_loss = reg_criterion(pred, data_orig)
            cls_loss = cls_criterion(logit, label)
            
            val_reg_loss += reg_loss.item()
            val_cls_loss += cls_loss.item()
            val_psnr += psnr(pred, data_orig).item()
            val_ssim += ssim(pred, data_orig).item()
            val_acc += accuracy(prob, label).item()
            
            torchvision.utils.save_image(torch.cat((data_degen, pred, data_orig),0), getattr(args,'sample_output_folder_'+mode)+str(step+1)+".jpg")
        
        print('val_reg_loss: %.6f, val_cls_loss: %.6f,val_psnr: %.6f, val_ssim: %.6f, val_acc: %.6f'%
                      (val_reg_loss/l,val_cls_loss/l,val_psnr/l,val_ssim/l,val_acc/l))
        table=table.append({'epoch':epoch_cnt,'mode':mode,'psnr':val_psnr/l,'ssim':val_ssim/l,'acc':val_acc/l},ignore_index=True)
        table.to_csv('/openbayes/home/experiment1.csv')
        model.train()
        
if __name__ == "__main__":
    global table
    global epoch_cnt
    table=pd.DataFrame(columns=['epoch','mode','psnr','ssim','acc1','acc2'])
    epoch_cnt=1
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_orig_images_path",default='/openbayes/input/input1/clear/',type=str)
    ap.add_argument("--train_degen_images_path",default='/openbayes/input/input1/degen/',type=str)
    ap.add_argument("--test_orig_images_path",default='/openbayes/input/input1/clear/',type=str)
    ap.add_argument("--test_degen_images_path",default='/openbayes/input/input1/degen/',type=str)
    ap.add_argument("--num_epochs",default=20,type=int)
    ap.add_argument("--batch_size",default=6,type=int)
    ap.add_argument("--num_workers",default=6,type=int)
    ap.add_argument("--lr",default=2e-4,type=float)
    ap.add_argument("--weight_decay",default=8e-4,type=float)
    ap.add_argument("--log_step",default=100,type=int)
    ap.add_argument("--sample_output_folder_rain",default='/openbayes/home/samples_abla1/rain/',type=str)
    ap.add_argument("--sample_output_folder_hazy",default='/openbayes/home/samples_abla1/hazy/',type=str)
    ap.add_argument("--sample_output_folder_rain_drop_test_a",default='/openbayes/home/samples_abla1/rain_drop_test_a/',type=str)
    ap.add_argument("--sample_output_folder_rain_drop_test_b",default='/openbayes/home/samples_abla1/rain_drop_test_b/',type=str)
    args = ap.parse_args()
    
    model = GCANet()
    model = model.cuda()
    #PATH="/openbayes/home/snapshots/model_30.pth"
    #model.load_state_dict(torch.load(PATH))
    train(args, model)