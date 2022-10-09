import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.linear import Linear
from ssim import MS_SSIM,SSIM
import numpy as np

global attn_ls
attn_ls=[]

class merge_loss(nn.Module):
    def __init__(self, alpha=0.15):
        super(merge_loss,self).__init__()
        self.alpha = alpha
        self.L1=nn.SmoothL1Loss()
        self.ssim=SSIM(data_range=1.0)
    def forward(self,pred,true):
        return self.alpha*(1-self.ssim(pred,true)) + (1-self.alpha)*self.L1(pred,true)

def edge_compute(x):
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y = torch.sum(y,1,keepdim=True)/3
    y /= 4
    return y

# Calculates mean and std channel-wise
def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                    # calculate std and reshape
    return mean, std

class TaskIN(nn.Module):
    def __init__(self, num_tasks, c):
        super(TaskIN, self).__init__()
        self.mean_lin = nn.Linear(num_tasks, c)
        self.std_lin = nn.Linear(num_tasks, c)
        self.c = c
        
    def forward(self, x, task_logit):
        mean, std = calc_mean_std(x)
        task_mean = self.mean_lin(task_logit).view(-1,self.c,1,1)
        task_std = self.std_lin(task_logit).view(-1,self.c,1,1)
        x = task_std*((x - mean) / (std)) + task_mean
        return x 

class TaskAttn(nn.Module):
    def __init__(self, num_tasks, c):
        super(TaskAttn, self).__init__()
        self.lin = nn.Linear(num_tasks, c)
        #self.norm = nn.LayerNorm1d(c,affine=True)
        
    def forward(self, x, task_logit):
        global attn_ls
        attn = task_logit.clone().detach()
        attn = self.lin(attn)
        attn = F.hardtanh(attn)+1
        attn_ls.append(attn.cpu().detach().numpy().tolist())
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        return x * attn,attn
        

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
        
    def forward(self, x):
        y = F.relu(x+self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))

        return F.relu(x+y)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
        
    def forward(self, x):
        y = F.relu(x+self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)
    
class DeconvBlock(nn.Module):
    def __init__(self,channel_num ,mode):
        super(DeconvBlock, self).__init__()
        self.mode=mode
        if mode==1:
            self.deconv = nn.ConvTranspose2d(channel_num, channel_num, 4, 2, 1)
            self.norm = nn.InstanceNorm2d(channel_num, affine=True)
        elif mode==2:
            self.deconv = nn.Conv2d(channel_num, channel_num, 3, 1, 1)
            self.norm = nn.InstanceNorm2d(channel_num, affine=True)
        elif mode==3:
            self.deconv = nn.Conv2d(channel_num, 3, 1)
            
    def forward(self,x):
        y = self.deconv(x)
        if self.mode == 3:
            return y
        y = F.relu(self.norm(y))
        return y


Operations_1 = [
    'TSD_2_1',
    'TSD_2_2',
    'TSD_2_3',
]
Operations_2 = [
    'TSD_3_1',
    'TSD_3_2',
    'TSD_3_3',
]

Operations_3 = [
    'TSD_4_1',
    'TSD_4_2',
    'TSD_4_3',
]

Operations_4 = [
    'TSD_1_1',
    'TSD_1_2',
    'TSD_1_3',
]

Operations_5 = [
    'DC_1_1',
    'DC_1_2',
    'DC_1_3',
]


Operations_6 = [
    'DC_2_1',
    'DC_2_2',
    'DC_2_3',
]

Operations_7 = [
    'DC_3_1',
    'DC_3_2',
    'DC_3_3',
]
OPS = {
    'TSD_2_1' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=2),
    'TSD_2_2' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=2),
    'TSD_2_3' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=2),
    'TSD_3_1' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=3),
    'TSD_3_2' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=3),
    'TSD_3_3' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=3),
    'TSD_4_1' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=4),
    'TSD_4_2' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=4),
    'TSD_4_3' : lambda channel:SmoothDilatedResidualBlock(channel,dilation=4),
    'TSD_1_1' : lambda channel:ResidualBlock(channel,dilation=1),
    'TSD_1_2' : lambda channel:ResidualBlock(channel,dilation=1),
    'TSD_1_3' : lambda channel:ResidualBlock(channel,dilation=1),
    'DC_1_1' : lambda channel:DeconvBlock(channel,mode=1),
    'DC_1_2' : lambda channel:DeconvBlock(channel,mode=1),
    'DC_1_3' : lambda channel:DeconvBlock(channel,mode=1),
    'DC_2_1' : lambda channel:DeconvBlock(channel,mode=2),
    'DC_2_2' : lambda channel:DeconvBlock(channel,mode=2),
    'DC_2_3' : lambda channel:DeconvBlock(channel,mode=2),
    'DC_3_1' : lambda channel:DeconvBlock(channel,mode=3),
    'DC_3_2' : lambda channel:DeconvBlock(channel,mode=3),
    'DC_3_3' : lambda channel:DeconvBlock(channel,mode=3),
}

## Operation layer
class OperationLayer(nn.Module):
    def __init__(self, channel, op_idx,need_ta=True):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        num_tasks = 3
        channel_num = 120
        self.need_ta=need_ta
        
        if op_idx==1:
            for o in Operations_1:
                op = OPS[o](channel)
                self._ops.append(op)
            
        elif op_idx==2:
            for o in Operations_2:
                op = OPS[o](channel)
                self._ops.append(op)
            
        elif op_idx==3:
            for o in Operations_3:
                op = OPS[o](channel)
                self._ops.append(op)
                
        elif op_idx==4:
            for o in Operations_4:
                op = OPS[o](channel)
                self._ops.append(op)
                
        elif op_idx==5:
            for o in Operations_5:
                op = OPS[o](channel)
                self._ops.append(op)
                
        elif op_idx==6:
            for o in Operations_6:
                op = OPS[o](channel)
                self._ops.append(op)
                
        elif op_idx==7:
            for o in Operations_7:
                op = OPS[o](channel)
                self._ops.append(op)
 
        if self.need_ta:
            self.task_attn = TaskAttn(num_tasks, channel_num)
        
    def forward(self, x, weights,task_logit):
        weights = weights.transpose(1,0)
        output = self._ops[0](x)*weights[0].view([-1, 1, 1, 1])+self._ops[1](x)*weights[1].view([-1, 1, 1, 1])+self._ops[2](x)*weights[2].view([-1, 1, 1, 1])
        if self.need_ta:
            ans,attn=self.task_attn(output,task_logit)
            return ans
        return output

## Operation-wise Attention Layer (OWAL)     
class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y
    
class TaskOA(nn.Module):
    def __init__(self, num_task, k,num_ops):
        super(TaskOA, self).__init__()
        self.output=k*num_ops
        self.k=k
        self.num_ops=num_ops
        self.attn_layer = nn.Sequential(
                          nn.Linear(num_task, self.output*3),
                          nn.ReLU(),
                          nn.Linear(self.output*3, self.output))

        
    def forward(self,task_logit):
        attn = self.attn_layer(task_logit)
        
        return attn.view(-1, self.k, self.num_ops)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

## a Group of operation layers
class GroupOLs(nn.Module):
    def __init__(self, steps, C, op_idx,need_ta=True):
        super(GroupOLs, self).__init__()
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        self.need_ta = need_ta
        
        for _ in range(self._steps):
            op = OperationLayer(C,op_idx,need_ta)
            self._ops.append(op)
 
    def forward(self, s0, weights,task_logit):
        x=s0
        for i in range(self._steps):
            if self.need_ta:
                res = s0
                s0 = self._ops[i](s0, weights[:, i, :],task_logit)
                if s0.size()[2]==res.size()[2]:
                    s0 = F.relu(s0+res)
            else:
                s0 = self._ops[i](s0, weights[:, i, :],task_logit)
        if s0.size()[2]==x.size()[2] and self.need_ta:
            return F.relu(s0+x)
        return s0

class GCAEncoder(nn.Module):
    def __init__(self, in_c=4):
        super(GCAEncoder, self).__init__()
        filter_num = 120
        self.conv1 = nn.Conv2d(in_c, filter_num, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(filter_num, affine=True)
        self.conv2 = nn.Conv2d(filter_num, filter_num, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(filter_num, affine=True)
        self.conv3 = nn.Conv2d(filter_num, filter_num, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(filter_num, affine=True)
        
    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))
        
        return y1
    
class GCADecoder(nn.Module):
    def __init__(self, out_c=3,num_tasks=3, only_residual=True):
        super(GCADecoder, self).__init__()
        filter_num = 120

        self._layer_num=7
        self._steps=3
        self.num_ops=len(Operations_1)
        self.num_task=3
        self.oa_1 = TaskOA(self.num_task,self._steps, self.num_ops)
        self.layer_1 = GroupOLs(self._steps, filter_num,1)
        self.oa_2 = TaskOA(self.num_task,self._steps, self.num_ops)
        self.layer_2 = GroupOLs(self._steps, filter_num,2)
        self.oa_3 = TaskOA(self.num_task,self._steps, self.num_ops)
        self.layer_3 = GroupOLs(self._steps, filter_num,3)
        self.oa_4 = TaskOA(self.num_task,self._steps, self.num_ops)
        self.layer_4 = GroupOLs(self._steps, filter_num,4)

        self.gate = nn.Conv2d(filter_num * 5, 5, 3, 1, 1, bias=True)

        self.layer_5 = GroupOLs(1, filter_num,5)
        self.oa_5 = TaskOA(self.num_task,1, self.num_ops)
        self.layer_6 = GroupOLs(1, filter_num,6)
        self.oa_6 = TaskOA(self.num_task,1, self.num_ops)
        self.layer_7 = GroupOLs(1, filter_num,7,need_ta=False)
        self.oa_7 = TaskOA(self.num_task,1, self.num_ops)
        
        self.only_residual = only_residual
        
    def forward(self, y1, task_logit):

        weights_dict={}
        
        weights_dict['oa_weights1']=self.oa_1(task_logit)
        y2=self.layer_1(y1,weights_dict['oa_weights1'],task_logit)
        weights_dict['oa_weights2']=self.oa_2(task_logit)
        y3=self.layer_2(y2,weights_dict['oa_weights2'],task_logit)
        weights_dict['oa_weights3']=self.oa_3(task_logit)
        y4=self.layer_3(y3,weights_dict['oa_weights3'],task_logit)
        weights_dict['oa_weights4']=self.oa_4(task_logit)
        y5=self.layer_4(y4,weights_dict['oa_weights4'],task_logit)

        gates = self.gate(torch.cat((y1, y2, y3 ,y4 ,y5), dim=1))
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]+ y4 * gates[:, [3], :, :] + y5 * gates[:, [4], :, :]
        weights_dict['oa_weights5']=self.oa_5(task_logit)
        y=self.layer_5(gated_y,weights_dict['oa_weights5'],task_logit)
        weights_dict['oa_weights6']=self.oa_6(task_logit)
        y=self.layer_6(y,weights_dict['oa_weights6'],task_logit)
        weights_dict['oa_weights7']=self.oa_7(task_logit)
        y=self.layer_7(y,weights_dict['oa_weights7'],task_logit)
        
        if self.only_residual:
            return weights_dict
        else:
            return F.relu(y)
        
# -------------------------------------------------------------------------------------------------
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class TaskClassifier(nn.Module):
    def __init__(self, in_c, num_tasks=3):
        super(TaskClassifier, self).__init__()
        filter_num = 120
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 48, 16, hswish(), None, 1),
            Block(3, 16, 64, 24, hswish(), None, 1),
            Block(3, 24, 72, 48, hswish(), None, 1),
            Block(3, 48, 106, filter_num, hswish(), SeModule(filter_num), 1),
            Block(3, filter_num, 128, filter_num, hswish(), SeModule(filter_num), 1),
        )
        
        self.conv2 = nn.Conv2d(filter_num, filter_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(filter_num)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(filter_num*4, filter_num)
        self.bn3 = nn.BatchNorm1d(filter_num)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(filter_num, num_tasks)
        self.init_params()
        
    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, (2,2))
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = F.dropout(out,p=0.75)
        out = self.linear4(out)
        return out
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# -------------------------------------------------------------------------------------------------

class GCANet(nn.Module):
    def __init__(self, in_c=4, out_c=3,num_tasks=3, only_residual=True):
        super(GCANet, self).__init__()
        self.encoder = GCAEncoder(in_c=in_c)
        self.decoder = GCADecoder(out_c=out_c,num_tasks=3, only_residual=only_residual)
        
        filter_num = 120
        self.task_cls1 = TaskClassifier(filter_num,num_tasks)
        #self.task_cls2 = TaskClassifier(filter_num,num_tasks)

    def forward(self, x):
        global attn_ls
        edge = edge_compute(x)
        x_new = torch.cat([x,edge],dim=1)
        
        out = self.encoder(x_new)
        task_x = out.clone().detach()
        
        task_logit1 = self.task_cls1(task_x)
        task_prob1 = F.softmax(task_logit1,dim=-1)
        #task_logit2 = self.task_cls2(task_x)
        #task_prob2 = F.softmax(task_logit2,dim=-1)
        #task_logit = torch.cat([task_logit1, task_logit2],dim=-1)
        #task_logit = torch.cat([task_prob1, task_prob2],dim=-1)
        
        #task_logit = torch.cat([task_logit, F.cosine_similarity(task_logit1, task_logit2).reshape(-1,1)],dim=-1)
        #task_logit = torch.cat([task_logit, F.cosine_similarity(task_prob1, task_prob2).reshape(-1,1)],dim=-1)
    
        weights_dict = self.decoder(out, task_logit1)
        for key in weights_dict:
            if len(weights_dict[key].shape)==3:
                weights_dict[key]=weights_dict[key].cpu().detach().numpy()
            else:
                weights_dict[key]=weights_dict[key].cpu().detach().numpy()
        temp=attn_ls
        attn_ls=[]
        return weights_dict,temp
        
if __name__ == '__main__':
    model = GCANet()
    model = model.cuda()
    x = torch.rand((1,3,100,100)).cuda()
    weights_dict,attn_ls = model(x)
    print(weights_dict)
    print(np.array(attn_ls).shape)