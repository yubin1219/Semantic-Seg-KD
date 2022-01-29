import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, args_s, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_label).to(device)

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=args_s.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss
    
    def forward(self, score, target):
        weights = args_s.LOSS.BALANCE_WEIGHTS

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

def CAM(x):
    out = F.normalize(x.pow(2).reshape(19,-1),dim=1)
    return out

class CriterionCAT(nn.Module):
    def __init__(self):
        super(CriterionCAT, self).__init__()
        self.CAM = CAM
    
    def forward(self, fs, ft, ot_t): 
        n = ft.size(0)
        m_t = F.softmax(ot_t,dim=1)
        m_t[m_t > 0.5] = 1
        norm_t = F.normalize(ft.pow(2).reshape(n,512,-1),dim=2).reshape(n,512,128,256)

        map_list_T=[]
        map_list_S=[]

        for i in range(19):
            mask_t = m_t[:,i,:,:].unsqueeze(1)   
            weight_t = torch.nn.AdaptiveAvgPool2d(1)(norm_t * mask_t) + 1e-8   # n x 512 x 1 x 1
            wt_max = weight_t.max(1)[0].unsqueeze(1)
            w_t = weight_t / wt_max    # 0 ~ 1

        att_t = (w_t * ft).mean(1).unsqueeze(1)
        att_s = (w_t * fs).mean(1).unsqueeze(1)
        map_list_T.append(att_t)
        map_list_S.append(att_s)

        out_t = torch.cat(map_list_T,dim=1)
        out_s = torch.cat(map_list_S,dim=1)

        loss = sum([(self.CAM(x)-self.CAM(y)).pow(2).sum() for x,y in zip(out_s, out_t)]) / n
        return loss

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, upsample = False, temperature = 4):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft):
        soft.detach()
        h, w = soft.size(2), soft.size(3)
        if self.upsample:
            scale_pred = F.interpolate(input=pred, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
            scale_soft = F.interpolate(input=soft, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
        else:
            scale_pred = pred
            scale_soft = soft
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.temperature, dim=1), F.softmax(scale_soft / self.temperature, dim=1))
        
        return loss * self.temperature * self.temperature
      
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap

class CriterionCWD(nn.Module):
    def __init__(self, norm_type='channel', divergence='kl', temperature = 4.0):    
        super(CriterionCWD, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self,preds_S, preds_T):        
        n,c,h,w = preds_S.shape

        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S
            norm_t = preds_T.detach()       
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)
        
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)

def at(x):
  return F.normalize(x.pow(2).mean(0).reshape(1,-1), dim= 1)

class CriterionAT(nn.Module):
    def __init__(self):
        super(CriterionAT, self).__init__()
        self.at = at
    
    def forward(self, fs, ft):
        n = ft.size(0)
        loss = sum([(self.at(x)-self.at(y)).pow(2).sum() for x,y in zip(fs, ft)]) / n
        return loss    
