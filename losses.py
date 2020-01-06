import torch.nn as nn
import torch
import torchvision

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        #probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    
class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        cat_class=targets.size(1)
        m1 = logits.view(num,cat_class, -1)
        m2 = targets.view(num,cat_class, -1)
        final_loss=0
        loss_list=[]
        weights=[0.6,0.4]
        for cat in range(cat_class):
            if cat==1:
                loss=nn.BCELoss()(m1[:,cat,:],m2[:,cat,:])
            else:
                loss=nn.BCELoss()(m1[:,cat,:],m2[:,cat,:])
            final_loss+=weights[cat]*loss
            

        return final_loss
    
def dice_metric(y_pred,y_true):
    smooth = 1
    num = y_true.size(0)
    categories=y_true.size(1)
    m1 = y_pred.view(num,categories, -1)
    m2 = y_true.view(num,categories, -1)
    weights=[0.5,0.5]
    final_score=0
    score_list=[]
    for cat in range(categories):
        
        
        intersection = (m1[:,cat,:] * m2[:,cat,:])

        score = 2. * (intersection.sum(1) + smooth) / (m1[:,cat,:].sum(1) + m2[:,cat,:].sum(1) + smooth)
        score = score.sum() / num
        score=score.detach().item()
        final_score+=score*weights[cat]
        score_list.append(score)
    return final_score,score_list
    