import numpy as np
from math import *
import os
eps=2.2204e-16
def ListFile(path,postfix):
    filelist = []
    files = os.listdir(path)
    for item in files:
        if os.path.isfile(path+item):
            if item[-3:]==postfix:
                filelist.append([path,item])
    return filelist
#StructureMeasure
def _S_object(pred, gt):
    #fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    fg=(gt==1)*pred
    #bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
    bg=(gt==0)*(1-pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q

def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + eps) 
    return score

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    # print(Q)
    return Q
    
def _centroid(gt):
    rows, cols = gt.shape[:2]
    if gt.sum() == 0:
        return round(cols/2),round(rows/2)
    else:
        total = gt.sum()
        i = np.arange(1,cols+1)
        j=np.arange(1,rows+1)
        X = round((gt.sum(0)*i).sum() / total)
        Y = round((gt.sum(1)* j).sum() / total)
    return int(X), int(Y)
    
def _divideGT( gt, X, Y):
    h, w = gt.shape[0:2]
    area = float(h*w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    #print(w1, w2, w3, w4)
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction( pred, X, Y):
    h, w = pred.shape[0:2]
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim( pred, gt):
    h, w = pred.shape[0:2]
    N = h*w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + eps)
    sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + eps)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + eps)
    aplha = 4 * x * y *sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + eps)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
def StructureMeasure(pred, gt):
    alpha = 0.5
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        Q = alpha * _S_object(pred, gt) + (1-alpha) * _S_region(pred, gt)
    return Q
if __name__=='__main__':
    a=np.random.rand(40,40)
    a=a>a.max()/2
    s=StructureMeasure(a,a)
    print(s)