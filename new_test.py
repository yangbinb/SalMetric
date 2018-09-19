import numpy as np
import cv2
from PIL import Image
import os
import threading

beta = np.sqrt(0.3)

NUM_THREADS = 40

def eval_mae(pred, gt):
    return np.abs(pred - gt).mean()

writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'w')
writefile.write('\n')

data_root = '/home/yangbinbin/DSS-master/MSRA-B/'
result_root = '/home/yangbinbin/DSS-master/Result/'
with open('/home/yangbinbin/DSS-master/test_mask.lst') as f:
    test_lst = f.readlines()

lst = [x.strip() for x in test_lst]
gt_lst = [data_root+x.strip() for x in test_lst]
mask_lst = [result_root+x.strip() for x in test_lst]

avg_mae = 0.0
avg_prec = 0.0
avg_recall = 0.0
avg_F_beta = 0.0

prec_list = []
recall_list = []
F_beta_list = []

data_size = len(gt_lst)

def eval_one_threshold(pred, gt, threshold):
    mask = (pred > threshold)
    gt = np.array((gt == 255), dtype=np.float)
    tp = (mask * gt).sum() + 1e-5
    prec, recall = tp / (mask.sum() + 1e-5), tp / gt.sum()
    # F_beta = (1 + beta ** 2) * prec * recall / (beta ** 2 * prec + recall)
    return prec, recall


for threshold in range(0, 256):
    one_thresh_prec = 0.0
    one_thresh_recall = 0.0
    for idx in range(0, data_size):
        gt = gt_lst[idx]
        mask = mask_lst[idx]
        gt = cv2.imread(gt_lst[idx], 0)  # 0~255
        mask = cv2.imread(mask_lst[idx], 0)  # 0~255
        temp_prec, temp_recall = eval_one_threshold(mask, gt, threshold)
        one_thresh_prec += temp_prec
        one_thresh_recall += temp_recall
    one_thresh_prec /= data_size
    one_thresh_recall /= data_size
    # print one_thresh_prec
    # print one_thresh_recall
    one_thresh_F_beta = (1 + beta ** 2) * one_thresh_prec * one_thresh_recall / (beta ** 2 * one_thresh_prec + one_thresh_recall)
    print('prec, recall, F-measure for threshold %d'%threshold)
    print one_thresh_prec
    print one_thresh_recall
    print one_thresh_F_beta
    print('\n')
    # avg_prec += one_thresh_prec
    # avg_recall += one_thresh_recall
    # avg_F_beta += one_thresh_F_beta
    prec_list.append(one_thresh_prec)
    recall_list.append(one_thresh_recall)
    F_beta_list.append(one_thresh_F_beta)
    writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'a+')
    writefile.write('%f %f %f\n'%(one_thresh_prec, one_thresh_recall, one_thresh_F_beta))

# avg_prec /= 256
# avg_recall /= 256
# avg_F_beta /= 256
# F_avg_beta = (1 + beta ** 2) * avg_prec * avg_recall / (beta ** 2 * avg_prec + avg_recall)

# print('\n\navg mae: %f, avg prec: %f, avg recall: %f, avg F beta: %f, F_avg_beta: %f' % (avg_mae, avg_prec, avg_recall, avg_F_beta, F_avg_beta))

prec_list = np.array(prec_list)
recall_list = np.array(recall_list)
F_beta_list = np.array(F_beta_list)

index = np.argmax(F_beta_list)

print('\n max F: %f, prec: %f, recall: %f' % (F_beta_list[index], prec_list[index], recall_list[index]))

writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'a+')
writefile.write('%f %f %f\n'%(F_beta_list[index], prec_list[index], recall_list[index]))
writefile.close()