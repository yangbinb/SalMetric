import numpy as np
import cv2
from PIL import Image
import os

beta = np.sqrt(0.3)
data_root = '/home/yangbinbin/DSS-master/MSRA-B/'
result_root = '/home/yangbinbin/DSS-master/Result/'
with open('/home/yangbinbin/DSS-master/test_mask.lst') as f:
    test_lst = f.readlines()

writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'w')
writefile.write('\n')

lst = [x.strip() for x in test_lst]
gt_lst = [data_root+x.strip() for x in test_lst]
mask_lst = [result_root+x.strip() for x in test_lst]
data_size = len(gt_lst)

gts = []
masks = []
for idx in range(0, data_size):
    gt = cv2.imread(gt_lst[idx], 0)
    mask = cv2.imread(mask_lst[idx], 0)
    gt = np.array(gt)
    mask = np.array(mask)
    gts.append(gt)
    masks.append(mask)

gts = np.array(gts)
masks = np.array(masks)

def eval_mae(pred, gt):
    return np.abs(pred - gt).mean()

def eval_one_threshold(pred, gt, threshold):
    mask = (pred > threshold)
    gt = np.array((gt == 255), dtype=np.float)
    tp = (mask * gt).sum() + 1e-5
    prec, recall = tp / (mask.sum() + 1e-5), tp / gt.sum()
    # F_beta = (1 + beta ** 2) * prec * recall / (beta ** 2 * prec + recall)
    return prec, recall

def eval_mean_threshold(pred, gt):
    threshold = 2 * pred.mean()
    if threshold > 255:
        threshold = 255
    mask = (pred > threshold)
    gt = np.array((gt == 255), dtype=np.float)
    tp = (mask * gt).sum() + 1e-5
    prec, recall = tp / (mask.sum() + 1e-5), tp / gt.sum()
    F_beta = (1 + beta ** 2) * prec * recall / (beta ** 2 * prec + recall)
    return F_beta

def compute_max_F_measure():
    prec_list = []
    recall_list = []
    F_beta_list = []

    # compute mae
    avg_mae = 0.0
    for idx in range(0, data_size):
        gt = gts[idx]
        mask = masks[idx]
        gt = np.array(gt, dtype=np.float32)/255.0
        mask = np.array(mask, dtype=np.float32)/ 255.0
        mae = eval_mae(mask, gt)
        avg_mae += mae
    avg_mae /= data_size

    for threshold in range(0, 256):
        one_thresh_prec = 0.0
        one_thresh_recall = 0.0
        for idx in range(0, data_size):
            gt = gts[idx] #cv2.imread(gt_lst[idx], 0)  # 0~255
            mask = masks[idx] #cv2.imread(mask_lst[idx], 0)  # 0~255
            temp_prec, temp_recall = eval_one_threshold(mask, gt, threshold)
            one_thresh_prec += temp_prec
            one_thresh_recall += temp_recall
        # First we compute the average prec and recall and then comput F-measure
        one_thresh_prec /= data_size
        one_thresh_recall /= data_size
        one_thresh_F_beta = (1 + beta ** 2) * one_thresh_prec * one_thresh_recall / (
                    beta ** 2 * one_thresh_prec + one_thresh_recall)
        print('Threshold %d: MAE: %f, Precision: %f, Recall: %f, Fmeasure: %f' % (threshold, avg_mae, one_thresh_prec, one_thresh_recall, one_thresh_F_beta))
        prec_list.append(one_thresh_prec)
        recall_list.append(one_thresh_recall)
        F_beta_list.append(one_thresh_F_beta)
        writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'a+')
        writefile.write('%f %f %f %f\n' % (avg_mae, one_thresh_prec, one_thresh_recall, one_thresh_F_beta))
    prec_list = np.array(prec_list)
    recall_list = np.array(recall_list)
    F_beta_list = np.array(F_beta_list)
    index = np.argmax(F_beta_list)
    print('Max F-measure: %f' % F_beta_list[index])
    print('Precision: %f' % prec_list[index])
    print('Recall: %f' % recall_list[index])
    print('MAE: %f' % avg_mae)
    # print('\n max F: %f, prec: %f, recall: %f' % (F_beta_list[index], prec_list[index], recall_list[index]))
    writefile = open('/home/yangbinbin/DSS-master/store_eval.txt', 'a+')
    writefile.write('%f %f %f %f\n' % (F_beta_list[index], prec_list[index], recall_list[index], avg_mae))
    writefile.close()

def compute_mean_F_measure():
    F_beta_measure = 0.0
    for idx in range(0, data_size):
        gt = gts[idx]
        mask = masks[idx]
        F_measure = eval_mean_threshold(mask, gt)
        F_beta_measure += F_measure
    F_beta_measure /= data_size
    print('Mean F-measure: %f' % F_beta_measure)

compute_max_F_measure()
compute_mean_F_measure()