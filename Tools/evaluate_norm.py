import scipy.io
import torch
import numpy as np
import time
import argparse
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
from scipy.stats import norm

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--test_weight', default='xx.mat', type=str)
parser.add_argument('--save_path', type=str, help='Your path')
parser.add_argument('--mode', default='D2S', type=str)
opt = parser.parse_args()


# --------------------- 读取保存的结果文件 --------------------- #
mat_file = f'{opt.save_path}'

print("--" * 50)
print("load from:", mat_file)


def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  # predict index from small to large
    index = index[::-1]
    query_index = np.argwhere(gl == ql)  # 正确的索引
    good_index = query_index
    junk_index = np.argwhere(gl == -1)  # 垃圾索引
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc, index, good_index


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc



# ------------------------ 在 mat 文件中存储了 特征和特征对应的标签 ------------------------ #
result = scipy.io.loadmat(mat_file)
query_feature = torch.FloatTensor(result['query_f']).cuda()
query_label = result['query_label'][0]  # 0-836

gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
gallery_label = result['gallery_label'][0]
gallery_label_light = gallery_label[::54]  # 0-1651


# test_name = result['name']
# query_name = result['query_name']
# gallery_name = result['gallery_name']
print("query feature:", query_feature.shape, "gallery feature:", gallery_feature.shape)

dismat = 1 - torch.mm(query_feature, gallery_feature.t())
dismat = dismat.cpu().numpy()
pos = []
neg = []

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

print('calculate distance')
since = time.time()

for i, q in enumerate(query_label):
    ap_tmp, CMC_tmp, indices, matches = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)

    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

    order = indices
    cmc = CMC_tmp
    sort_idx = order
    ind_pos = np.where(cmc == 0)[0]
    q_dist = dismat[i]
    pos.extend(q_dist[sort_idx[ind_pos]])
    ind_neg = np.where(cmc == 1)[0]
    neg.extend(q_dist[sort_idx[ind_neg]])

scores = np.hstack((pos, neg))
labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))
fpr, tpr, thresholds = metrics.roc_curve(labels, scores)



time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100, ap / len(query_label) * 100))
print("--" * 50 + '\n')
