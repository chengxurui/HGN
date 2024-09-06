import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
import argparse
from sklearn.model_selection import KFold
import numpy as np
import link_prediction_model as LP
import node_classification_model as NP
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sklearn.metrics as Metric
from sklearn.metrics import roc_curve,auc
import ds
import torch

parser = argparse.ArgumentParser(description='input data process')
parser.add_argument('--S_n', type=int, default=251,
                    help='number of SM node')
parser.add_argument('--M_n', type=int, default=901,
                    help='number of miRANs node')
parser.add_argument('--D_n', type=int, default=361,
                    help='number of disease node')
parser.add_argument('--C_n', type=int, default=4,
                    help='number of node class label')
parser.add_argument('--data_path', type=str, default='D:/codee/data/abide1/',
                    help='path to data')
parser.add_argument('--walk_n', type=int, default=20,
                    help='number of walk per root node')
parser.add_argument('--walk_L', type=int, default=30,
                    help='length of each walk')
parser.add_argument('--window', type=int, default=7,
                    help='window size for relation extration')
parser.add_argument('--T_split', type=int, default=2012,
                    help='split time of train/test data')
parser.add_argument('--embed_d', type=int, default=1950,
                    help='embedding dimension')

args = parser.parse_args()


def a_p_cite_feature_settingFC(train_set,test_set):
    d_embed = np.around(np.random.normal(0, 0.01, [51607, args.embed_d]), 4)
    print('de:',d_embed.shape) #  (51607, 1950)
    # l_embed = np.around(np.random.normal(0, 0.01, [args.M_n, args.embed_d]), 4)
    embed_f = open(args.data_path + "node_ag14.txt", "r")
    # m=0
    for line in islice(embed_f, 0, None):
        # m+=1
        # if m <600:
        line = line.strip()
        node_id = re.split(' ', line)[0]
        # print('no:',node_id)
        if len(node_id) and (node_id[0] in ('s', 'm', 'd')):
            type_label = node_id[0]
            index = int(node_id[1:])
            embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
            if type_label == 's':
                d_embed[index] = embed
            # elif type_label == 'm':
            #     l_embed[index] = embed
    print(d_embed)
    embed_f.close()

    train_num = 0
    d_l_cite_list_train_feature_f = open(args.data_path + "train_feature.txt", "w")
    for h in range(len(train_set)):
        d_1 = int(train_set[h][0])
        # l_2 = int(train_set[h][1])
        label = int(train_set[h][1])
        if random.random() < 1: # training data ratio
            train_num += 1
            d_l_cite_list_train_feature_f.write("%d, %d," % (d_1, label))
            for d in range(args.embed_d-1):
              d_l_cite_list_train_feature_f.write("%f," % (d_embed[d_1][d]))
            d_l_cite_list_train_feature_f.write("%f" % (d_embed[d_1][args.embed_d - 1]))
            # d_l_cite_list_train_feature_f.write(
            #     "%f" % (2*d_embed[d_1][args.embed_d - 1] * l_embed[l_2][args.embed_d - 1]))
            d_l_cite_list_train_feature_f.write("\n")
    d_l_cite_list_train_feature_f.close()

    test_num = 0
    d_l_cite_list_test_feature_f1 = open(args.data_path + "test_feature.txt", "w")
    for h in range(len(test_set)):
        d_1 = int(test_set[h][0])
        # l_2 = int(test_set[h][1])
        label = int(test_set[h][1])
        test_num += 1
        d_l_cite_list_test_feature_f1.write("%d, %d," % (d_1, label))
        for d in range(args.embed_d - 1):
            d_l_cite_list_test_feature_f1.write("%f," % (d_embed[d_1][d]))
        d_l_cite_list_test_feature_f1.write("%f" % (d_embed[d_1][args.embed_d - 1]))
        # d_l_cite_list_train_feature_f.write(
        #     "%f" % (2*d_embed[d_1][args.embed_d - 1] * l_embed[l_2][args.embed_d - 1]))
        d_l_cite_list_test_feature_f1.write("\n")
    d_l_cite_list_test_feature_f1.close()

    print("a_p_cite_train_num: " + str(train_num))
    print("a_p_cite_test_num: " + str(test_num))

    return train_num, test_num
def s_m_cite_train_testFC():
    # s_m_not_list_total = [[] for k in range(args.S_n)]
    # f_name = "s_m_not_list_total.txt"
    # neigh_f = open(args.data_path + "total/" + f_name, "r")
    # for line in neigh_f:
    #     line = line.strip()
    #     node_id = int(re.split(':', line)[0])
    #     neigh_list = re.split(':', line)[1]
    #     neigh_list_id = re.split(',', neigh_list)
    #     if f_name == 's_m_not_list_total.txt':
    #         for j in range(len(neigh_list_id)):
    #             s_m_not_list_total[node_id].append(neigh_list_id[j])
    # neigh_f.close()
    # s_m_list_total = [[] for k in range(args.S_n)]
    # f_name = "s_m_list_total.txt"
    # neigh_f = open(args.data_path + "total/" + f_name, "r")
    # for line in neigh_f:
    #     line = line.strip()
    #     node_id = int(re.split(':', line)[0])
    #     neigh_list = re.split(':', line)[1]
    #     neigh_list_id = re.split(',', neigh_list)
    #     if f_name == 's_m_list_total.txt':
    #         for j in range(len(neigh_list_id)):
    #             s_m_list_total[node_id].append(neigh_list_id[j])
    # neigh_f.close()
    # s_m_cite_list_total_f = open(args.data_path + "s_m_cite_list_total.txt", "w")
    # for i in range(len(s_m_list_total)):
    #     for j in range(len(s_m_list_total[i])):
    #         l_id = s_m_list_total[i][j]
    #         s_m_cite_list_total_f.write("%d, %d, %d\n" % (i, int(l_id), 1))
    #         nl_id = random.choice(s_m_not_list_total[i])
    #         s_m_cite_list_total_f.write("%d, %d, %d\n" % (i, int(nl_id), 0))
    # s_m_cite_list_total_f.close()
    # fname = open(args.data_path + "/s_m_cite_list_total.txt", "r")
    # total=[]
    # for line in fname:
    #     line.strip()
    #     s_id = int(re.split(',', line)[0])
    #     m_id=int(re.split(',', line)[1])
    #     label=int(re.split(',', line)[2])
    #     total.append([s_id,m_id,label])
    # fname.close()
    # total=np.array(total)
    # fold=5
    # kf=KFold(n_splits=fold)
    # i=0
    # auc_total=0
    # auc2=0
    # prei_total=0
    # recall_total=0
    # f1_total=0
    # accuracy_total = 0
    # spec_total=0
    # test_target = [[] for k in range(5)]
    # test_predict = [[] for k in range(5)]
    # for train_index,test_index in kf.split(total):
    #     i=i+1
    #     print(i)

    # train_set=total[train_index]
    # test_set=total[test_index]
    subject_IDs = ds.get_ids()
    subject_IDs_ = ds.get_id1()
    labels = ds.get_subject_score(subject_IDs_, score='DX_GROUP')
    y = np.zeros([len(subject_IDs_), 1])
    for i in range(len(subject_IDs_)):
        y[i] = int(labels[subject_IDs_[i]])
    y = torch.tensor(y - 1, dtype=int)
    y = np.squeeze(y)
    # print(y)
    train_set = [[] for k in range(782)]
    for i in range(len(subject_IDs)):
        if i < 782:
            train_set[i].append(int(subject_IDs[i]))
            train_set[i].append(int(y[i]))
    print('de:', train_set)
    k=0
    test_set = [[] for k in range(89)]
    for i in range(len(subject_IDs)):
        if i > 781 :
            test_set[k].append(int(subject_IDs[i]))
            # print('tes:',test_set)
            test_set[k].append(int(y[i]))
            k += 1
    # print('tes:', test_set)

    train_num, test_num = a_p_cite_feature_settingFC(train_set,test_set) # setup of author-paper citation prediction task
    auc1, f1, accuracy= NP.model(train_num, test_num)
    auc_total+=auc1
    # prei_total+=prei
    if i==3:
        auc2=auc1
    # recall_total+=recall
    f1_total+=f1
    accuracy_total+=accuracy
    # spec_total+=spec


    print("------author paper citation link prediction------")
    print("AUC: " + str(auc_total/fold))
    print("precison:" + str(prei_total/fold))
    print("recall:" + str(recall_total/fold))
    print("F1: " + str(f1_total/fold))
    print("accuracy: " + str(accuracy_total / fold))
    print("spec: " + str(spec_total / fold))
    print("------author paper citation link prediction end------")





s_m_cite_train_testFC()
