import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse
import ds
import numpy as np
import torch
import dd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import roc_curve,auc

parser = argparse.ArgumentParser(description = 'link prediction task')
parser.add_argument('--A_n', type = int, default = 28646,
			   help = 'number of author node')
parser.add_argument('--P_n', type = int, default = 21044,
			   help = 'number of paper node')
parser.add_argument('--V_n', type = int, default = 18,
			   help = 'number of venue node')
parser.add_argument('--data_path', type = str, default = 'D:/codee/data/abide1/',
				   help='path to data')
parser.add_argument('--embed_d', type = int, default = 1950,
			   help = 'embedding dimension')

args = parser.parse_args()
print(args)


def load_data(data_file_name, n_features, n_samples):
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        print('data:', data.shape)
        for i, d in enumerate(data_file):
            # print('1:',i)
            # print('d:',d)
            data[i] = numpy.asarray(d[:], dtype=numpy.float)
        f.close
        # print('data:',data)
        return data


def model(train_num, test_num):
	train_data_f = args.data_path + "train_feature.txt"
	train_data = load_data(train_data_f, args.embed_d + 2, train_num)
	train_features = train_data.astype(numpy.float32)[:, 2:-1]
	print('tr:', train_num)
	# input = dd.input_data
	subject_IDs = ds.get_ids()
	print('su:', subject_IDs)
	subject_IDs_ = ds.get_id1()
	labels = ds.get_subject_score(subject_IDs_, score='DX_GROUP')
	train_target = train_data.astype(numpy.float32)[:, 1]
	# print('te:', train_target)
	# print(train_target.shape)
	# y = np.zeros([len(subject_IDs_), 1])
	# for i in range(len(subject_IDs_)):
	# 	y[i] = int(labels[subject_IDs_[i]])
	# y = torch.tensor(y - 1, dtype=int)
	# y = np.squeeze(y)
	# print('y:',y)


	#print(train_target[1])
	learner = linear_model.LogisticRegression()
	learner.fit(train_features, train_target)
	train_features = None
	train_target = None

	print("training finish!")

	test_data_f = args.data_path + "test_feature.txt"
	test_data = load_data(test_data_f, args.embed_d + 2, test_num)
	test_id = test_data.astype(numpy.int32)[:,0]
	test_features = test_data.astype(numpy.float32)[:,2:-1]
	test_target = test_data.astype(numpy.float32)[:,1]
	# print('trser:', test_target)
	test_predict = learner.predict(test_features)
	print('te:', test_predict)
	test_features = None

	print("test prediction finish!")

	output_f = open(args.data_path + "NC_prediction.txt", "w")
	for i in range(len(test_predict)):
	    output_f.write('%d,%lf\n'%(test_id[i],test_predict[i]));
	output_f.close();

	AUC_score = Metric.roc_auc_score(test_target, test_predict)
	print("AUC: " + str(AUC_score))
	total_count = 0
	correct_count = 0
	true_p_count = 0
	true_n_count = 0
	false_p_count = 0
	false_n_count = 0
	true_p_count = 0
	for i in range(len(test_predict)):
		total_count += 1
		if (int(test_predict[i]) == int(test_target[i])):
			correct_count += 1
		if (int(test_predict[i]) == 1 and int(test_target[i]) == 1):
			true_p_count += 1
		if (int(test_predict[i]) == 0 and int(test_target[i]) == 0):
			true_n_count += 1
		if (int(test_predict[i]) == 1 and int(test_target[i]) == 0):
			false_p_count += 1
		if (int(test_predict[i]) == 0 and int(test_target[i]) == 1):
			false_n_count += 1
	accuracy = float(true_p_count + true_n_count) / (true_p_count + true_n_count + false_n_count + false_p_count)
	# TPR=TP/ (TP+ FN)  TPR即为敏感度（sensitivity）  也是recall
	sensitivity = TPR(test_target, test_predict)
	print('Sensitivity of Test Samples:', sensitivity)
	specificity = TNR(test_target, test_predict)
	print('specificity of Test Samples:', specificity)
	f1 = sklearn.metrics.f1_score(test_target, test_predict, average='macro')
	print("MacroF1: ", f1)
	print("accuracy: " + str(accuracy))

	# print ("MicroF1: ")
	# print (sklearn.metrics.f1_score(test_target,test_predict,average='micro'))
	# print("accuracy: " + str(accuracy))
	return AUC_score, f1, accuracy



def calculate_TP(y, y_pred):
    tp = 0
    for i, j in zip(y, y_pred):
        if i == j == 1:
            tp += 1
    return tp
def calculate_FN(y, y_pred):
    fn = 0
    for i, j in zip(y, y_pred):
        if i == 1 and j == 0:
            fn += 1
    return fn
def calculate_TN(y, y_pred):
    tn = 0
    for i, j in zip(y, y_pred):
        if i == j == 0:
            tn += 1
    return tn
def calculate_FP(y, y_pred):
    fp = 0
    for i, j in zip(y, y_pred):
        if i == 0 and j == 1:
            fp += 1
    return fp

def TPR(y, y_pred):
    tp1 = calculate_TP(y, y_pred)
    # print('tp1:', tp1)
    fn = calculate_FN(y, y_pred)
    return tp1 / (fn + tp1)

# TNR= TN / (FP + TN)   , return tp / (tp + fp), specificity
def TNR(y, y_pred):
    tn = calculate_TN(y, y_pred)
    fp = calculate_FP(y, y_pred)
    return tn / (fp + tn)
#
	# print ("MacroF1: ")
	# print (sklearn.metrics.f1_score(test_target,test_predict,average='macro'))
	#
	# print ("MicroF1: ")
	# print (sklearn.metrics.f1_score(test_target,test_predict,average='micro'))




