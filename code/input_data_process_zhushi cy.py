import six.moves.cPickle as pickle
import numpy as np
import argparse
import string
import re
import random
import math
from collections import Counter
from itertools import *
import os

parser = argparse.ArgumentParser(description = 'input data process')
parser.add_argument('--data_path', type = str, default = '../data/academic/',
			   help='path to data')
parser.add_argument('--data_path1', type=str, default='../data/abide1/',
					help='path to data')
parser.add_argument('--model_path', type = str, default = '../model_save/',
			   help='path to save model')
parser.add_argument('--m_n', type=int, default=871,
					help='number of author node')
parser.add_argument('--l_n', type=int, default=65,
					help='number of author node') # 节点特征最大是64
parser.add_argument('--s_n', type=int, default=871,
					help='number of author node')
parser.add_argument('--ag11_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--ag14_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--ag16_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--ag20_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--ag58_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--site1_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--site2_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--site3_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--site4_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--sex1_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--sex2_n', type=int, default=1,
					help='number of author node')
parser.add_argument('--A_n', type = int, default = 28646,
			   help = 'number of author node')
parser.add_argument('--P_n', type = int, default = 21044,
			   help = 'number of paper node')
parser.add_argument('--V_n', type = int, default = 18,
			   help = 'number of venue node')
parser.add_argument('--in_f_d', type = int, default = 128,
			   help = 'input feature dimension')
parser.add_argument('--embed_d', type = int, default = 128,
			   help = 'embedding dimension')
parser.add_argument('--lr', type = int, default = 0.001,
			   help = 'learning rate')
parser.add_argument('--batch_s', type = int, default = 20000,
			   help = 'batch size')
parser.add_argument('--mini_batch_s', type = int, default = 200,
			   help = 'mini batch size')
parser.add_argument('--train_iter_n', type = int, default = 50,
			   help = 'max number of training iteration')
parser.add_argument('--walk_n', type = int, default = 10,
			   help='number of walk per root node')
parser.add_argument('--walk_L', type = int, default = 30,
			   help='length of each walk')
parser.add_argument('--window', type = int, default = 5,
			   help='window size for relation extration')
parser.add_argument("--random_seed", default = 10, type = int)
parser.add_argument('--train_test_label', type= int, default = 0,
			   help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
parser.add_argument('--save_model_freq', type = float, default = 2,
			   help = 'number of iterations to save model')
parser.add_argument("--cuda", default = 0, type = int)
parser.add_argument("--checkpoint", default = '', type=str)
args = parser.parse_args()
print(args)


class input_data(object):
	def __init__(self, args):
		self.args = args
		self.root_folder = 'E:///data//abide1'
		self.data_folder = os.path.join(self.root_folder, 'filt_noglobal/filt_noglobal')
		self.phenotype = os.path.join(self.root_folder, 'Phenotypic_V1_0b_preprocessed1.csv')
		a_p_list_train = [[] for k in range(self.args.A_n)]
		p_a_list_train = [[] for k in range(self.args.P_n)]
		p_p_cite_list_train = [[] for k in range(self.args.P_n)]
		v_p_list_train = [[] for k in range(self.args.V_n)]

		s_ag11_list_train = [[] for k in range(self.args.m_n)]
		s_ag14_list_train = [[] for k in range(self.args.m_n)]
		s_ag16_list_train = [[] for k in range(self.args.m_n)]
		s_ag20_list_train = [[] for k in range(self.args.m_n)]
		s_ag58_list_train = [[] for k in range(self.args.m_n)]
		s_site1_list_train = [[] for k in range(self.args.m_n)]
		s_site2_list_train = [[] for k in range(self.args.m_n)]
		s_site3_list_train = [[] for k in range(self.args.m_n)]
		s_site4_list_train = [[] for k in range(self.args.m_n)]
		s_sex1_list_train = [[] for k in range(self.args.m_n)]
		s_sex2_list_train = [[] for k in range(self.args.m_n)]
		ag11_s_list_train = [[] for k in range(self.args.ag11_n)]
		ag14_s_list_train = [[] for k in range(self.args.ag14_n)]
		ag16_s_list_train = [[] for k in range(self.args.ag16_n)]
		ag20_s_list_train = [[] for k in range(self.args.ag20_n)]
		ag58_s_list_train = [[] for k in range(self.args.ag58_n)]
		site1_s_list_train = [[] for k in range(self.args.site1_n)]
		site2_s_list_train = [[] for k in range(self.args.site2_n)]
		site3_s_list_train = [[] for k in range(self.args.site3_n)]
		site4_s_list_train = [[] for k in range(self.args.site4_n)]
		sex1_s_list_train = [[] for k in range(self.args.sex1_n)]
		sex2_s_list_train = [[] for k in range(self.args.sex2_n)]

		relation_f = ["ag11-s.csv", "ag14-s.csv", "ag16-s.csv", "ag20-s.csv", "ag58-s.csv", "site1-s.csv",
					  "site2-s.csv", "site3-s.csv", "site4-s.csv", "sex1-s.csv", "sex2-s.csv",
					  "s-ag11.txt", "s-ag14.txt", "s-ag16.txt", "s-ag20.txt", "s-ag58.txt", "s-site1.txt",
					  "s-site2.txt", "s-site3.txt", "s-site4.txt", "s-sex1.txt", "s-sex2.txt"]

		# store academic relational data
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path1 + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])  # 节点本身
				# print('node_id', node_id)
				neigh_list = re.split(':', line)[1]  # 节点邻居
				neigh_list_id = re.split(',', neigh_list)  # 节点邻居分开
				# print('hr:',neigh_list_id)
				if f_name == 'ag11-s.csv':
					for j in range(len(neigh_list_id)):
						ag11_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'ag14-s.csv':
					for j in range(len(neigh_list_id)):
						ag14_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'ag16-s.csv':
					for j in range(len(neigh_list_id)):
						ag16_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'ag20-s.csv':
					for j in range(len(neigh_list_id)):
						ag20_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'ag58-s.csv':
					for j in range(len(neigh_list_id)):
						ag58_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'site1-s.csv':
					for j in range(len(neigh_list_id)):
						site1_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'site2-s.csv':
					for j in range(len(neigh_list_id)):
						site2_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'site3-s.csv':
					for j in range(len(neigh_list_id)):
						site3_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'site4-s.csv':
					for j in range(len(neigh_list_id)):
						site4_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'sex1-s.csv':
					for j in range(len(neigh_list_id)):
						sex1_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 'sex2-s.csv':
					for j in range(len(neigh_list_id)):
						sex2_s_list_train[0].append('s' + str(neigh_list_id[j]))
				elif f_name == 's-ag11.txt':
					for j in range(len(neigh_list_id)):
						s_ag11_list_train[node_id].append('a' + str(neigh_list_id[j]))
				elif f_name == 's-ag14.txt':
					for j in range(len(neigh_list_id)):
						s_ag14_list_train[node_id].append('b' + str(neigh_list_id[j]))
				elif f_name == 's-ag16.txt':
					for j in range(len(neigh_list_id)):
						s_ag16_list_train[node_id].append('c' + str(neigh_list_id[j]))
				elif f_name == 's-ag20.txt':
					for j in range(len(neigh_list_id)):
						s_ag20_list_train[node_id].append('d' + str(neigh_list_id[j]))
				elif f_name == 's-ag58.txt':
					for j in range(len(neigh_list_id)):
						s_ag58_list_train[node_id].append('e' + str(neigh_list_id[j]))
				elif f_name == 's-site1.txt':
					for j in range(len(neigh_list_id)):
						s_site1_list_train[node_id].append('f' + str(neigh_list_id[j]))
				elif f_name == 's-site2.txt':
					for j in range(len(neigh_list_id)):
						s_site2_list_train[node_id].append('g' + str(neigh_list_id[j]))
				elif f_name == 's-site3.txt':
					for j in range(len(neigh_list_id)):
						s_site3_list_train[node_id].append('h' + str(neigh_list_id[j]))
				elif f_name == 's-site4.txt':
					for j in range(len(neigh_list_id)):
						s_site4_list_train[node_id].append('i' + str(neigh_list_id[j]))
				elif f_name == 's-sex1.txt':
					for j in range(len(neigh_list_id)):
						s_sex1_list_train[node_id].append('j' + str(neigh_list_id[j]))
				else:
					for j in range(len(neigh_list_id)):
						s_sex2_list_train[node_id].append('k' + str(neigh_list_id[j]))

			neigh_f.close()
		#print (p_a_list_train[0])
		#
		# #store paper venue
		# p_v = [0] * self.args.P_n
		# p_v_f = open(self.args.data_path + 'p_v.txt', "r")
		# for line in p_v_f:
		# 	line = line.strip()
		# 	p_id = int(re.split(',',line)[0])
		# 	v_id = int(re.split(',',line)[1])
		# 	p_v[p_id] = v_id
		# p_v_f.close()
		#
		# #paper neighbor: author + citation + venue
		# p_neigh_list_train = [[] for k in range(self.args.P_n)]
		# for i in range(self.args.P_n):
		# 	p_neigh_list_train[i] += p_a_list_train[i]
		# 	p_neigh_list_train[i] += p_p_cite_list_train[i]
		# 	p_neigh_list_train[i].append('v' + str(p_v[i]))
		# #print p_neigh_list_train[11846]

			# 得到邻居合集
		s_neigh_list_train = [[] for k in range(self.args.m_n)]
		for i in range(self.args.m_n):
			s_neigh_list_train[i] += s_ag11_list_train[i]
			s_neigh_list_train[i] += s_ag14_list_train[i]
			s_neigh_list_train[i] += s_ag16_list_train[i]
			s_neigh_list_train[i] += s_ag20_list_train[i]
			s_neigh_list_train[i] += s_ag58_list_train[i]
			s_neigh_list_train[i] += s_site1_list_train[i]
			s_neigh_list_train[i] += s_site2_list_train[i]
			s_neigh_list_train[i] += s_site3_list_train[i]
			s_neigh_list_train[i] += s_site4_list_train[i]
			s_neigh_list_train[i] += s_sex1_list_train[i]
			s_neigh_list_train[i] += s_sex2_list_train[i]

		self.s_ag11_list_train = s_ag11_list_train
		self.s_ag14_list_train = s_ag14_list_train
		self.s_ag16_list_train = s_ag16_list_train
		self.s_ag20_list_train = s_ag20_list_train
		self.s_ag58_list_train = s_ag58_list_train
		self.s_site1_list_train = s_site1_list_train
		self.s_site2_list_train = s_site2_list_train
		self.s_site3_list_train = s_site3_list_train
		self.s_site4_list_train = s_site4_list_train
		self.s_sex1_list_train = s_sex1_list_train
		self.s_sex2_list_train = s_sex2_list_train
		self.ag11_s_list_train = ag11_s_list_train
		self.ag14_s_list_train = ag14_s_list_train
		self.ag16_s_list_train = ag16_s_list_train
		self.ag20_s_list_train = ag20_s_list_train
		self.ag58_s_list_train = ag58_s_list_train
		self.site1_s_list_train = site1_s_list_train
		self.site2_s_list_train = site2_s_list_train
		self.site3_s_list_train = site3_s_list_train
		self.site4_s_list_train = site4_s_list_train
		self.sex1_s_list_train = sex1_s_list_train
		self.sex2_s_list_train = sex2_s_list_train
		self.s_neigh_list_train = s_neigh_list_train


	def gen_het_rand_walk(self):
		het_walk_f = open(self.args.data_path1 + "het_random_walk_train.txt", "w")
		#print len(self.p_neigh_list_train)
		for i in range(self.args.walk_n):
			for j in range(self.args.m_n):
				if len(self.s_neigh_list_train[j]):
					curNode = "s" + str(j)
					het_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == "s":
							curNode = int(curNode[1:])
							curNode = random.choice(self.s_neigh_list_train[curNode])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "a":
							curNode = int(curNode[1:])
							curNode = random.choice(self.ag11_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "b":
							curNode = int(curNode[1:])
							curNode = random.choice(self.ag14_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "c":
							curNode = int(curNode[1:])
							curNode = random.choice(self.ag16_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "d":
							# print('co:', curNode)
							curNode = int(curNode[1:])
							# print('co:',curNode)
							curNode = random.choice(self.ag20_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "e":
							curNode = int(curNode[1:])
							curNode = random.choice(self.ag58_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "f":
							curNode = int(curNode[1:])
							curNode = random.choice(self.site1_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "g":
							curNode = int(curNode[1:])
							curNode = random.choice(self.site2_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "h":
							curNode = int(curNode[1:])
							curNode = random.choice(self.site3_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "i":
							curNode = int(curNode[1:])
							curNode = random.choice(self.site4_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "j":
							curNode = int(curNode[1:])
							curNode = random.choice(self.sex1_s_list_train[0])
							het_walk_f.write(curNode + " ")
						elif curNode[0] == "k":
							curNode = int(curNode[1:])
							curNode = random.choice(self.sex2_s_list_train[0])
							het_walk_f.write(curNode + " ")
					het_walk_f.write("\n")
		het_walk_f.close()


	def gen_meta_rand_walk_APVPA(self):
		meta_walk_f = open(self.args.data_path + "meta_random_walk_APVPA_test.txt", "w")
		#print len(self.p_neigh_list_train)
		for i in range(self.args.walk_n):
			for j in range(self.args.A_n):
				if len(self.a_p_list_train[j]):
					curNode = "a" + str(j)
					preNode = "a" + str(j)
					meta_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == "a":
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.a_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
						elif curNode[0] == "p":
							curNode = int(curNode[1:])
							if preNode[0] == "a":
								preNode = "p" + str(curNode)
								curNode = "p" + str(self.p_v[curNode])
								meta_walk_f.write(curNode + " ")
							else:
								preNode = "p" + str(curNode)
								curNode = random.choice(self.p_neigh_list_train[curNode])
								meta_walk_f.write(curNode + " ")
						elif curNode[0] == "v": 
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.v_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
					meta_walk_f.write("\n")
		meta_walk_f.close()



	def a_a_collaborate_train_test(self):
		a_a_list_train = [[] for k in range(self.args.A_n)]
		a_a_list_test = [[] for k in range(self.args.A_n)]
		p_a_list = [self.p_a_list_train, self.p_a_list_test]
		
		for t in range(len(p_a_list)):
			for i in range(len(p_a_list[t])):
				for j in range(len(p_a_list[t][i])):
					for k in range(j+1, len(p_a_list[t][i])):
						if t == 0:
							a_a_list_train[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
							a_a_list_train[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
						else:#remove duplication in test and only consider existing authors
							if len(a_a_list_train[int(p_a_list[t][i][j][1:])]) and len(a_a_list_train[int(p_a_list[t][i][k][1:])]):#transductive case
								if int(p_a_list[t][i][k][1:]) not in a_a_list_train[int(p_a_list[t][i][j][1:])]:
									a_a_list_test[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
								if int(p_a_list[t][i][j][1:]) not in a_a_list_train[int(p_a_list[t][i][k][1:])]:
									a_a_list_test[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
		
		#print (a_a_list_train[1])

		for i in range(self.args.A_n):
			a_a_list_train[i]=list(set(a_a_list_train[i]))
			a_a_list_test[i]=list(set(a_a_list_test[i]))

		a_a_list_train_f = open(args.data_path + "a_a_list_train.txt", "w")
		a_a_list_test_f = open(args.data_path + "a_a_list_test.txt", "w")
		a_a_list = [a_a_list_train, a_a_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_a_list)):
			for i in range(len(a_a_list[t])):
				#print (i)
				if len(a_a_list[t][i]):
					if t == 0:
						for j in range(len(a_a_list[t][i])):
							a_a_list_train_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i]: 
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
							train_num += 2
					else:
						for j in range(len(a_a_list[t][i])):
							a_a_list_test_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i] or node_n in a_a_list_train[i] or len(a_a_list_train[i]) == 0:
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
							test_num += 2
		a_a_list_train_f.close()
		a_a_list_test_f.close()

		print("a_a_train_num: " + str(train_num))
		print("a_a_test_num: " + str(test_num))


	def a_p_citation_train_test(self):
		p_time = [0] * args.P_n
		p_time_f = open(args.data_path + "p_time.txt", "r")
		for line in p_time_f:
			line = line.strip()
			p_id = int(re.split('\t',line)[0])
			time = int(re.split('\t',line)[1])
			p_time[p_id] = time + 2005
		p_time_f.close()

		a_p_cite_list_train = [[] for k in range(self.args.A_n)]
		a_p_cite_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		p_p_cite_list_train = self.p_p_cite_list_train
		p_p_cite_list_test = self.p_p_cite_list_test
		
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					if t == 0:
						p_id = int(a_p_list[t][i][j][1:])
						for k in range(len(p_p_cite_list_train[p_id])):
							a_p_cite_list_train[i].append(int(p_p_cite_list_train[p_id][k][1:]))
					else:#remove duplication in test and only consider existing papers
						if len(self.a_p_list_train[i]):#tranductive inference
							p_id = int(a_p_list[t][i][j][1:])
							for k in range(len(p_p_cite_list_test[p_id])):
								cite_index = int(p_p_cite_list_test[p_id][k][1:])
								if p_time[cite_index] < args.T_split and (cite_index not in a_p_cite_list_train[i]):
									a_p_cite_list_test[i].append(cite_index)


		for i in range(self.args.A_n):
			a_p_cite_list_train[i] = list(set(a_p_cite_list_train[i]))
			a_p_cite_list_test[i] = list(set(a_p_cite_list_test[i]))

		test_count = 0 
		#print (a_p_cite_list_test[56])
		a_p_cite_list_train_f = open(args.data_path + "a_p_cite_list_train.txt", "w")
		a_p_cite_list_test_f = open(args.data_path + "a_p_cite_list_test.txt", "w")
		a_p_cite_list = [a_p_cite_list_train, a_p_cite_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_p_cite_list)):
			for i in range(len(a_p_cite_list[t])):
				#print (i)
				#if len(a_p_cite_list[t][i]):
				if t == 0:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]: 
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
						train_num += 2
				else:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]:
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
						test_num += 2
		a_p_cite_list_train_f.close()
		a_p_cite_list_test_f.close()

		print("a_p_cite_train_num: " + str(train_num))
		print("a_p_cite_test_num: " + str(test_num))


	def a_v_train_test(self):
		a_v_list_train = [[] for k in range(self.args.A_n)]
		a_v_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					p_id = int(a_p_list[t][i][j][1:])
					if t == 0:
						a_v_list_train[i].append(self.p_v[p_id])
					else:
						if self.p_v[p_id] not in a_v_list_train[i] and len(a_v_list_train[i]):
							a_v_list_test[i].append(self.p_v[p_id])

		for k in range(self.args.A_n):
			a_v_list_train[k] = list(set(a_v_list_train[k]))
			a_v_list_test[k] = list(set(a_v_list_test[k]))

		a_v_list_train_f = open(args.data_path + "a_v_list_train.txt", "w")
		a_v_list_test_f = open(args.data_path + "a_v_list_test.txt", "w")
		a_v_list = [a_v_list_train, a_v_list_test]
		# train_num = 0
		# test_num = 0
		# test_a_num = 0
		for t in range(len(a_v_list)):
			for i in range(len(a_v_list[t])):
				if t == 0:
					if len(a_v_list[t][i]):
						a_v_list_train_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_train_f.write(str(a_v_list[t][i][j])+",")
							#train_num += 1
						a_v_list_train_f.write("\n")
				else:
					if len(a_v_list[t][i]):
						#test_a_num += 1
						a_v_list_test_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_test_f.write(str(a_v_list[t][i][j])+",")
							#test_num += 1
						a_v_list_test_f.write("\n")
		a_v_list_train_f.close()
		a_v_list_test_f.close()

		# print("a_v_train_num: " + str(train_num))
		# print("a_v_test_num: " + str(test_num))
		# print (float(test_num) / test_a_num)



input_data_class = input_data(args = args)


input_data_class.gen_het_rand_walk()


#input_data_class.gen_meta_rand_walk_APVPA()


#input_data_class.a_a_collaborate_train_test() #set author-author collaboration data 


#input_data_class.a_p_citation_train_test() #set author-paper citation data 


#input_data_class.a_v_train_test() #generate author-venue data 



