import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from raw_data_process import *
import scipy.io as sio
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
import csv


class input_data(object):
	def __init__(self, args):
		self.args = args
		self.root_folder = 'D:/codee/data/abide1'
		self.data_folder = os.path.join(self.root_folder, 'filt_noglobal/filt_noglobal')
		self.phenotype = os.path.join(self.root_folder, 'Phenotypic_V1_0b_preprocessed1.csv')

		with open('D:/codee/data/abide1/s-ag11.csv') as filein, open(
				'D:/codee/data/abide1/s-ag11.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag14.csv') as filein, open(
				'D:/codee/data/abide1/s-ag14.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag16.csv') as filein, open(
				'D:/codee/data/abide1/s-ag16.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag20.csv') as filein, open(
				'D:/codee/data/abide1/s-ag20.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag58.csv') as filein, open(
				'D:/codee/data/abide1/s-ag58.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site1.csv') as filein, open(
				'D:/codee/data/abide1/s-site1.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site2.csv') as filein, open(
				'D:/codee/data/abide1/s-site2.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site3.csv') as filein, open(
				'D:/codee/data/abide1/s-site3.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site4.csv') as filein, open(
				'D:/codee/data/abide1/s-site4.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-sex1.csv') as filein, open(
				'D:/codee/data/abide1/s-sex1.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-sex2.csv') as filein, open(
				'D:/codee/data/abide1/s-sex2.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag11-test.csv') as filein, open(
				'D:/codee/data/abide1/s-ag11-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag14-test.csv') as filein, open(
				'D:/codee/data/abide1/s-ag14-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag16-test.csv') as filein, open(
				'D:/codee/data/abide1/s-ag16-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag20-test.csv') as filein, open(
				'D:/codee/data/abide1/s-ag20-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-ag58-test.csv') as filein, open(
				'D:/codee/data/abide1/s-ag58-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site1-test.csv') as filein, open(
				'D:/codee/data/abide1/s-site1-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-site2-test.csv') as filein, open(
				'D:/codee/data/abide1/s-site2-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-sex1-test.csv') as filein, open(
				'D:/codee/data/abide1/s-sex1-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
		with open('D:/codee/data/abide1/s-sex2-test.csv') as filein, open(
				'D:/codee/data/abide1/s-sex2-test.txt', 'w') as fileout:
			for line in filein:
				line = line.replace(",", ":")
				fileout.write(line)
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


		relation_f = ["ag11-s.csv", "ag14-s.csv", "ag16-s.csv","ag20-s.csv", "ag58-s.csv", "site1-s.csv", "site2-s.csv", "site3-s.csv", "site4-s.csv", "sex1-s.csv", "sex2-s.csv",
		 "s-ag11.txt", "s-ag14.txt", "s-ag16.txt","s-ag20.txt", "s-ag58.txt","s-site1.txt","s-site2.txt","s-site3.txt","s-site4.txt","s-sex1.txt","s-sex2.txt"]

		#store academic relational data
		for i in range(len(relation_f)):
			# print('i:',i)
			f_name = relation_f[i]
			# print(f_name)
			neigh_f = open(self.args.data_path1 + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				# print(re.split(':', line)[0])
				node_id = int(re.split(':', line)[0])  # 节点本身
				# print('node_id', node_id)
				neigh_list = re.split(':', line)[1] # 节点邻居
				neigh_list_id = re.split(',', neigh_list) #节点邻居分开
				# print('hr:',neigh_list_id)
				if f_name == 'ag11-s.csv':
					for j in range(len(neigh_list_id)):
						ag11_s_list_train[0].append('s'+str(neigh_list_id[j]))
				elif f_name == 'ag14-s.csv':
					for j in range(len(neigh_list_id)):
						ag14_s_list_train[0].append('s'+str(neigh_list_id[j]))
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

		#
		# s_ag11 = [0] * self.args.s_n
		# s_ag14 = [0] * self.args.s_n
		# s_ag16 = [0] * self.args.s_n
		# s_ag20 = [0] * self.args.s_n
		# s_ag58 = [0] * self.args.s_n
		# s_site1 = [0] * self.args.s_n
		# s_site2 = [0] * self.args.s_n
		# s_site3 = [0] * self.args.s_n
		# s_site4 = [0] * self.args.s_n
		# s_sex1 = [0] * self.args.s_n
		# s_sex2 = [0] * self.args.s_n
		#
		# s_ag11_f = open(self.args.data_path + 's-ag11.csv', "r")
		# for line in s_ag11_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	ag11_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = ag11_id
		# # print('e"',p_v)
		# s_ag11_f.close()
		#
		# s_ag14_f = open(self.args.data_path + 's-ag14.csv', "r")
		# for line in s_ag14_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	ag14_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = ag14_id
		# # print('e"',p_v)
		# s_ag14_f.close()
		#
		# s_ag16_f = open(self.args.data_path + 's-ag16.csv', "r")
		# for line in s_ag16_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	ag16_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = ag16_id
		# # print('e"',p_v)
		# s_ag16_f.close()
		#
		# s_ag20_f = open(self.args.data_path + 's-ag20.csv', "r")
		# for line in s_ag20_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	ag20_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = ag20_id
		# # print('e"',p_v)
		# s_ag20_f.close()
		#
		# s_ag58_f = open(self.args.data_path + 's-ag58.csv', "r")
		# for line in s_ag58_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	ag58_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = ag58_id
		# # print('e"',p_v)
		# s_ag58_f.close()
		#
		# s_site1_f = open(self.args.data_path + 's-site1.csv', "r")
		# for line in s_site1_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	site1_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = site1_id
		# # print('e"',p_v)
		# s_site1_f.close()
		#
		# s_site2_f = open(self.args.data_path + 's-site2.csv', "r")
		# for line in s_site2_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	site2_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = site2_id
		# # print('e"',p_v)
		# s_site2_f.close()
		#
		# s_site3_f = open(self.args.data_path + 's-site3.csv', "r")
		# for line in s_site3_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	site3_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = site3_id
		# # print('e"',p_v)
		# s_site3_f.close()
		#
		# s_site4_f = open(self.args.data_path + 's-site4.csv', "r")
		# for line in s_site4_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	site4_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = site4_id
		# # print('e"',p_v)
		# s_site4_f.close()
		#
		# s_sex1_f = open(self.args.data_path + 's-sex1.csv', "r")
		# for line in s_sex1_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	sex1_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = sex1_id
		# # print('e"',p_v)
		# s_sex1_f.close()
		#
		# s_sex2_f = open(self.args.data_path + 's-sex2.csv', "r")
		# for line in s_sex2_f:
		# 	line = line.strip()
		# 	s_id = int(re.split(',', line)[0])  # 论文的节点
		# 	sex2_id = int(re.split(',', line)[1])  # 论文的地点
		# 	p_v[s_id] = sex2_id
		# # print('e"',p_v)
		# s_sex2_f.close()

		# #store paper venue
		p_v = [0] * self.args.P_n  # 定义全为0的数组

		# 每篇论文的地点
		p_v_f = open(self.args.data_path + 'p_v.txt', "r")
		for line in p_v_f:
			line = line.strip()
			p_id = int(re.split(',', line)[0])  # 论文的节点
			v_id = int(re.split(',', line)[1])   # 论文的地点
			p_v[p_id] = v_id
		# print('e"',p_v)
		p_v_f.close()


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
			# s_neigh_list_train[i].append('v' + str(p_v[i]))
		# print('s:',s_neigh_list_train)
		ag11_neigh_list_train = []
		for i in range(self.args.ag11_n):
			ag11_neigh_list_train.append(ag11_s_list_train[i])

		ag14_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.ag14_n):
			ag14_neigh_list_train.append(ag14_s_list_train[i])

		ag16_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.ag16_n):
			ag16_neigh_list_train.append(ag16_s_list_train[i])

		ag20_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.ag20_n):
			ag20_neigh_list_train.append(ag20_s_list_train[i])

		ag58_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.ag58_n):
			ag58_neigh_list_train.append(ag58_s_list_train[i])

		site1_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.site1_n):
			site1_neigh_list_train.append(site1_s_list_train[i])

		site2_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.site2_n):
			site2_neigh_list_train.append(site2_s_list_train[i])

		site3_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.site3_n):
			site3_neigh_list_train.append(site3_s_list_train[i])

		site4_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.site4_n):
			site4_neigh_list_train.append(site4_s_list_train[i])

		sex1_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.sex1_n):
			sex1_neigh_list_train.append(sex1_s_list_train[i])

		sex2_neigh_list_train = [[] for k in range(self.args.l_n)]
		for i in range(self.args.sex2_n):
			sex2_neigh_list_train.append(sex2_s_list_train[i])



		#paper neighbor: author + citation + venue  作者+引用+地点
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		for i in range(self.args.P_n):
			p_neigh_list_train[i] += p_a_list_train[i]
			p_neigh_list_train[i] += p_p_cite_list_train[i]
			p_neigh_list_train[i].append('v' + str(p_v[i]))
		# print('de:',p_neigh_list_train)
		#print p_neigh_list_train[11846]

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




		self.a_p_list_train =  a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.v_p_list_train = v_p_list_train



		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()


			# store neighbor set from random walk sequence
			# 存储随机游走序列的邻居集合

			a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(3)]
			p_neigh_list_train = [[[] for i in range(self.args.P_n)] for j in range(3)]
			v_neigh_list_train = [[[] for i in range(self.args.V_n)] for j in range(3)]
			ag11_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			ag14_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			ag16_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			ag20_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			ag58_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			site1_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			site2_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			site3_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			site4_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			sex1_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			sex2_neigh_list_train1 = [[[] for i in range(1)] for j in range(12)]
			s_neigh_list_train1 = [[[] for i in range(self.args.m_n)] for j in range(12)]
			# print('sd:',ag58_neigh_list_train1)

			for i in range(len(s_neigh_list_train)):
				if len(s_neigh_list_train[i])!=0:
					neigh_list1 = s_neigh_list_train[i]
					for j in range(len(neigh_list1)):
						if neigh_list1[j][0] == 's':
							s_neigh_list_train1[0][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'a':
							s_neigh_list_train1[1][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'b':
							s_neigh_list_train1[2][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'c':
							s_neigh_list_train1[3][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'd':
							s_neigh_list_train1[4][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'e':
							s_neigh_list_train1[5][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'f':
							s_neigh_list_train1[6][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'g':
							s_neigh_list_train1[7][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'h':
							s_neigh_list_train1[8][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'i':
							s_neigh_list_train1[9][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'j':
							s_neigh_list_train1[10][i].append(int(neigh_list1[j][1:]))
						elif neigh_list1[j][0] == 'k':
							s_neigh_list_train1[11][i].append(int(neigh_list1[j][1:]))

			print('ff:',s_neigh_list_train1[1])

			for i in range(len(ag11_neigh_list_train)):
				if len(ag11_neigh_list_train[i])!=0:
					for j in range(len(ag11_neigh_list_train[i])):
						n1=ag11_neigh_list_train[i][j]
						# print('n1:',n1)
						ag11_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff:',ag11_neigh_list_train1)

			for i in range(len(ag14_neigh_list_train)):
				if len(ag14_neigh_list_train[i])!=0:
					for j in range(len(ag14_neigh_list_train[i])):
						n1=ag14_neigh_list_train[i][j]
						# print('n1:',n1)
						ag14_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff:',ag14_neigh_list_train1)

			for i in range(len(ag16_neigh_list_train)):
				if len(ag16_neigh_list_train[i])!=0:
					for j in range(len(ag16_neigh_list_train[i])):
						n1=ag16_neigh_list_train[i][j]
						# print('n1:',n1)
						ag16_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff:',ag16_neigh_list_train1)

			for i in range(len(ag20_neigh_list_train)):
				if len(ag20_neigh_list_train[i]) != 0:
					for j in range(len(ag20_neigh_list_train[i])):
						n1 = ag20_neigh_list_train[i][j]
						# print('n1:',n1)
						ag20_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff:', ag20_neigh_list_train1)

			for i in range(len(ag58_neigh_list_train)):
				if len(ag58_neigh_list_train[i]) != 0:
					for j in range(len(ag58_neigh_list_train[i])):
						n1 = ag58_neigh_list_train[i][j]
						# print('n1:',n1)
						ag58_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff:', ag58_neigh_list_train1)

			for i in range(len(site1_neigh_list_train)):
				if len(site1_neigh_list_train[i]) != 0:
					for j in range(len(site1_neigh_list_train[i])):
						n1 = site1_neigh_list_train[i][j]
						# print('n1:',n1)
						site1_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff6:', site1_neigh_list_train1)

			for i in range(len(site2_neigh_list_train)):
				if len(site2_neigh_list_train[i]) != 0:
					for j in range(len(site2_neigh_list_train[i])):
						n1 = site2_neigh_list_train[i][j]
						# print('n1:',n1)
						site2_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff5:', site2_neigh_list_train1)

			for i in range(len(site3_neigh_list_train)):
				if len(site3_neigh_list_train[i]) != 0:
					for j in range(len(site3_neigh_list_train[i])):
						n1 = site3_neigh_list_train[i][j]
						# print('n1:',n1)
						site3_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff4:', site3_neigh_list_train1)

			for i in range(len(site4_neigh_list_train)):
				if len(site4_neigh_list_train[i]) != 0:
					for j in range(len(site4_neigh_list_train[i])):
						n1 = site4_neigh_list_train[i][j]
						# print('n1:',n1)
						site4_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff3:', site4_neigh_list_train1)

			for i in range(len(sex1_neigh_list_train)):
				if len(sex1_neigh_list_train[i]) != 0:
					for j in range(len(sex1_neigh_list_train[i])):
						n1 = sex1_neigh_list_train[i][j]
						# print('n1:',n1)
						sex1_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff2:', sex1_neigh_list_train1)

			for i in range(len(sex2_neigh_list_train)):
				if len(sex2_neigh_list_train[i]) != 0:
					for j in range(len(sex2_neigh_list_train[i])):
						n1 = sex2_neigh_list_train[i][j]
						# print('n1:',n1)
						sex2_neigh_list_train1[0][0].append(int(n1[1:]))
			# print('ff1:', sex2_neigh_list_train1)


			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)
				if node_id[0] == 'a' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))  # 是a邻居中a开头的
						elif neigh_list[j][0] == 'p':
							a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))  # [11846, 11765, 11765, 11846,  是a邻居中p开头的
						elif neigh_list[j][0] == 'v':
							a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))  # 是a邻居中v开头的

				elif node_id[0] == 'p' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'p':
							p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'v':
							p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'v' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'p':
							v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'v':
							v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
			# print('dd',a_neigh_list_train)

			het_neigh_train_f.close()
			# print(a_neigh_list_train[0][1])

			#store top neighbor set (based on frequency) from random walk sequence
			# 存储随机游走序列的顶部邻居集(基于频率)
			a_neigh_list_train_top = [[[] for i in range(self.args.A_n)] for j in range(3)]  # 储存出现频率最多的a p v 的下标
			p_neigh_list_train_top = [[[] for i in range(self.args.P_n)] for j in range(3)]
			v_neigh_list_train_top = [[[] for i in range(self.args.V_n)] for j in range(3)]

			ag11_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			ag14_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			ag16_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			ag20_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			ag58_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			site1_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			site2_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			site3_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			site4_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			sex1_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			sex2_neigh_list_train1_top = [[[] for i in range(1)] for j in range(12)]
			s_neigh_list_train1_top = [[[] for i in range(self.args.m_n)] for j in range(12)]
			# print('ag:',ag11_neigh_list_train1_top)


			top_k = [10, 10, 10,10,10,10,10,10,10,10,10,10,10] #  fix each neighor type size  固定每个邻接类型的大小

			for i in range(self.args.m_n):
				for j in range(12):
					s_neigh_list_train_temp = Counter(s_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
					# print('r:',ag11_neigh_list_train_temp)
					# print('k:',top_k[j])
					top_list = s_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
					# print('top:',top_list)
					# print('top:',top_list)
					neigh_size = 0

					neigh_size = 10
					# else:
					# 	neigh_size = 3
					for k in range(len(top_list)):
						s_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						# print('ag11:',ag11_neigh_list_train1_top)
					if len(s_neigh_list_train1_top[j][i]) and len(s_neigh_list_train1_top[j][i]) < neigh_size:
						# print(' :', ag11_neigh_list_train1_top[j][i])
						for l in range(len(s_neigh_list_train1_top[j][i]),neigh_size):
							s_neigh_list_train1_top[j][i].append(random.choice(s_neigh_list_train1_top[j][i]))

			for i in range(self.args.ag11_n):
				for j in range(12):
					ag11_neigh_list_train_temp = Counter(ag11_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
					# print('r:',ag11_neigh_list_train_temp)
					# print('k:',top_k[j])
					top_list = ag11_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
					# print('top:',top_list)
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						ag11_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						# print('ag11:',ag11_neigh_list_train1_top)
					if len(ag11_neigh_list_train1_top[j][i]) and len(ag11_neigh_list_train1_top[j][i]) < neigh_size:
						for l in range(len(ag11_neigh_list_train1_top[j][i]), neigh_size):
							ag11_neigh_list_train1_top[j][i].append(random.choice(ag11_neigh_list_train1_top[j][i]))

			for i in range(self.args.ag14_n):
				for j in range(12):
					ag14_neigh_list_train_temp = Counter(
						ag14_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
					top_list = ag14_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						ag14_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
					# print('ag11:',ag11_neigh_list_train1_top)
					if len(ag14_neigh_list_train1_top[j][i]) and len(ag14_neigh_list_train1_top[j][i]) < neigh_size:
						for l in range(len(ag14_neigh_list_train1_top[j][i]), neigh_size):
							ag14_neigh_list_train1_top[j][i].append(random.choice(ag14_neigh_list_train1_top[j][i]))

				for i in range(self.args.ag16_n):
					for j in range(12):
						ag16_neigh_list_train_temp = Counter(
							ag16_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；

						top_list = ag16_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							ag16_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						# print('ag11:',ag11_neigh_list_train1_top)
						if len(ag16_neigh_list_train1_top[j][i]) and len(ag16_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(ag16_neigh_list_train1_top[j][i]), neigh_size):
								ag16_neigh_list_train1_top[j][i].append(random.choice(ag16_neigh_list_train1_top[j][i]))

				for i in range(self.args.ag20_n):
					for j in range(12):
						ag20_neigh_list_train_temp = Counter(
							ag20_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；

						top_list = ag20_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							ag20_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(ag20_neigh_list_train1_top[j][i]) and len(ag20_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(ag20_neigh_list_train1_top[j][i]), neigh_size):
								ag20_neigh_list_train1_top[j][i].append(random.choice(ag20_neigh_list_train1_top[j][i]))

				for i in range(self.args.ag58_n):
					for j in range(12):
						ag58_neigh_list_train_temp = Counter(
							ag58_neigh_list_train1[j][
								i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；

						top_list = ag58_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							ag58_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(ag58_neigh_list_train1_top[j][i]) and len(ag58_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(ag58_neigh_list_train1_top[j][i]), neigh_size):
								ag58_neigh_list_train1_top[j][i].append(random.choice(ag58_neigh_list_train1_top[j][i]))

				for i in range(self.args.site1_n):
					for j in range(12):
						site1_neigh_list_train_temp = Counter(
							site1_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = site1_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							site1_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(site1_neigh_list_train1_top[j][i]) and len(site1_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(site1_neigh_list_train1_top[j][i]), neigh_size):
								site1_neigh_list_train1_top[j][i].append(random.choice(site1_neigh_list_train1_top[j][i]))

				for i in range(self.args.site2_n):
					for j in range(12):
						site2_neigh_list_train_temp = Counter(
							site2_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = site2_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							site2_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(site2_neigh_list_train1_top[j][i]) and len(site2_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(site2_neigh_list_train1_top[j][i]), neigh_size):
								site2_neigh_list_train1_top[j][i].append(random.choice(site2_neigh_list_train1_top[j][i]))

				for i in range(self.args.site3_n):
					for j in range(12):
						site3_neigh_list_train_temp = Counter(
							site3_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = site3_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							site3_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(site3_neigh_list_train1_top[j][i]) and len(site3_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(site3_neigh_list_train1_top[j][i]), neigh_size):
								site3_neigh_list_train1_top[j][i].append(random.choice(site3_neigh_list_train1_top[j][i]))

				for i in range(self.args.site4_n):
					for j in range(12):
						site4_neigh_list_train_temp = Counter(
							site4_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = site4_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							site4_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(site4_neigh_list_train1_top[j][i]) and len(site4_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(site4_neigh_list_train1_top[j][i]), neigh_size):
								site4_neigh_list_train1_top[j][i].append(random.choice(site4_neigh_list_train1_top[j][i]))

				for i in range(self.args.sex1_n):
					for j in range(12):
						sex1_neigh_list_train_temp = Counter(sex1_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = sex1_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							sex1_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(sex1_neigh_list_train1_top[j][i]) and len(
								sex1_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(sex1_neigh_list_train1_top[j][i]), neigh_size):
								sex1_neigh_list_train1_top[j][i].append(random.choice(sex1_neigh_list_train1_top[j][i]))

				for i in range(self.args.sex2_n):
					for j in range(12):
						sex2_neigh_list_train_temp = Counter(sex2_neigh_list_train1[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
						top_list = sex2_neigh_list_train_temp.most_common(top_k[j])  # 统计出现的次数、频率
						neigh_size = 0
						if j == 0 or j == 1:
							neigh_size = 10
						else:
							neigh_size = 3
						for k in range(len(top_list)):
							sex2_neigh_list_train1_top[j][i].append(int(top_list[k][0]))
						if len(sex2_neigh_list_train1_top[j][i]) and len(
								sex2_neigh_list_train1_top[j][i]) < neigh_size:
							for l in range(len(sex2_neigh_list_train1_top[j][i]), neigh_size):
								sex2_neigh_list_train1_top[j][i].append(random.choice(sex2_neigh_list_train1_top[j][i]))


			for i in range(self.args.A_n):
				for j in range(3):
					a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])  # Counter()用来统计列表，元组，字符串等可迭代数组的出现次数，并返回key为迭代元素，value为出现次数的字典；
					top_list = a_neigh_list_train_temp.most_common(top_k[j])   # 统计出现的次数、频率
					# print('top:',top_list)
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
							a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))


			for i in range(self.args.P_n):
				for j in range(3):
					p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
					top_list = p_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
							p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

			for i in range(self.args.V_n):
				for j in range(3):
					v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
					top_list = v_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
							v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))

			a_neigh_list_train[:] = []
			p_neigh_list_train[:] = []
			v_neigh_list_train[:] = []

			s_neigh_list_train1[:] = []
			ag11_neigh_list_train1[:] = []
			ag14_neigh_list_train1[:] = []
			ag16_neigh_list_train1[:] = []
			ag20_neigh_list_train1[:] = []
			ag58_neigh_list_train1[:] = []
			site1_neigh_list_train1[:] = []
			site2_neigh_list_train1[:] = []
			site3_neigh_list_train1[:] = []
			site4_neigh_list_train1[:] = []
			sex1_neigh_list_train1[:] = []
			sex2_neigh_list_train1[:] = []

			# print('ee:',ag14_neigh_list_train1_top)
			self.a_neigh_list_train = a_neigh_list_train_top
			self.p_neigh_list_train = p_neigh_list_train_top
			self.v_neigh_list_train = v_neigh_list_train_top

			# print('ff:', s_neigh_list_train1_top[1])
			self.s_neigh_list_train1 = s_neigh_list_train1_top
			self.ag11_neigh_list_train1 = ag11_neigh_list_train1_top
			self.ag14_neigh_list_train1 = ag14_neigh_list_train1_top
			self.ag16_neigh_list_train1 = ag16_neigh_list_train1_top
			self.ag20_neigh_list_train1 = ag20_neigh_list_train1_top
			self.ag58_neigh_list_train1 = ag58_neigh_list_train1_top
			self.site1_neigh_list_train1 = site1_neigh_list_train1_top
			self.site2_neigh_list_train1 = site2_neigh_list_train1_top
			self.site3_neigh_list_train1 = site3_neigh_list_train1_top
			self.site4_neigh_list_train1 = site4_neigh_list_train1_top
			self.sex1_neigh_list_train1 = sex1_neigh_list_train1_top
			self.sex2_neigh_list_train1 = sex2_neigh_list_train1_top

			s_train_id_list = []
			lneigh_f = open(self.args.data_path1+"/1.txt", "r")
			for line in lneigh_f:
				line = line.strip()
				s_train_id_list.append(int(line))
			self.s_train_id_list = np.array(s_train_id_list)
			print('g:', self.s_train_id_list)
			self.ag11_train_id_list = [1]
			self.ag14_train_id_list = [1]
			self.ag16_train_id_list = [1]
			self.ag20_train_id_list = [1]
			self.ag58_train_id_list = [1]
			self.site1_train_id_list = [1]
			self.site2_train_id_list = [1]
			self.site3_train_id_list = [1]
			self.site4_train_id_list = [1]
			self.sex1_train_id_list = [1]
			self.sex2_train_id_list = [1]
			# print('e:', self.ag11_train_id_list)
			# print(len(self.ag11_train_id_list))




	def get_networks(self,subject_list, kind, atlas_name="aal", variable='connectivity'):


		all_networks = []
		for subject in subject_list:
			fl = os.path.join(self.data_folder, subject,subject + "_" + atlas_name + "_" + kind + ".mat")
			# os.path.join() 是连接两个或更多的路径名组件
			matrix = sio.loadmat(fl)[variable]
			all_networks.append(matrix)
		# all_networks=np.array(all_networks)

		idx = np.triu_indices_from(all_networks[0], 1)
		norm_networks = [np.arctanh(mat) for mat in all_networks]
		vec_networks = [mat[idx] for mat in norm_networks]
		matrix = np.vstack(vec_networks)
		return matrix

	def get_ids(self, num_subjects=None):


		subject_IDs = np.genfromtxt(os.path.join(self.data_folder, 'subject_IDs.txt'), dtype=str)
		# os.path.join(data_folder, 'subject_IDs.txt') 输出 ABIDE_pcp/cpac/filt_noglobal\subject_IDs.txt

		if num_subjects is not None:
			subject_IDs = subject_IDs[:num_subjects]

		return subject_IDs

	# 降维
	def feature_selection(self,matrix, labels, train_ind, fnum):
		

		estimator = RidgeClassifier()  # 岭回归
		selector = RFE(estimator, fnum, step=100, verbose=1)

		featureX = matrix[train_ind, :]

		featureY = labels[train_ind]
		# print(f'featureY:{featureY}')
		# print(featureY.shape)
		# print('X:', featureX.shape)
		# print('Y:', featureY.shape)
		selector = selector.fit(featureX, featureY.ravel())
		# print(f'selector:{selector}')
		x_data = selector.transform(matrix)
		# print('x_data:', x_data)

		print("Number of labeled samples %d" % len(train_ind))
		print("Number of features selected %d" % x_data.shape[1])

		return x_data

	# 获取受试者列表的表型值
	def get_subject_score(self, subject_list, score):
		scores_dict = {}

		with open(self.phenotype) as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				if row['SUB_ID'] in subject_list:
					scores_dict[row['SUB_ID']] = row[score]
		# print('scores_dict:', scores_dict)
		return scores_dict

	def site_percentage(self,train_ind, perc, subject_list):
		"""
            train_ind    : indices of the training samples
            perc         : percentage of training set used (使用训练集的百分比)
            subject_list : list of subject IDs

        return:
            labeled_indices      : indices of the subset of training samples(训练样本子集的指标)
        """

		train_list = subject_list[train_ind]
		sites = input_data.get_subject_score(self, train_list, score='SITE_ID')  # 寻找在所有站点中的同时也在训练集中站点
		# print(f'sites:{sites}') 输出：sites:{'50145': 'OHSU', '50146': 'OHSU', '50147': 'OHSU',。。}
		unique = np.unique(list(sites.values())).tolist()  # np.unique函数排除重复元素之后，升序排列 toist是将转换为列表形式
		site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])
		# index可以查询到unique中某个字符（sites[train_list[x]]）的位置下标，np.array函数是将括号里的内容转换为数组
		labeled_indices = []

		for i in np.unique(site):
			id_in_site = np.argwhere(site == i).flatten()  # np.argwhere函数返回site==i的数组下标 flatten()函数可以执行展平操作，返回一个一维数组

			num_nodes = len(id_in_site)
			labeled_num = int(round(perc * num_nodes))  # round函数返回四舍五入的值 例如round（56.4567,2）返回56.45保留小数点后两位
			labeled_indices.extend(train_ind[id_in_site[:labeled_num]])
		# print(f'labeled_indices:{labeled_indices}')
		print(len(labeled_indices))
		return labeled_indices

	def het_walk_restart(self):
		s_neigh_list_train = [[] for k in range(self.args.s_n)]
		ag11_neigh_list_train = [[] for k in range(self.args.ag11_n)]
		ag14_neigh_list_train = [[] for k in range(self.args.ag14_n)]
		ag16_neigh_list_train = [[] for k in range(self.args.ag16_n)]
		ag20_neigh_list_train = [[] for k in range(self.args.ag20_n)]
		ag58_neigh_list_train = [[] for k in range(self.args.ag58_n)]
		site1_neigh_list_train = [[] for k in range(self.args.site1_n)]
		site2_neigh_list_train = [[] for k in range(self.args.site2_n)]
		site3_neigh_list_train = [[] for k in range(self.args.site3_n)]
		site4_neigh_list_train = [[] for k in range(self.args.site4_n)]
		sex1_neigh_list_train = [[] for k in range(self.args.sex1_n)]
		sex2_neigh_list_train = [[] for k in range(self.args.sex2_n)]


		a_neigh_list_train = [[] for k in range(self.args.A_n)]
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		v_neigh_list_train = [[] for k in range(self.args.V_n)]

		# generate neighbor set via random walk with restart
		# 通过重启随机游走生成邻居集合
		node_n = [self.args.A_n, self.args.P_n, self.args.V_n]
		node_n1 = [self.args.s_n, self.args.ag11_n, self.args.ag14_n,self.args.ag16_n,self.args.ag20_n,self.args.ag58_n,self.args.site1_n,self.args.site2_n,self.args.site3_n,self.args.site4_n,self.args.sex1_n,self.args.sex2_n]
		for i in range(11):
			for j in range(node_n1[i]):
				if i == 0:
					# neigh_temp = self.a_p_list_train[j]
					neigh_temp = self.s_ag11_list_train
					neigh_train = s_neigh_list_train[j]
					curNode = "s" + str(j)
				elif i == 1:
					neigh_temp = self.p_a_list_train[j]
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_temp = self.v_p_list_train[j]
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				if len(neigh_temp):
					neigh_L = 0
					a_L = 0
					p_L = 0
					v_L = 0
					while neigh_L < 100: #maximum neighbor size = 100
						rand_p = random.random() #return p
						if rand_p > 0.5:
							if curNode[0] == "s":
								curNode = random.choice(self.s_ag11_list_train[int(curNode[1:])])
								if p_L < 46: #size constraint (make sure each type of neighobr is sampled)
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
								if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 46:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'v':
									if v_L < 11:
										neigh_train.append(curNode)
										neigh_L += 1
										v_L += 1
							elif curNode[0] == "v":
								curNode = random.choice(self.v_p_list_train[int(curNode[1:])])
								if p_L < 46:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
						else:
							if i == 0:
								curNode = ('s' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							else:
								curNode = ('v' + str(j))

		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					s_neigh_list_train[j] = list(s_neigh_list_train[j])
				elif i == 1:
					p_neigh_list_train[j] = list(p_neigh_list_train[j])
				else:
					v_neigh_list_train[j] = list(v_neigh_list_train[j])

		neigh_f = open(self.args.data_path1 + "het_neigh_train.txt", "w")
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = s_neigh_list_train[j]
					curNode = "s" + str(j)
				elif i == 1:
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n

		total_triple_n = [0.0] * 23 # nine kinds of triples
		het_walk_f = open(self.args.data_path1 + "het_random_walk_train.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 's':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[0] += 1
								elif neighNode[0] == 'a':
									total_triple_n[1] += 1
								elif neighNode[0] == 'b':
									total_triple_n[2] += 1
								elif neighNode[0] == 'c':
									total_triple_n[3] += 1
								elif neighNode[0] == 'd':
									total_triple_n[4] += 1
								elif neighNode[0] == 'r':
									total_triple_n[5] += 1
								elif neighNode[0] == 'f':
									total_triple_n[6] += 1
								elif neighNode[0] == 'g':
									total_triple_n[7] += 1
								elif neighNode[0] == 'h':
									total_triple_n[8] += 1
								elif neighNode[0] == 'i':
									total_triple_n[9] += 1
								elif neighNode[0] == 'j':
									total_triple_n[10] += 1
								elif neighNode[0] == 'k':
									total_triple_n[11] += 1
					elif centerNode[0]=='a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[12] += 1
								# elif neighNode[0] == 'p':
								# 	total_triple_n[4] += 1
								# elif neighNode[0] == 'v':
								# 	total_triple_n[5] += 1
					elif centerNode[0]=='b':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[13] += 1
								# elif neighNode[0] == 'p':
								# 	total_triple_n[7] += 1
								# elif neighNode[0] == 'v':
								# 	total_triple_n[8] += 1
					elif centerNode[0]=='c':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[14] += 1
					elif centerNode[0]=='d':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[15] += 1
					elif centerNode[0]=='e':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[16] += 1
					elif centerNode[0]=='f':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[17] += 1
					elif centerNode[0]=='g':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[18] += 1
					elif centerNode[0]=='h':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[19] += 1
					elif centerNode[0]=='i':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[20] += 1
					elif centerNode[0]=='j':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[21] += 1
					elif centerNode[0]=='k':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's':
									total_triple_n[22] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / ((total_triple_n[i] * 10)+float("1e-8"))
		print("sampling ratio computing finish.")

		return total_triple_n



	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(23)]
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		m_n = self.args.m_n

		triple_sample_p = self.triple_sample_p  # use sampling to avoid memory explosion

		het_walk_f = open(self.args.data_path1 + "het_random_walk_train.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				# print('tr:', centerNode)
				if len(centerNode) > 1:
					if centerNode[0] == 's':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_neigh_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'a' and random.random() < triple_sample_p[1]:
									negNode = 0
									while len(self.ag11_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'b' and random.random() < triple_sample_p[2]:
									negNode = 0
									while len(self.ag14_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
								elif neighNode[0] == 'c' and random.random() < triple_sample_p[3]:
									negNode = 0
									while len(self.ag16_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'd' and random.random() < triple_sample_p[4]:
									negNode = 0
									while len(self.ag20_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'e' and random.random() < triple_sample_p[5]:
									negNode = 0
									while len(self.ag58_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
								elif neighNode[0] == 'f' and random.random() < triple_sample_p[6]:
									negNode = 0
									while len(self.site1_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'g' and random.random() < triple_sample_p[7]:
									negNode = 0
									while len(self.site2_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'h' and random.random() < triple_sample_p[8]:
									negNode = 0
									while len(self.site3_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
								elif neighNode[0] == 'i' and random.random() < triple_sample_p[9]:
									negNode = 0
									while len(self.site4_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[9].append(triple)
								elif neighNode[0] == 'j' and random.random() < triple_sample_p[10]:
									negNode = 0
									while len(self.sex1_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[10].append(triple)
								elif neighNode[0] == 'k' and random.random() < triple_sample_p[11]:
									negNode = 0
									while len(self.sex2_s_list_train[negNode]) == 0:
										negNode = 0
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[11].append(triple)
					elif centerNode[0]=='a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[12]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_ag11_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[12].append(triple)
								# elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
								# 	negNode = random.randint(0, P_n - 1)
								# 	while len(self.p_a_list_train[negNode]) == 0:
								# 		negNode = random.randint(0, P_n - 1)
								# 	triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
								# 	triple_list[4].append(triple)
								# elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
								# 	negNode = random.randint(0, V_n - 1)
								# 	while len(self.v_p_list_train[negNode]) == 0:
								# 		negNode = random.randint(0, V_n - 1)
								# 	triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
								# 	triple_list[5].append(triple)
					elif centerNode[0]=='b':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[13]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_ag14_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[13].append(triple)
								# elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
								# 	negNode = random.randint(0, P_n - 1)
								# 	while len(self.p_a_list_train[negNode]) == 0:
								# 		negNode = random.randint(0, P_n - 1)
								# 	triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
								# 	triple_list[7].append(triple)
								# elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
								# 	negNode = random.randint(0, V_n - 1)
								# 	while len(self.v_p_list_train[negNode]) == 0:
								# 		negNode = random.randint(0, V_n - 1)
								# 	triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
								# 	triple_list[8].append(triple)
					elif centerNode[0]=='c':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[14]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_ag16_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[14].append(triple)
					elif centerNode[0]=='d':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[15]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_ag20_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[15].append(triple)
					elif centerNode[0]=='e':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[16]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_ag58_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[16].append(triple)
					elif centerNode[0]=='f':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[17]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_site1_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[17].append(triple)
					elif centerNode[0]=='g':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[18]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_site2_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[18].append(triple)
					elif centerNode[0]=='h':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[19]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_site3_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[19].append(triple)
					elif centerNode[0]=='i':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[20]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_site4_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[20].append(triple)
					elif centerNode[0]=='j':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[21]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_sex1_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[21].append(triple)
					elif centerNode[0]=='k':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 's' and random.random() < triple_sample_p[22]:
									negNode = random.randint(0, m_n - 1)
									while len(self.s_sex2_list_train[negNode]) == 0:
										negNode = random.randint(0, m_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[22].append(triple)
		het_walk_f.close()

		return triple_list




