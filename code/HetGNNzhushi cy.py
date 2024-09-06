
import torch
import torch.optim as optim
import data_generator
import dd
import tools
import toolzhushi
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
import site2
import ag14
torch.set_num_threads(2)  # 设置线程
import os
from sklearn.model_selection import StratifiedKFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class model_class(object):
    def __init__(self, args):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = args.cuda

        # 导入各种数据
        input = dd.input_data(args=self.args)
        input_data = data_generator.input_data(args=self.args)
        print('in:',input_data)
        # input_data.gen_het_rand_walk()

        self.input_data = input_data
        self.input = input

        if self.args.train_test_label == 2:  # 为每个node生成邻居
            # 重启策略的随机游走，为每个节点采样固定数量的强相关的异构邻居，然后按类型分组
            # 任意节点开始随机游走，以p概率返回。采样到固定数量后就停止。
            # 为了采样采样的邻居包含所有类型的节点，不同类型节点的数量是受限的。
            # 对每个类型都选出按频率的topk邻居
            input_data.het_walk_restart()
            print("neighbor set generation finish")
            exit(0)




        num_classes = 2
        subject_IDs = input.get_ids()
        num_nodes = len(subject_IDs)  # 871
        labels = input.get_subject_score(subject_IDs, score='DX_GROUP')
        sites = input.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        y_data = np.zeros([num_nodes, num_classes])
        y1 = np.zeros([882, 1])
        site = np.zeros([num_nodes, 1], dtype=np.int)
        y = np.zeros([num_nodes, 1])
        for i in range(num_nodes):
            y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]])
            y1[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])


        features = input.get_networks(subject_IDs, kind=args.connectivity, atlas_name=args.atlas)  # (871, 6105)

        # train/test
        skf = StratifiedKFold(n_splits=10)
        cv_splits = list(skf.split(features, np.squeeze(y)))
        train_index = cv_splits[args.folds][0]  # 782
        train_index = np.array(train_index)
        test_index = cv_splits[args.folds][1]  # 89
        labeled_ind = input.site_percentage(train_index, self.args.num_training, subject_IDs)
        # a = np.zeros([11, 6105])
        # features = np.append(features, values=a, axis=0)
        features = input.feature_selection(features, y, labeled_ind, self.args.num_features)
        print('ded:', features)
        # print(type(features))
        features_ag11 = np.zeros((12, 1950))
        # print(type(features_ag11))
        # print('        ',features_ag11)
        features_ag14 = np.zeros((15, 1950))
        features_ag16 = np.zeros((17,1950))
        features_ag20 = np.zeros((21, 1950))
        features_ag58 = np.zeros((59, 1950))
        features_site1 = np.zeros((64, 1950))
        features_site2 = np.zeros((61, 1950))
        features_site3 = np.zeros((62, 1950))
        features_site4 = np.zeros((63, 1950))
        features_sex1 = np.zeros((64, 1950))
        features_sex2 = np.zeros((66, 1950))
        feature_list1 = [features, features_ag11,features_ag14,  features_ag16, features_ag20, features_ag58, features_site1, features_site2, features_site3, features_site4, features_sex1, features_sex2]
        # print(feature_list1)
        for i in range(len(feature_list1)):
            # print('1:',i)
            feature_list1[i] = torch.from_numpy(np.array(feature_list1[i])).float()
            # print(feature_list1[i].dtype)

        if self.gpu:
            for i in range(len(feature_list1)):
                feature_list1[i] = feature_list1[i].cuda()
        # self.feature_list = feature_list


        s_neigh_list_train1 = input.s_neigh_list_train1
        print("sert:",s_neigh_list_train1)
        ag11_neigh_list_train1 = input.ag11_neigh_list_train1
        ag14_neigh_list_train1 = input.ag14_neigh_list_train1
        ag16_neigh_list_train1 = input.ag16_neigh_list_train1
        ag20_neigh_list_train1 = input.ag20_neigh_list_train1
        ag58_neigh_list_train1 = input.ag58_neigh_list_train1
        site1_neigh_list_train1 = input.site1_neigh_list_train1
        site2_neigh_list_train1 = input.site2_neigh_list_train1
        site3_neigh_list_train1 = input.site3_neigh_list_train1
        site4_neigh_list_train1 = input.site4_neigh_list_train1
        sex1_neigh_list_train1 = input.sex1_neigh_list_train1
        sex2_neigh_list_train1 = input.sex2_neigh_list_train1

        s_train_id_list = input.s_train_id_list
        print('sefg:', s_train_id_list)
        ag11_train_id_list = input.ag11_train_id_list
        ag14_train_id_list = input.ag14_train_id_list
        ag16_train_id_list = input.ag16_train_id_list
        ag20_train_id_list = input.ag20_train_id_list
        ag58_train_id_list = input.ag58_train_id_list
        site1_train_id_list = input.site1_train_id_list
        site2_train_id_list = input.site2_train_id_list
        site3_train_id_list = input.site3_train_id_list
        site4_train_id_list = input.site4_train_id_list
        sex1_train_id_list = input.sex1_train_id_list
        sex2_train_id_list = input.sex2_train_id_list



        # 各自的邻居列表
        a_neigh_list_train = input_data.a_neigh_list_train
        p_neigh_list_train = input_data.p_neigh_list_train
        v_neigh_list_train = input_data.v_neigh_list_train

        a_train_id_list = input_data.a_train_id_list
        p_train_id_list = input_data.p_train_id_list
        v_train_id_list = input_data.v_train_id_list

        # self.model = tools.HetAgg(args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train, \
        #                           a_train_id_list, p_train_id_list, v_train_id_list)  # 实例化model，tools会对异构的信息进行聚合

        self.model = toolzhushi.HetAgg(args, feature_list1, s_neigh_list_train1, ag11_neigh_list_train1,
                                       ag14_neigh_list_train1, ag16_neigh_list_train1, ag20_neigh_list_train1,
                                       ag58_neigh_list_train1, site1_neigh_list_train1, site2_neigh_list_train1,
                                       site3_neigh_list_train1, site4_neigh_list_train1, sex1_neigh_list_train1,
                                       sex2_neigh_list_train1, s_train_id_list, ag11_train_id_list, ag14_train_id_list,
                                       ag16_train_id_list, ag20_train_id_list, ag58_train_id_list, site1_train_id_list,
                                       site2_train_id_list, site3_train_id_list, site4_train_id_list, sex1_train_id_list,
                                       sex2_train_id_list,)  # 实例化model，tools会对异构的信息进行聚合

        # self.model = ag14.HetAgg(args, feature_list1, s_neigh_list_train1, ag11_neigh_list_train1,
        #                               ag16_neigh_list_train1, ag20_neigh_list_train1,
        #                          ag58_neigh_list_train1,
        #                          site1_neigh_list_train1, site2_neigh_list_train1, site3_neigh_list_train1,
        #                          site4_neigh_list_train1, sex1_neigh_list_train1, sex2_neigh_list_train1,
        #                          s_train_id_list, ag11_train_id_list, ag16_train_id_list,
        #                          ag20_train_id_list, ag58_train_id_list, site1_train_id_list, site2_train_id_list,
        #                          site3_train_id_list, site4_train_id_list, sex1_train_id_list,
        #                          sex2_train_id_list, )  # 实例化model，tools会对异构的信息进行聚合

        # self.model = site2.HetAgg(args, feature_list1, s_neigh_list_train1, ag11_neigh_list_train1, ag14_neigh_list_train1,
        #                               ag16_neigh_list_train1, ag20_neigh_list_train1,
        #                                ag58_neigh_list_train1,
        #                                site1_neigh_list_train1, site3_neigh_list_train1,
        #                                site4_neigh_list_train1, sex1_neigh_list_train1, sex2_neigh_list_train1,
        #                                s_train_id_list, ag11_train_id_list, ag14_train_id_list, ag16_train_id_list,
        #                                ag20_train_id_list, ag58_train_id_list, site1_train_id_list,
        #                                site3_train_id_list, site4_train_id_list, sex1_train_id_list,
        #                                sex2_train_id_list, )  # 实例化model，tools会对异构的信息进行聚合


        if self.gpu:
            self.model.cuda()
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)  # Adam优化器
        self.model.init_weights()

    def model_train(self):
        # 开始训练
        print('model training ...')
        if self.args.checkpoint != '':
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.train()  # 模型调到训练模式
        mini_batch_s = self.args.mini_batch_s  # batch 50
        embed_d = self.args.embed_d  # 嵌入维度

        for iter_i in range(self.args.train_iter_n):  # 迭代次数
            print('iteration ' + str(iter_i) + ' ...')
            triple_list = self.input.sample_het_walk_triple()  # 异构三元组（含正例 负例）采样 23
            # print('tri:',triple_list.shape)
            min_len = 1e10
            # print('len:',len(triple_list))
            for ii in range(len(triple_list)):
                if len(triple_list[ii]) < min_len:
                    min_len = len(triple_list[ii])
            print('min_', min_len)
            batch_n = int(min_len / mini_batch_s)
            # batch_n = 23
            print('4',batch_n)  # 9
            for k in range(batch_n):
                c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])  # [23, 200, 2100]
                p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])  # pos，正例
                n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])  # neg，负例
                print(';;;;;;;;;;;;;;;;')

                for triple_index in range(len(triple_list)):
                    triple_list_temp = triple_list[triple_index]
                    # print('l:', triple_list_temp)
                    triple_list_batch = triple_list_temp[k * mini_batch_s: (k + 1) * mini_batch_s]  # 取其中一部分 200个
                    # print('tr:', triple_list_batch)
                    # print(triple_list_batch.shape)
                    # print('tr:',triple_index)
                    # print(triple_list_batch)
                    # 得到模型的预测结果
                    c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)
                    # print('ce:',c_out_temp.shape)
                    # print(p_out_temp.shape)
                    # print(n_out_temp.shape)


                    c_out[triple_index] = c_out_temp
                    p_out[triple_index] = p_out_temp
                    n_out[triple_index] = n_out_temp
                # print('c:', c_out.shape)  # [9, 200, 128]
                # print(p_out)
                # print(n_out)

                loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)  # 计算三元组交叉熵

                self.optim.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                self.optim.step()  # 参数更新

                if k % 100 == 0:  # 打印结果
                    print("loss: " + str(loss))

            if iter_i % self.args.save_model_freq == 0:
                torch.save(self.model.state_dict(), self.args.model_path + "HetGNN_" + str(iter_i) + ".pt")
                # 存储参数用于评估
                triple_index = 23  # 一共有9种case，在tools文件中定义
                a_out, p_out, v_out = self.model([], triple_index)
            print('iteration ' + str(iter_i) + ' finish.')


if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))

    # 可复现随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # 实例化模型
    model_object = model_class(args)

    if args.train_test_label == 0:
        model_object.model_train()

