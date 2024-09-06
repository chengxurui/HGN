import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
import pandas as pd

args = read_args()


class HetAgg(nn.Module):
    def __init__(self, args, feature_list, s_neigh_list_train1, ag11_neigh_list_train1,  ag16_neigh_list_train1, ag20_neigh_list_train1, ag58_neigh_list_train1,
                 site1_neigh_list_train1, site2_neigh_list_train1, site3_neigh_list_train1, site4_neigh_list_train1, sex1_neigh_list_train1, sex2_neigh_list_train1,
                 s_train_id_list, ag11_train_id_list,  ag16_train_id_list, ag20_train_id_list, ag58_train_id_list, site1_train_id_list,
                 site2_train_id_list, site3_train_id_list, site4_train_id_list, sex1_train_id_list, sex2_train_id_list,):
        super(HetAgg, self).__init__()
        embed_d = args.embed_d  # 嵌入维度
        in_f_d = args.in_f_d  # 输入维度
        self.args = args
        self.s_n = args.m_n
        self.ag11 = args.ag11_n
        # self.ag14 = args.ag14_n
        self.ag16 = args.ag16_n
        self.ag20 = args.ag20_n
        self.ag58 = args.ag58_n
        self.site1 = args.site1_n
        self.site2 = args.site2_n
        self.site3 = args.site3_n
        self.site4 = args.site4_n
        self.sex1 = args.sex1_n
        self.sex2 = args.sex2_n
        self.feature_list = feature_list  # 特征列表
        self.s_neigh_list_train1 = s_neigh_list_train1
        self.ag11_neigh_list_train1 = ag11_neigh_list_train1
        # self.ag14_neigh_list_train1 = ag14_neigh_list_train1
        self.ag16_neigh_list_train1 = ag16_neigh_list_train1
        self.ag20_neigh_list_train1 = ag20_neigh_list_train1
        self.ag58_neigh_list_train1 = ag58_neigh_list_train1
        self.site1_neigh_list_train1 = site1_neigh_list_train1
        self.site2_neigh_list_train1 = site2_neigh_list_train1
        self.site3_neigh_list_train1 = site3_neigh_list_train1
        self.site4_neigh_list_train1 = site4_neigh_list_train1
        self.sex1_neigh_list_train1 = sex1_neigh_list_train1
        self.sex2_neigh_list_train1 = sex2_neigh_list_train1
        self.s_train_id_list = s_train_id_list
        self.ag11_train_id_list = ag11_train_id_list
        # self.ag14_train_id_list = ag14_train_id_list
        self.ag16_train_id_list = ag16_train_id_list
        self.ag20_train_id_list = ag20_train_id_list
        self.ag58_train_id_list = ag58_train_id_list
        self.site1_train_id_list = site1_train_id_list
        self.site2_train_id_list = site2_train_id_list
        self.site3_train_id_list = site3_train_id_list
        self.site4_train_id_list = site4_train_id_list
        self.sex1_train_id_list = sex1_train_id_list
        self.sex2_train_id_list = sex2_train_id_list
        # self.a_neigh_list_train = a_neigh_list_train  # a，p，v的邻居
        # self.p_neigh_list_train = p_neigh_list_train
        # self.v_neigh_list_train = v_neigh_list_train
        # self.a_train_id_list = a_train_id_list  # a，p，v的id
        # self.p_train_id_list = p_train_id_list
        # self.v_train_id_list = v_train_id_list

        # self.fc_a_agg = nn.Linear(embed_d * 4, embed_d)

        # 特征由bilstm抽取
        self.a_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.p_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.v_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.s_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag11_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        # self.ag14_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag16_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag20_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag58_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site1_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site2_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site3_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site4_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.sex1_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.sex2_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.p_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.s_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag11_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        # self.ag14_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag16_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag20_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.ag58_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site1_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site2_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site3_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.site4_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.sex1_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.sex2_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

        # 注意力权重
        self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.s_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.ag11_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        # self.ag14_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.ag16_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.ag20_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.ag58_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.site1_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.site2_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.site3_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.site4_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.sex1_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.sex2_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)  # softmax
        self.act = nn.LeakyReLU()  # 激活函数
        self.drop = nn.Dropout(p=0.5)  # dropout
        self.bn = nn.BatchNorm1d(embed_d)  # 批正则
        self.embed_d = embed_d  # 嵌入维度

    def init_weights(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    # 下面三个函数都是各自的异质内容的聚合
    def s_content_agg(self, id_batch):
        embed_d = self.embed_d
        # print len(id_batch)
        # embed_d = in_f_d, it is flexible to add feature transformer (e.g., FC) here
        s_net_embed_batch = self.feature_list[0][id_batch]

        '''参数列表
        feature_list = [input_data.p_abstract_embed, input_data.p_title_embed,\
input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
input_data.v_net_embed, input_data.v_text_embed]
        '''
        # 7是a网络的嵌入
        # a_text_embed_batch_1 = self.feature_list[7][id_batch, :embed_d][0]
        # a_text_embed_batch_2 = self.feature_list[7][id_batch, embed_d: embed_d * 2][0]
        # a_text_embed_batch_3 = self.feature_list[7][id_batch, embed_d * 2: embed_d * 3][0]

        # concate_embed = torch.cat((a_net_embed_batch, a_text_embed_batch_1, a_text_embed_batch_2, \
        #                            a_text_embed_batch_3), 1).view(len(id_batch[0]), 4, embed_d)
        concate_embed = s_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.s_content_rnn(concate_embed)  # 用bilstm聚合异质内容 # [1, 2000, 1950] 1, 200, 1950]
        print('f:', all_state.shape)
        return torch.mean(all_state, 0)  # 降维

    def ag11_content_agg(self, id_batch):
        embed_d = self.embed_d
        ag11_net_embed_batch = self.feature_list[1][id_batch]
        concate_embed = ag11_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.ag11_content_rnn(concate_embed)  # 用bilstm聚合异质内容 [1, 2000, 1950]
        print('f1:', all_state.shape)
        print(all_state)
        return torch.mean(all_state, 0)

    # def ag14_content_agg(self, id_batch):
    #     embed_d = self.embed_d
    #     ag14_net_embed_batch = self.feature_list[2][id_batch]
    #     concate_embed = ag14_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
    #     concate_embed = torch.transpose(concate_embed, 0, 1)
    #     all_state, last_state = self.ag14_content_rnn(concate_embed)  # 用bilstm聚合异质内容
    #     print('all:', all_state)
    #     return torch.mean(all_state, 0)

    def ag16_content_agg(self, id_batch):
        embed_d = self.embed_d
        ag16_net_embed_batch = self.feature_list[3][id_batch]
        concate_embed = ag16_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        # print('ag26', concate_embed)
        all_state, last_state = self.ag16_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        print('ag26', all_state)
        return torch.mean(all_state, 0)

    def ag20_content_agg(self, id_batch):
        embed_d = self.embed_d
        ag20_net_embed_batch = self.feature_list[4][id_batch]
        concate_embed = ag20_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        # print('ag20', concate_embed)
        all_state, last_state = self.ag20_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        print('ag20', all_state)
        return torch.mean(all_state, 0)

    def ag58_content_agg(self, id_batch):
        embed_d = self.embed_d
        ag58_net_embed_batch = self.feature_list[5][id_batch]
        concate_embed = ag58_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.ag58_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        print('ag58', all_state)
        return torch.mean(all_state, 0)

    def site1_content_agg(self, id_batch):
        embed_d = self.embed_d
        site1_net_embed_batch = self.feature_list[6][id_batch]
        concate_embed = site1_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.site1_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        return torch.mean(all_state, 0)

    def site2_content_agg(self, id_batch):
        embed_d = self.embed_d
        site2_net_embed_batch = self.feature_list[7][id_batch]
        concate_embed = site2_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.site2_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        print('site2:', all_state)
        return torch.mean(all_state, 0)

    def site3_content_agg(self, id_batch):
        embed_d = self.embed_d
        site3_net_embed_batch = self.feature_list[8][id_batch]
        concate_embed = site3_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.site3_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        return torch.mean(all_state, 0)

    def site4_content_agg(self, id_batch):
        embed_d = self.embed_d
        site4_net_embed_batch = self.feature_list[9][id_batch]
        concate_embed = site4_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.site4_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        return torch.mean(all_state, 0)

    def sex1_content_agg(self, id_batch):
        embed_d = self.embed_d
        sex1_net_embed_batch = self.feature_list[10][id_batch]
        concate_embed = sex1_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.sex1_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        return torch.mean(all_state, 0)

    def sex2_content_agg(self, id_batch):
        embed_d = self.embed_d
        sex2_net_embed_batch = self.feature_list[11][id_batch]
        concate_embed = sex2_net_embed_batch.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.sex2_content_rnn(concate_embed)  # 用bilstm聚合异质内容
        return torch.mean(all_state, 0)
    def p_content_agg(self, id_batch):
        '''参数列表
        feature_list = [input_data.p_abstract_embed, input_data.p_title_embed,\
input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
input_data.v_net_embed, input_data.v_text_embed]
        '''
        embed_d = self.embed_d
        p_a_embed_batch = self.feature_list[0][id_batch]  # 论文的摘要
        p_t_embed_batch = self.feature_list[1][id_batch]  # 论文的标题
        p_v_net_embed_batch = self.feature_list[2][id_batch]  # 地点
        p_a_net_embed_batch = self.feature_list[3][id_batch]  # 论文中作者图
        p_net_embed_batch = self.feature_list[5][id_batch]  # 论文嵌入特征

        concate_embed = torch.cat((p_a_embed_batch, p_t_embed_batch, p_v_net_embed_batch, \
                                   p_a_net_embed_batch, p_net_embed_batch), 1).view(len(id_batch[0]), 5, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.p_content_rnn(concate_embed)  # 用bilstm聚合异质内容

        return torch.mean(all_state, 0)

    def v_content_agg(self, id_batch):
        '''参数列表
        feature_list = [input_data.p_abstract_embed, input_data.p_title_embed,\
input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
input_data.v_net_embed, input_data.v_text_embed]
        '''
        embed_d = self.embed_d
        v_net_embed_batch = self.feature_list[8][id_batch]  # 地点嵌入
        v_text_embed_batch_1 = self.feature_list[9][id_batch, :embed_d][0]  # 内容嵌入
        v_text_embed_batch_2 = self.feature_list[9][id_batch, embed_d: 2 * embed_d][0]
        v_text_embed_batch_3 = self.feature_list[9][id_batch, 2 * embed_d: 3 * embed_d][0]
        v_text_embed_batch_4 = self.feature_list[9][id_batch, 3 * embed_d: 4 * embed_d][0]
        v_text_embed_batch_5 = self.feature_list[9][id_batch, 4 * embed_d:][0]

        concate_embed = torch.cat((v_net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, v_text_embed_batch_3, \
                                   v_text_embed_batch_4, v_text_embed_batch_5), 1).view(len(id_batch[0]), 6, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.v_content_rnn(concate_embed)

        return torch.mean(all_state, 0)

    # 先分别聚合自己的邻居，再聚合异构的邻居，最后针对不同类型的重要性不一样给个注意力。
    def node_neigh_agg(self, id_batch, node_type):  # 用bilstm按类型聚合自己的邻居
        embed_d = self.embed_d

        # if node_type == 1 :
        batch_s = int(len(id_batch[0]) / 10)  # 200
        # else:
        #     # print (len(id_batch[0]))
        #     batch_s = 1

        if node_type == 1:  # s类型
            neigh_agg = self.s_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.s_neigh_rnn(neigh_agg)
        elif node_type == 2:  # ag11类型
            neigh_agg = self.ag11_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.ag11_neigh_rnn(neigh_agg)
        # elif node_type == 3:  # ag14类型
        #     neigh_agg = self.ag14_content_agg(id_batch).view(batch_s, 10, embed_d)
        #     neigh_agg = torch.transpose(neigh_agg, 0, 1)
        #     all_state, last_state = self.ag14_neigh_rnn(neigh_agg)
        elif node_type == 4:
            neigh_agg = self.ag16_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            # print('a16all:', neigh_agg) ####################################################
            all_state, last_state = self.ag16_neigh_rnn(neigh_agg)
            # print('a16all:', all_state)
        elif node_type == 5:
            neigh_agg = self.ag20_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            # print('ag20all:', neigh_agg)
            all_state, last_state = self.ag20_neigh_rnn(neigh_agg)
            # print('ag20all:',all_state)
        elif node_type == 6:
            neigh_agg = self.ag58_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            # print('ag58all:', neigh_agg)
            all_state, last_state = self.ag58_neigh_rnn(neigh_agg)
            # print('ag58all:', all_state)
        elif node_type == 7:
            neigh_agg = self.site1_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.site1_neigh_rnn(neigh_agg)
        elif node_type == 8:
            neigh_agg = self.site2_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.site2_neigh_rnn(neigh_agg)
        elif node_type == 9:
            neigh_agg = self.site3_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.site3_neigh_rnn(neigh_agg)
        elif node_type == 10:
            neigh_agg = self.site4_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.site4_neigh_rnn(neigh_agg)
        elif node_type == 11:
            neigh_agg = self.sex1_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.sex1_neigh_rnn(neigh_agg)
        elif node_type == 12:
            neigh_agg = self.sex2_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.sex2_neigh_rnn(neigh_agg)

        neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)

        return neigh_agg

    def node_het_agg(self, id_batch, node_type):  # 异构的邻居
        # a_neigh_batch = [[0] * 10] * len(id_batch)
        # p_neigh_batch = [[0] * 10] * len(id_batch)
        # v_neigh_batch = [[0] * 3] * len(id_batch)
        s_neigh_batch = [[0] * 10] * len(id_batch)  # 10 * 200
        ag11_neigh_batch = [[0] * 10] * len(id_batch)  # 10 * 200
        # ag14_neigh_batch = [[0] * 10] * len(id_batch)
        ag16_neigh_batch = [[0] * 10] * len(id_batch)
        ag20_neigh_batch = [[0] * 10] * len(id_batch)
        ag58_neigh_batch = [[0] * 10] * len(id_batch)
        site1_neigh_batch = [[0] * 10] * len(id_batch)
        site2_neigh_batch = [[0] * 10] * len(id_batch)
        site3_neigh_batch = [[0] * 10] * len(id_batch)
        site4_neigh_batch = [[0] * 10] * len(id_batch)
        sex1_neigh_batch = [[0] * 10] * len(id_batch)
        sex2_neigh_batch = [[0] * 10] * len(id_batch)
        for i in range(len(id_batch)):
            if node_type == 1:  # s类型，找到s的邻居列表中的3种类型的列表
                # s_neigh_batch[i] = self.s_neigh_list_train1[0][id_batch[i]]
                # print('frs',self.s_neigh_list_train1[1])
                if (len(self.s_neigh_list_train1[1][id_batch[i]])):
                    ag11_neigh_batch[i] = self.s_neigh_list_train1[1][id_batch[i]]
                # if (len(self.s_neigh_list_train1[2][id_batch[i]])):
                #     ag14_neigh_batch[i] = self.s_neigh_list_train1[2][id_batch[i]]
                if (len(self.s_neigh_list_train1[3][id_batch[i]])):
                    ag16_neigh_batch[i] = self.s_neigh_list_train1[3][id_batch[i]]
                if (len(self.s_neigh_list_train1[4][id_batch[i]])):
                    ag20_neigh_batch[i] = self.s_neigh_list_train1[4][id_batch[i]]
                if (len(self.s_neigh_list_train1[5][id_batch[i]])):
                    ag58_neigh_batch[i] = self.s_neigh_list_train1[5][id_batch[i]]
                if (len(self.s_neigh_list_train1[6][id_batch[i]])):
                    site1_neigh_batch[i] = self.s_neigh_list_train1[6][id_batch[i]]
                if (len(self.s_neigh_list_train1[7][id_batch[i]])):
                    site2_neigh_batch[i] = self.s_neigh_list_train1[7][id_batch[i]]
                if (len(self.s_neigh_list_train1[8][id_batch[i]])):
                    site3_neigh_batch[i] = self.s_neigh_list_train1[8][id_batch[i]]
                if (len(self.s_neigh_list_train1[9][id_batch[i]])):
                    site4_neigh_batch[i] = self.s_neigh_list_train1[9][id_batch[i]]
                if (len(self.s_neigh_list_train1[10][id_batch[i]])):
                    sex1_neigh_batch[i] = self.s_neigh_list_train1[10][id_batch[i]]
                if (len(self.s_neigh_list_train1[11][id_batch[i]])):
                    sex2_neigh_batch[i] = self.s_neigh_list_train1[11][id_batch[i]]
            elif node_type == 2:  # p类型，找到p的邻居列表中的3种类型的列
                if (len(self.ag11_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.ag11_neigh_list_train1[0][0]
                # p_neigh_batch[i] = self.p_neigh_list_train[1][id_batch[i]]
                # v_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]

            # elif node_type == 3:  # v类型，找到v的邻居列表中的3种类型的列表
            #     if (len(self.ag14_neigh_list_train1[0][0])):
            #        s_neigh_batch[i] = self.ag14_neigh_list_train1[0][0]

            elif node_type == 4:
                if (len(self.ag16_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.ag16_neigh_list_train1[0][0]
            elif node_type == 5:
                if (len(self.ag20_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.ag20_neigh_list_train1[0][0]
            elif node_type == 6:
                if (len(self.ag58_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.ag58_neigh_list_train1[0][0]
            elif node_type == 7:
                if (len(self.site1_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.site1_neigh_list_train1[0][0]
            elif node_type == 8:
                if (len(self.site2_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.site2_neigh_list_train1[0][0]
            elif node_type == 9:
                if (len(self.site3_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.site3_neigh_list_train1[0][0]
            elif node_type == 10:
                if (len(self.site4_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.site4_neigh_list_train1[0][0]
            elif node_type == 11:
                if (len(self.sex1_neigh_list_train1[0][0])):
                   s_neigh_batch[i] = self.sex1_neigh_list_train1[0][0]
            elif node_type == 12:
                if (len(self.sex2_neigh_list_train1[0][0])):
                  s_neigh_batch[i] = self.sex2_neigh_list_train1[0][0]

        # a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))
        # a_agg_batch = self.node_neigh_agg(a_neigh_batch, 1)  # 异构邻居聚合
        # p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
        # p_agg_batch = self.node_neigh_agg(p_neigh_batch, 2)
        # v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
        # v_agg_batch = self.node_neigh_agg( v_neigh_batch, 3)
        b = [[]]
        # print('0000:', len(s_neigh_batch)) 200
        # print(s_neigh_batch)  # 现在全为0
        for i in range(len(s_neigh_batch)):
            for j in range(len(s_neigh_batch[i])):
                b[0].append(s_neigh_batch[i][j])
        s_neigh_batch = np.array(b)  # 2000
        s_agg_batch = self.node_neigh_agg(s_neigh_batch, 1)  # 异构邻居聚合 返回节点的特征维度[1, 2000, 1950]
        b = [[]]
        for i in range(len(ag11_neigh_batch)):
            for j in range(len(ag11_neigh_batch[i])):
                b[0].append(ag11_neigh_batch[i][j])
        ag11_neigh_batch = np.array(b)  # 合并为一维数组[0,0,0,0,0,11,11....] len为2000
        # print('ag11', ag11_neigh_batch)
        ag11_agg_batch = self.node_neigh_agg(ag11_neigh_batch, 2)
        # print('ag11', ag11_agg_batch)
        # b = [[]]
        # for i in range(len(ag14_neigh_batch)):
        #     for j in range(len(ag14_neigh_batch[i])):
        #         b[0].append(ag14_neigh_batch[i][j])
        # ag14_neigh_batch = np.array(b)
        # ag14_agg_batch = self.node_neigh_agg(ag14_neigh_batch, 3)
        b = [[]]
        for i in range(len(ag16_neigh_batch)):
            for j in range(len(ag16_neigh_batch[i])):
                b[0].append(ag16_neigh_batch[i][j])
        ag16_neigh_batch = np.array(b)
        # print('ag16a:', ag16_neigh_batch)
        ag16_agg_batch = self.node_neigh_agg(ag16_neigh_batch, 4)
        # print('ag16a:', ag16_agg_batch)
        b = [[]]
        for i in range(len(ag20_neigh_batch)):
            for j in range(len(ag20_neigh_batch[i])):
                b[0].append(ag20_neigh_batch[i][j])
        ag20_neigh_batch = np.array(b)
        # print('ag20:', ag20_neigh_batch)
        ag20_agg_batch = self.node_neigh_agg(ag20_neigh_batch, 5)
        # print('ag20:', ag20_agg_batch)
        b = [[]]
        for i in range(len(ag58_neigh_batch)):
            for j in range(len(ag58_neigh_batch[i])):
                b[0].append(ag58_neigh_batch[i][j])
        ag58_neigh_batch = np.array(b)
        # print('ag58:', ag58_neigh_batch)
        ag58_agg_batch = self.node_neigh_agg(ag58_neigh_batch, 6)
        # print('ag58:', ag58_agg_batch)
        b = [[]]
        for i in range(len(site1_neigh_batch)):
            for j in range(len(site1_neigh_batch[i])):
                b[0].append(site1_neigh_batch[i][j])
        site1_neigh_batch = np.array(b)
        # print('dite1:', site1_neigh_batch)
        site1_agg_batch = self.node_neigh_agg(site1_neigh_batch, 7)
        # print('dite1:', site1_agg_batch)
        b = [[]]
        for i in range(len(site2_neigh_batch)):
            for j in range(len(site2_neigh_batch[i])):
                b[0].append(site2_neigh_batch[i][j])
        site2_neigh_batch = np.array(b)
        # print('site2-agg:', site2_neigh_batch)
        site2_agg_batch = self.node_neigh_agg(site2_neigh_batch, 8)
        # print('site2-agg:', site2_agg_batch)
        b = [[]]
        for i in range(len(site3_neigh_batch)):
            for j in range(len(site3_neigh_batch[i])):
                b[0].append(site3_neigh_batch[i][j])
        site3_neigh_batch = np.array(b)
        # print('dite3:', site3_neigh_batch)
        site3_agg_batch = self.node_neigh_agg(site3_neigh_batch, 9)
        # print('dite3:', site3_agg_batch)
        b = [[]]
        for i in range(len(site4_neigh_batch)):
            for j in range(len(site4_neigh_batch[i])):
                b[0].append(site4_neigh_batch[i][j])
        site4_neigh_batch = np.array(b)
        # print('dite4:', site4_neigh_batch)
        site4_agg_batch = self.node_neigh_agg(site4_neigh_batch, 10)
        # print('dite4:', site4_agg_batch)
        b = [[]]
        for i in range(len(sex1_neigh_batch)):
            for j in range(len(sex1_neigh_batch[i])):
                b[0].append(sex1_neigh_batch[i][j])
        sex1_neigh_batch = np.array(b)
        # print('sex1:', sex1_neigh_batch)
        sex1_agg_batch = self.node_neigh_agg(sex1_neigh_batch, 11)
        # print('sex1:', sex1_agg_batch)
        b = [[]]
        for i in range(len(sex2_neigh_batch)):
            for j in range(len(sex2_neigh_batch[i])):
                b[0].append(sex2_neigh_batch[i][j])
        sex2_neigh_batch = np.array(b)
        # print('sex2:', sex2_neigh_batch)
        sex2_agg_batch = self.node_neigh_agg(sex2_neigh_batch, 12)
        # print('sex2:', sex2_agg_batch)



        # 注意力模块
        id_batch = np.reshape(id_batch, (1, -1))   # [[0,1,2,3,4......,199]] len为200
        if node_type == 1:
            c_agg_batch = self.s_content_agg(id_batch) # [1,200,1950]
        elif node_type == 2:
            c_agg_batch = self.ag11_content_agg(id_batch)
        # elif node_type == 3:
        #     c_agg_batch = self.ag14_content_agg(id_batch)
        elif node_type == 4:
            c_agg_batch = self.ag16_content_agg(id_batch)
        elif node_type == 5:
            c_agg_batch = self.ag20_content_agg(id_batch)
        elif node_type == 6:
            c_agg_batch = self.ag58_content_agg(id_batch)
        elif node_type == 7:
            c_agg_batch = self.site1_content_agg(id_batch)
        elif node_type == 8:
            c_agg_batch = self.site2_content_agg(id_batch)
        elif node_type == 9:
            c_agg_batch = self.site3_content_agg(id_batch)
        elif node_type == 10:
            c_agg_batch = self.site4_content_agg(id_batch)
        elif node_type == 11:
            c_agg_batch = self.sex1_content_agg(id_batch)
        elif node_type == 12:
            c_agg_batch = self.sex2_content_agg(id_batch)
        # print('fv:', c_agg_batch.shape)
        # print(s_agg_batch.shape)
        # print(ag11_neigh_batch.shape)
        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)  # [200, 3900]
        # print('c:', c_agg_batch_2)
        # a_agg_batch_2 = torch.cat((c_agg_batch, a_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # v_agg_batch_2 = torch.cat((c_agg_batch, v_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        s_agg_batch_2 = torch.cat((c_agg_batch, s_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('s:', s_agg_batch_2)
        ag11_agg_batch_2 = torch.cat((c_agg_batch, ag11_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('ag11a:', ag11_agg_batch_2)
        # ag14_agg_batch_2 = torch.cat((c_agg_batch, ag14_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        ag16_agg_batch_2 = torch.cat((c_agg_batch, ag16_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('ag16', ag16_agg_batch_2)
        ag20_agg_batch_2 = torch.cat((c_agg_batch, ag20_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('ag20a:', ag20_agg_batch_2)
        ag58_agg_batch_2 = torch.cat((c_agg_batch, ag58_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('ag58a:', ag58_agg_batch_2)
        site1_agg_batch_2 = torch.cat((c_agg_batch, site1_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('site1a:', site1_agg_batch_2)
        site2_agg_batch_2 = torch.cat((c_agg_batch, site2_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('site2a:', site2_agg_batch_2)
        site3_agg_batch_2 = torch.cat((c_agg_batch, site3_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('siteag:', site3_agg_batch_2)
        site4_agg_batch_2 = torch.cat((c_agg_batch, site4_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('site4a:', site4_agg_batch_2)
        sex1_agg_batch_2 = torch.cat((c_agg_batch, sex1_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('sex1a:', sex1_agg_batch_2)
        sex2_agg_batch_2 = torch.cat((c_agg_batch, sex2_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # print('sex2a:', sex2_agg_batch_2)
        # 最后用权重把不同的节点类型聚合起来
        concate_embed = torch.cat((c_agg_batch_2, s_agg_batch_2, ag11_agg_batch_2, ag16_agg_batch_2, ag20_agg_batch_2, ag58_agg_batch_2,
                                   site1_agg_batch_2, site2_agg_batch_2, site3_agg_batch_2, site4_agg_batch_2, sex1_agg_batch_2, sex2_agg_batch_2), 1).view(len(c_agg_batch), 12, self.embed_d * 2)
        # print('en1:', concate_embed)  #[200, 13, 3900]
        if node_type == 1:
            # print('tretefgrtg',self.s_neigh_att)
            atten_w = self.act(torch.bmm(concate_embed, self.s_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.s_neigh_att.size())))
            # print(atten_w.shape) [200, 13, 1]
        elif node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.ag11_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.ag11_neigh_att.size())))
        # elif node_type == 3:
        #     atten_w = self.act(torch.bmm(concate_embed, self.ag14_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
        #                                                                                      *self.ag14_neigh_att.size())))
        elif node_type == 4:
            atten_w = self.act(torch.bmm(concate_embed, self.ag16_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.ag16_neigh_att.size())))
        elif node_type == 5:
            atten_w = self.act(torch.bmm(concate_embed, self.ag20_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.ag20_neigh_att.size())))
        elif node_type == 6:
            atten_w = self.act(torch.bmm(concate_embed, self.ag58_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.ag58_neigh_att.size())))
        elif node_type == 7:
            atten_w = self.act(torch.bmm(concate_embed, self.site1_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.site1_neigh_att.size())))
        elif node_type == 8:
            atten_w = self.act(torch.bmm(concate_embed, self.site2_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.site2_neigh_att.size())))
        elif node_type == 9:
            atten_w = self.act(torch.bmm(concate_embed, self.site3_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.site3_neigh_att.size())))
        elif node_type == 10:
            atten_w = self.act(torch.bmm(concate_embed, self.site4_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.site4_neigh_att.size())))
        elif node_type == 11:
            atten_w = self.act(torch.bmm(concate_embed, self.sex1_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.sex1_neigh_att.size())))
        elif node_type == 12:
            atten_w = self.act(torch.bmm(concate_embed, self.sex2_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                                                                                             *self.sex2_neigh_att.size())))

        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 12)  # softmax torch.Size([200, 1, 13])  (batch_size, n, m)
        a = []
        for i in atten_w[0]:
            a.append(i)
        print('a:', a)
        print('atten_w:', atten_w)
        print(atten_w.shape)


        # 最后用权重把不同的节点类型聚合起来
        concate_embed = torch.cat((c_agg_batch, s_agg_batch, ag11_agg_batch,  ag16_agg_batch,ag20_agg_batch, ag58_agg_batch,
                                   site1_agg_batch, site2_agg_batch, site3_agg_batch, site4_agg_batch, sex1_agg_batch, sex2_agg_batch), 1).view(len(c_agg_batch), 12, self.embed_d)  # [200,13,1950]
        # print(concate_embed.shape)  torch.Size([200, 13, 1950]) (batch_size, m, p)
        # mm=torch.bmm(atten_w, concate_embed)  # torch.Size([200, 1, 1950]) (batch_size, n, p)
        print('concate14:', concate_embed)
        c = []
        for j in concate_embed[0]:
            c.append(j)
        print('c:', c)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)  # torch.Size([200, 1950])
        print('wei:', weight_agg_batch)
        return weight_agg_batch



    def average(self, numbers):

        total_sum = sum(numbers)  # 求和
        num_elements = len(numbers)  # 元素数量

        average = total_sum / num_elements  # 计算平均值
        # print("Average:", average)
        return average
    def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        embed_d = self.embed_d
        # batch processing
        # nine cases for academic data (author, paper, venue)
        # triple是由正例和负例组成。由于有三种类型，所以可以组成9种不同的case
        # print('r:', triple_index)
        if triple_index == 0:
            c_agg = self.node_het_agg(c_id_batch, 1)   # len 200 三元组中的第一个元素的集合
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 1:
            c_agg = self.node_het_agg(c_id_batch, 1)
            # print('po:',pos_id_batch)
            p_agg = self.node_het_agg(pos_id_batch, 2)
            n_agg = self.node_het_agg(neg_id_batch, 2)
        elif triple_index == 2:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 3)
            n_agg = self.node_het_agg(neg_id_batch, 3)
        elif triple_index == 3:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 4)
            n_agg = self.node_het_agg(neg_id_batch, 4)
        elif triple_index == 4:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 5)
            n_agg = self.node_het_agg(neg_id_batch, 5)
        elif triple_index == 5:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 6)
            n_agg = self.node_het_agg(neg_id_batch, 6)
        elif triple_index == 6:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 7)
            n_agg = self.node_het_agg(neg_id_batch, 7)
        elif triple_index == 7:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 8)
            n_agg = self.node_het_agg(neg_id_batch, 8)
        elif triple_index == 8:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 9)
            n_agg = self.node_het_agg(neg_id_batch, 9)
        elif triple_index == 9:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 10)
            n_agg = self.node_het_agg(neg_id_batch, 10)
        elif triple_index == 10:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 11)
            n_agg = self.node_het_agg(neg_id_batch, 11)
        elif triple_index == 11:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 12)
            n_agg = self.node_het_agg(neg_id_batch, 12)
        elif triple_index == 12:
            c_agg = self.node_het_agg(c_id_batch, 2)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 13:
            c_agg = self.node_het_agg(c_id_batch, 3)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 14:
            c_agg = self.node_het_agg(c_id_batch, 4)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 15:
            c_agg = self.node_het_agg(c_id_batch, 5)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 16:
            c_agg = self.node_het_agg(c_id_batch, 6)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 17:
            c_agg = self.node_het_agg(c_id_batch, 7)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 18:
            c_agg = self.node_het_agg(c_id_batch, 8)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 19:
            c_agg = self.node_het_agg(c_id_batch, 9)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 20:
            c_agg = self.node_het_agg(c_id_batch, 10)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 21:
            c_agg = self.node_het_agg(c_id_batch, 11)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 22:
            c_agg = self.node_het_agg(c_id_batch, 12)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 23:  # 保存学习到的节点嵌入
            embed_file = open(self.args.data_path1 + "node_ag14.txt", "w")
            save_batch_s = self.args.mini_batch_s
            for i in range(12):
                print('898989')
                if i == 0:
                    batch_number = int(len(self.s_train_id_list) / save_batch_s)
                    # print('0:', batch_number) 4
                elif i == 1:
                    batch_number = int(len(self.ag11_train_id_list) / save_batch_s)
                    # print('1:', batch_number) # 0
                # elif i == 2:
                #     batch_number = int(len(self. _train_id_list) / save_batch_s)
                elif i == 3:
                    batch_number = int(len(self.ag16_train_id_list) / save_batch_s)
                elif i == 4:
                    batch_number = int(len(self.ag20_train_id_list) / save_batch_s)
                elif i == 5:
                    batch_number = int(len(self.ag58_train_id_list) / save_batch_s)
                elif i == 6:
                    batch_number = int(len(self.site1_train_id_list) / save_batch_s)
                elif i == 7:
                    batch_number = int(len(self.site2_train_id_list) / save_batch_s)
                elif i == 8:
                    batch_number = int(len(self.site3_train_id_list) / save_batch_s)
                elif i == 9:
                    batch_number = int(len(self.site4_train_id_list) / save_batch_s)
                elif i == 10:
                    batch_number = int(len(self.sex1_train_id_list) / save_batch_s)
                elif i == 11:
                    batch_number = int(len(self.sex2_train_id_list) / save_batch_s)
                # print('ba:', batch_number)
                for j in range(batch_number):  # 不同的batch
                    if i == 0:  # 按同一类型存储
                        id_batch = self.s_train_id_list[j * save_batch_s: (j + 1) * save_batch_s] # 200
                        print('id:', id_batch)
                        out_temp = self.node_het_agg(id_batch, 1)
                        print('out:', out_temp)
                    elif i == 1:
                        id_batch = self.ag11_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        print('id:', id_batch)
                        out_temp = self.node_het_agg(id_batch, 2)
                        print('out1:', out_temp)
                    # elif i == 2:
                    #     id_batch = self.ag14_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    #     out_temp = self.node_het_agg(id_batch, 3)
                    elif i == 3:
                        id_batch = self.ag16_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 4)
                    elif i == 4:
                        id_batch = self.ag20_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 5)
                    elif i == 5:
                        id_batch = self.ag58_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 6)
                    elif i == 6:
                        id_batch = self.site1_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 7)
                    elif i == 7:
                        id_batch = self.site2_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 8)
                    elif i == 8:
                        id_batch = self.site3_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 9)
                    elif i == 9:
                        id_batch = self.site4_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 10)
                    elif i == 10:
                        id_batch = self.sex1_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 11)
                    elif i == 11:
                        id_batch = self.sex2_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 12)
                    out_temp = out_temp.data.cpu().numpy()
                    # print('d',len(id_batch))
                    for k in range(len(id_batch)):
                        index = id_batch[k]
                        if i == 0:
                            embed_file.write('s' + str(index) + " ")
                        elif i == 1:
                            embed_file.write('a' + str(index) + " ")
                        elif i == 2:
                            embed_file.write('b' + str(index) + " ")
                        elif i == 3:
                            embed_file.write('c' + str(index) + " ")
                        elif i == 4:
                            embed_file.write('d' + str(index) + " ")
                        elif i == 5:
                            embed_file.write('e' + str(index) + " ")
                        elif i == 6:
                            embed_file.write('f' + str(index) + " ")
                        elif i == 7:
                            embed_file.write('g' + str(index) + " ")
                        elif i == 8:
                            embed_file.write('h' + str(index) + " ")
                        elif i == 9:
                            embed_file.write('i' + str(index) + " ")
                        elif i == 10:
                            embed_file.write('j' + str(index) + " ")
                        elif i == 11:
                            embed_file.write('k' + str(index) + " ")
                        for l in range(embed_d - 1):
                            embed_file.write(str(out_temp[k][l]) + " ")
                        embed_file.write(str(out_temp[k][-1]) + "\n")
                # 不够batch剩下的也要save
                if i == 0:
                    id_batch = self.s_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 1)
                    out_temp = out_temp.detach().numpy()
                elif i == 1:
                    id_batch = self.ag11_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 2)
                    out_temp = out_temp.detach().numpy()
                # elif i == 2:
                #     id_batch = self.ag14_train_id_list[batch_number * save_batch_s: -1]
                #     out_temp = self.node_het_agg(id_batch, 3)
                #     out_temp = out_temp.detach().numpy()
                elif i == 3:
                    id_batch = self.ag16_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 4)
                    out_temp = out_temp.detach().numpy()
                elif i == 4:
                    id_batch = self.ag20_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 5)
                    out_temp = out_temp.detach().numpy()
                elif i == 5:
                    id_batch = self.ag58_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 6)
                    out_temp = out_temp.detach().numpy()
                elif i == 8:
                    id_batch = self.site1_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 7)
                    out_temp = out_temp.detach().numpy()
                elif i == 7:
                    id_batch = self.site2_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 8)
                    out_temp = out_temp.detach().numpy()
                elif i == 8:
                    id_batch = self.site3_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 9)
                    out_temp = out_temp.detach().numpy()
                elif i == 9:
                    id_batch = self.site4_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 10)
                    out_temp = out_temp.detach().numpy()
                elif i == 10:
                    id_batch = self.sex1_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 11)
                    out_temp = out_temp.detach().numpy()
                elif i == 11:
                    id_batch = self.sex2_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, 12)
                    out_temp = out_temp.detach().numpy()

                # out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    # print('rgggggggggggggg')
                    index = id_batch[k]
                    if i == 0:
                        embed_file.write('s' + str(index) + " ")
                    elif i == 1:
                        embed_file.write('a' + str(index) + " ")
                    elif i == 2:
                        embed_file.write('b' + str(index) + " ")
                    elif i == 3:
                        embed_file.write('c' + str(index) + " ")
                    elif i == 4:
                        embed_file.write('d' + str(index) + " ")
                    elif i == 5:
                        embed_file.write('e' + str(index) + " ")
                    elif i == 6:
                        embed_file.write('f' + str(index) + " ")
                    elif i == 7:
                        embed_file.write('g' + str(index) + " ")
                    elif i == 8:
                        embed_file.write('h' + str(index) + " ")
                    elif i == 9:
                        embed_file.write('i' + str(index) + " ")
                    elif i == 10:
                        embed_file.write('j' + str(index) + " ")
                    elif i ==11:
                        embed_file.write('k' + str(index) + " ")
                    for l in range(embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")
            embed_file.close()
            return [], [], []

        return c_agg, p_agg, n_agg

    def aggregate_all(self, triple_list_batch, triple_index):
        c_id_batch = [x[0] for x in triple_list_batch]  # [a,b,c]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]
        print('cid:', len(c_id_batch))

        c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

        return c_agg, pos_agg, neg_agg

    def forward(self, triple_list_batch, triple_index):
        # 汇聚所有
        print("forward-------------" + str(triple_index))
        c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
        print('c_out:', c_out)
        return c_out, p_out, n_out


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
    # 交叉熵损失，pos是正例，neg是负例。
    batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

    c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
    neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

    out_p = torch.bmm(c_embed, pos_embed)
    out_n = - torch.bmm(c_embed, neg_embed)
    # 计算正例和负例的log Sigmoid loss
    sum_p = F.logsigmoid(out_p)
    sum_n = F.logsigmoid(out_n)
    loss_sum = - (sum_p + sum_n)

    # loss_sum = loss_sum.sum() / batch_size

    return loss_sum.mean()  # 求平均


