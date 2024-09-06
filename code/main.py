import data_generator
import random
from args import  read_args

class model_class(object):
    def __init__(self, args):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = args.cuda
    def het_walk_restart(self, args):
        self.args =args
        a_neigh_list_train = [[] for k in range(self.args.A_n)]
        p_neigh_list_train = [[] for k in range(self.args.P_n)]
        v_neigh_list_train = [[] for k in range(self.args.V_n)]

        # generate neighbor set via random walk with restart
        # 通过重启随机游走生成邻居集合
        node_n = [self.args.A_n, self.args.P_n, self.args.V_n]
        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_temp = self.a_p_list_train[j]
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
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
                    while neigh_L < 100:  # maximum neighbor size = 100
                        rand_p = random.random()  # return p
                        if rand_p > 0.5:
                            if curNode[0] == "a":
                                curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
                                if p_L < 46:  # size constraint (make sure each type of neighobr is sampled)
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
                                    neigh_L += 1
                                    p_L += 1
                        else:
                            if i == 0:
                                curNode = ('a' + str(j))
                            elif i == 1:
                                curNode = ('p' + str(j))
                            else:
                                curNode = ('v' + str(j))

        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    a_neigh_list_train[j] = list(a_neigh_list_train[j])
                elif i == 1:
                    p_neigh_list_train[j] = list(p_neigh_list_train[j])
                else:
                    v_neigh_list_train[j] = list(v_neigh_list_train[j])

        neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
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
        print(neigh_f)
        return neigh_f
if __name__ == '__main__':
     args= read_args()
     model_class(args=args)
