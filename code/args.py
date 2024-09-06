import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type = str, default = 'D:/codee/data/academic/',
				   help='path to data')
	parser.add_argument('--data_path1', type=str, default='D:/codee/data/abide1/',
						help='path to data')
	parser.add_argument('--model_path', type = str, default = 'D:/codee/model_save/',
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
	parser.add_argument('--lr', type = int, default = 0.001, # 0.001
				   help = 'learning rate')
	parser.add_argument('--batch_s', type = int, default = 2000, # 5000
				   help = 'batch size')
	parser.add_argument('--mini_batch_s', type = int, default = 200,
				   help = 'mini batch size')
	parser.add_argument('--train_iter_n', type = int, default = 50,
				   help = 'max number of training iteration')
	parser.add_argument('--walk_n', type = int, default = 10, # 10
				   help='number of walk per root node')
	parser.add_argument('--walk_L', type = int, default = 30, # 30
				   help='length of each walk')
	parser.add_argument('--window', type = int, default = 5,
				   help='window size for relation extration')
	parser.add_argument("--random_seed", default = 10, type = int)
	parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
	parser.add_argument('--save_model_freq', type = float, default = 1,
				   help = 'number of iterations to save model')
	parser.add_argument("--cuda", default = 0, type = int)
	parser.add_argument("--checkpoint", default = '', type=str)
	parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
																	  'construction 用于网络构建的连接类型(default: correlation, '
																	  'options: correlation, partial correlation, '
																	  'tangent)')
	parser.add_argument('--atlas', default='ho',
						help='atlas for network construction (node definition) 网络构建图集(节点定义) (default: ho, '
							 'see preprocessed-connectomes-project.org/abide/Pipelines.html '
							 'for more options )')
	parser.add_argument('--num_training', default=1, type=float, help='Percentage of training set used for '
																	  'training (default: 1.0)')  # 0.9
	parser.add_argument('--folds', default=0, type=int, help='For cross validation, specifies which fold will be '
															 'used. All folds are used if set to 11 (default: 11)')
	parser.add_argument('--num_features', default=1950, type=int, help='Number of features to keep for '
																	   'the feature selection step  为特征选择步骤保留的特征数量(default: 2000)')  # 1950


	parser.add_argument('--embed_d', type=int, default=1950,
						help='embedding dimension')
	args = parser.parse_args()
	# params = dict()
	params = dict()
	params['num_features'] = args.num_features
	params['num_training'] = args.num_training

	return args
