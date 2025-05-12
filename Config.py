# train
epochs = 200
batch_size = 48
device = 'cuda:0'  # cpu or 'cuda:0'
id_dim = 16

labels_path = "/TCdata/NameTime_Idx_BST.pkl"
'''newk8_norm_values_split      newk8_norm_values'''
k8_sta_path = "/opt/data/private/norm_data_npy/Full2015_2023_Dataets/newk8_norm_values_split.pkl"
p12hpr_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/pre12h_labels.pkl'

train_k8_path = '/TCdata/k89_4ch1/train/'
valid_k8_path = '/TCdata/k89_4ch1/valid/'
predict_k8_path = '/TCdata/k89_4ch1/test/'

model_output_dir = '/opt/data/private/model/IDPil_NIPS/TCMM_CNF/'
predict_model = '/opt/data/private/model/IDPil_NIPS/TCMM_CNF/epoch_200.pth'
save_fig_dir = '/opt/data/private/model/IDPil_NIPS/TCMM_CNF/exp_img/'

model_output_dir_dkgi = '/opt/data/private/model/IDPil_NIPS/TCMM_dkgi/'
predict_model_dkgi = '/opt/data/private/model/IDPil_NIPS/TCMM_dkgi/epoch_200.pth'
save_fig_dir_dkgi = '/opt/data/private/model/IDPil_NIPS/TCMM_dkgi/exp_img/'

num_workers = 4  # 加载数据集线程并发数
best_loss = 0.005  # 当loss小于等于该值会保存模型
save_model_iter = 20  # 每多少次保存一份模型

data_format = 'npy'