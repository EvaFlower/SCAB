import numpy as np
import random
import torch
from External import Figure_Plot, cluster_measure
from datasets import read_har_dataset
from sklearn.cluster import KMeans

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
flag_cuda = torch.cuda.is_available()

SEED = 2333 #9159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False


num_cluster = 6
learning_rate = 5e-4
dim = 10
iter = 1001
input_dim = 561  # input dimension
c_dim = 30  # categories of confounding factor

dataloader = read_har_dataset.load_data()
kmeans = KMeans(n_clusters=num_cluster, n_init=40, random_state=0, max_iter=5000)
predicted = kmeans.fit_predict(dataloader[0])  # predict centers from features.
acc, nmi, ari = cluster_measure(dataloader[1][:, 0], predicted)
print(acc, nmi, ari)

batch_size = 1024

from main_base_EM_ae_inv_rep import GMM_base_AE
run = 1
save_path = './Results/ae_inv_cluster/{}/'.format(run)
Embedding, label, recon_center, centroid, domain_labels = GMM_base_AE(flag_cuda, num_cluster, learning_rate, \
    dim, dataloader, input_dim, batch_size, c_dim, save_path=save_path).process(iter)
file_name = save_path+'embedding.png'
Figure_Plot(Embedding, label, None, file_name, None, centroid)
