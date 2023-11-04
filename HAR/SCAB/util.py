"""
https://github.com/dcmoyer/inv-rep/blob/master/src/kl_tools.py
https://github.com/dcmoyer/invariance-tutorial/blob/master/src/kl_tools.py
https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
"""

import torch
from torch.distributions import Categorical

import math
import numpy as np
import logging 
import traceback

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from sklearn.mixture import GaussianMixture as GMM

import logging as logger

#KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) + 
#  (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
#  \log( \frac{\det \sigma_1}{\det \sigma_0} )
def all_pairs_gaussian_kl(mu, sigma, dim_z, add_third_term=False): # mu is [batchsize x dim_z], sigma is [batchsize x dim_z]
    sigma_sq = sigma**2 + 1e-8
    sigma_sq_inv = torch.reciprocal(sigma_sq) # sigma_inv is [batchsize x sizeof(latent_space)]
    
    first_term = torch.matmul(sigma_sq, torch.t(sigma_sq_inv))
    r = torch.matmul(mu * mu, torch.t(sigma_sq_inv)) # r is now [batchsize x batchsize] = sum(mu[:,i]**2 / Sigma[j])
    r2 = mu * mu * sigma_sq_inv 
    r2 = torch.sum(r2,1) # r2 is now [batchsize, 1] = mu[j]**2 / Sigma[j]

    #squared distance
    #(mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
    #uses broadcasting
    second_term = 2*torch.matmul(mu, torch.t(mu*sigma_sq_inv))
    second_term = r - second_term + torch.t(r2)

    # log det A = tr log A
    # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
    #   \tr\log \Sigma_1 - \tr\log \Sigma_0 
    # for each sample, we have B comparisons to B other samples...
    #   so this cancels out
    if(add_third_term):
        r = torch.sum(torch.log(sigma_sq),1)
        r = r.view(-1, 1)
        third_term = r - torch.t(r)
    else:
        third_term = 0

    return 0.5 * ( first_term + second_term + third_term - dim_z )


# kl_conditional_and_marg
#   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
def kl_conditional_and_marg(z_mean, z_log_sigma_sq, dim_z, gauss_ent=False):
    z_sigma = torch.exp(0.5*z_log_sigma_sq)
    if gauss_ent:
        tmp = 2*math.pi*math.e
        gaussian_ent = 0.5 * torch.sum(dim_z*torch.log(torch.tensor(tmp)) \
            + torch.log(torch.square(z_sigma) + 1e-8),1)
    #  0.5 * tf.log(\
    #    tf.reduce_prod(2 * math.pi * math.e * tf.square(sigma) + 1e-8,0)\
    #  )
        B = torch.tensor(z_mean.size(0)).float() 
        all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, dim_z, True)
        return (1.0 / B) * torch.mean(\
            (torch.sum(all_pairs_GKL, 1) + (B - 1) * torch.t(gaussian_ent)) - torch.log( B )\
        ) #- log(B)
        #(tf.reduce_sum(all_pairs_GKL,1) + (B - 1)*gaussian_ent)\
    else:
        all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, dim_z, True)
        return torch.mean(all_pairs_GKL)


def one_hot(labels, NUM_LABELS=10):
    num_labels_data = labels.size(0)
    one_hot_encoding = torch.zeros((num_labels_data, NUM_LABELS), dtype=torch.float)
    one_hot_encoding[torch.arange(0, num_labels_data), labels] = 1
    one_hot_encoding = one_hot_encoding.view(-1, NUM_LABELS)
    return one_hot_encoding

def to_img(x, img_shape=(64, 64)):
    x = x.view(x.size(0), 1, img_shape[1], img_shape[0])
    return x

def check_tensor(vector, name=None):
    logger = logging.getLogger(__name__)
    try:
        if isinstance(vector, torch.Tensor):
            if not np.any(np.isnan(vector.cpu().detach().numpy())):
                return True
            print("[{}] is false".format(name))
            return False
        elif isinstance(vector, np.ndarray):
            if not np.any(np.isnan(vector)):
                return True
            print("[{}] is false".format(name))
            return False
        elif isinstance(vector, list):
            vector = np.asarray(vector, dtype=np.float32)
            if not np.any(np.isnan(vector)):
                return True
            print("[{}] is false".format(name))
            return False
    except Exception as ex:
        logger.error(traceback.format_exc())
        logger.error("name is [{}] value is {}".format(name, vector))
        return False

def negative_sampling(img, label, dataset):
    neg_data = []
    neg_label = []
    for i in range(img.size(0)):
        rand_idx = np.random.choice(torch.sum(dataset.targets==label[i].cpu()).data.numpy(), 1)
        neg_data.append(dataset.data[dataset.targets==label[i].cpu()][rand_idx])
        neg_label.append(label[i])
    neg_data = torch.cat(neg_data, dim=0)
    neg_label = torch.stack(neg_label)
    return neg_data, neg_label

def loss_entropy(gamma):
    entropy = Categorical(probs=gamma).entropy().mean()
    return entropy

def cal_cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    NMI_fun = normalized_mutual_info_score
    ARI_fun = adjusted_rand_score

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    Acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = NMI_fun(y_true, y_pred)
    ari = ARI_fun(y_true, y_pred)
    return Acc, nmi, ari

def Kmeans_cluster(embedding, num_cluster=5):
    kmeans = KMeans(n_clusters=num_cluster, n_init=20)
    y_pred = kmeans.fit_predict(embedding)
    return y_pred

def GMM_cluster(embedding, num_cluster=5):
    gmm = GMM(n_components=num_cluster, covariance_type='full').fit(embedding)
    y_pred = gmm.predict(embedding)
    return y_pred
