from re import L
import seaborn as sns
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.utils.data as data_utils
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def cluster_measure(y_true, y_pred):
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

def get_YTF_dataset():
    path = './YTF/data4torch.h5'
    data = h5py.File(path, 'r')['data'][:]
    label = h5py.File(path, 'r')['labels'][:]
    data = (data - data.mean()) / data.std()
    data = torch.from_numpy(data).float()
    label = (torch.from_numpy(label) - 1).int()
    traindata = data_utils.TensorDataset(data, label)
    # a = torch.zeros([label.max()+1, 1])
    # for i in range(label.max()+1):
    #     a[i] = torch.sum(label==i).item()/label.shape[0]
    return traindata

def get_data_loader(dataset, batch_size=256, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def scatter(x, mu, colors, ID):
    n_color = colors.max() + 1
    palette = np.array(sns.color_palette("hls", n_color))

    f = plt.figure()
    ax = plt.subplot(aspect='equal')
    for i in range(n_color):
        positions = np.where(colors == i)
        ax.scatter(x[positions[0], 0], x[positions[0], 1], lw=0, s=8, alpha=0.3,
                   c=palette[colors[positions[0]].astype(np.int)], label='{}'.format(i))
    for i in range(mu.shape[0]):
        ax.scatter(mu[i, 0], mu[i, 1], lw=0, s=200, alpha=0.5,
                   c='grey', label='{}'.format(i))
    ax.axis('off')
    ax.axis('tight')
    #plt.legend()

    txts = []
    for i in range(n_color):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=12, alpha=0.3)
        #txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)
    if ID == 1:
        plt.savefig('embeddings_11.png')
    elif ID == 2:
        plt.savefig('embeddings_12.png')
    elif ID == 3:
        plt.savefig('embeddings_13.png')
    else:
        plt.savefig(ID)

    #plt.show()
    plt.close(f)
    return f


def tsne_figure(feature, mu, number_label, ID):
    all_data = np.concatenate([feature, mu], axis=0)
    proj_feat = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=9159).fit_transform(all_data)
    feature = proj_feat[0:feature.shape[0],:]
    mu = proj_feat[feature.shape[0]: feature.shape[0]+mu.shape[0],:]
    scatter(feature, mu, number_label, ID)

def center_figure(centroids, ID, img_shape):
    plt.gray()
    fig = plt.figure(figsize=(25,25))
    norm = plt.Normalize(centroids.min(), centroids.max())
    for i in range(0, centroids.shape[0]):
        #ax = fig.add_subplot(2, 5, i+1, title = "Centroid for Digit:{}".format(str( centroids.label[i] )))
        ax = fig.add_subplot(6, 6, i+1)
        #ax.matshow( centroids[i,].reshape((55,55)).astype(float))

        ax.imshow(norm(centroids[i,].transpose((1, 2, 0)).astype(float)))
    if ID == 1:
        plt.savefig('./Results/ours/centroids_2.png')
        cv2.imwrite('./Results/ours/centroid_2.png', norm(centroids[i,].transpose((1, 2, 0)))*255)
    #plt.show()
    plt.close(fig)

def Figure_Plot(Embedding, label, centroids, ID, img_shape, mu):
    tsne_figure(Embedding, mu, label, ID)
    if centroids is not None:
        center_figure(centroids, ID, img_shape)


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    Args:
        y_true (list): list of true cluster numbers, an integer array 0-indexed
        y_predicted (list): list of predicted cluster numbers, an integer array 0-indexed
        cluster_number (int): number of clusters, if None then calculated from entropy_input
    Returns:
        reassignment dictionary, clustering accuracy
    """

    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size

    return reassignment, accuracy


def entropy(entropy_input):
    '''

    Args:
        entropy_input (Tensor):
    Returns:
        entropy (float)
    '''
    epsilon = 1e-5  # for numerical stability
    entropy = -entropy_input * np.log(entropy_input + epsilon)
    entropy = np.sum(entropy, axis=0)
    return entropy

    
def balance(cluster_labels, group_labels):
    """

    Args:
        predicted:
        size_0:
        k:

    Returns:

    """
    print(cluster_labels.shape, group_labels.shape)
    k = max(cluster_labels)+1
    g = max(group_labels)+1
    print(k, g)
    count = np.zeros((k, g), dtype=np.float)
    for i in range(g):
        for j in range(k):
            count[j, i] = np.sum((cluster_labels==j) & (group_labels==i))
    # for i in range(size_0):
    #     count[cluster_labels[i], 0] += 1
    # for i in range(size_0, cluster_labels.shape[0]):
    #     count[cluster_labels[i], 1] += 1
    count[count == 0] = 1e-5
    print(count)
    print(max(count[:, 0]/count[:, 1]), min(count[:, 1]/count[:, 0]))

    min_balance = float('inf')

    for i in range(g):
        for j in range(i+1, g):
            b = min(count[:, i]/count[:, j])
            if min_balance > b:
                min_balance = b
            b = min(count[:, j]/count[:, i])
            if min_balance > b:
                min_balance = b

    # balance_0 = torch.min(count[:, 0] / count[:, 1])
    # balance_1 = torch.min(count[:, 1] / count[:, 0])

    ens =[0 for i in range(g)]
    for i in range(g):
        ens[i] = entropy(count[:, i]/np.sum(count[:, i]))
    # en_0 = entropy(count[:, 0] / torch.sum(count[:, 0]))
    # en_1 = entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min_balance, ens


def entropy2(entropy_input):
    '''

    Args:
        entropy_input (Tensor):
    Returns:
        entropy (float)
    '''
    epsilon = 1e-5  # for numerical stability
    entropy = -entropy_input * torch.log(entropy_input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def balance2(cluster_labels, group_labels): #predicted, size_0, k=10):
    """

    Args:
        predicted:
        size_0:
        k:

    Returns:

    """
    predicted = torch.tensor(cluster_labels)
    k = max(predicted)+1
    size_0 = np.sum(group_labels==0)
    print(k, size_0)
    count = torch.zeros((k, 2), dtype=float)
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5
    
    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])

    en_0 = entropy2(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = entropy2(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()
