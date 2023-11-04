import os
import numpy as np
from scipy import io


def load_data():
    root = 'datasets/'
    root = os.path.expanduser(root)
    har = io.loadmat(root+'/har/raw/HAR.mat')
    train_data = np.array(har['X_train'])
    test_data = np.array(har['X_test'])
    train_y = np.squeeze(np.array(har['y_train']))
    test_y = np.squeeze(np.array(har['y_test']))
    train_attrs = np.array(har['subject_train'])
    test_attrs = np.array(har['subject_test'])
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0)
    all_c = np.concatenate([train_attrs, test_attrs], axis=0)
    all_c = all_c-1
    for i in range(train_y.max()+1):
        print(np.sum(train_y==i))
    print(all_data.shape, all_y.shape, all_c.shape)
    print('data range: ', np.min(all_data), np.max(all_data))
    print('cluster label range: ', np.min(all_y), np.max(all_y))
    print('confounding factor range: ', np.min(all_c), np.max(all_c))
    all_yc = np.concatenate([all_y.reshape(-1, 1), all_c.reshape(-1, 1)], axis=1)
    return [all_data, all_yc]
    
