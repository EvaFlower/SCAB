import numpy as np
import torch
import random
import time
import torch.optim.lr_scheduler as lr_scheduler
from network.linear_base_EM_ae import EM_base_AE
from scipy.optimize import linear_sum_assignment
from torch.distributions import Categorical
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter
import os
from torchvision import transforms as T


class GMM_base_AE():
    def __init__(self, flag_cuda, num_cluster, learning_rate, dim, data_loader, \
        input_dim, batch_size, c_dim, save_path):
        super(GMM_base_AE, self).__init__()
        self.flag_cuda = flag_cuda
        self.num_cluster = num_cluster
        self.lr = learning_rate
        self.data_loader = data_loader
        self.dim = dim
        self.img_shape = None
        self.batch_size = batch_size
        self.use_tensorboard = True
        log_path = save_path+'log/'
        if self.use_tensorboard:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            else:
                pass
            self.tensorboard = SummaryWriter(log_path)
        if flag_cuda:
            self.EM_2head_AE = EM_base_AE(num_classes=self.num_cluster, dim=self.dim, img_dim=input_dim, c_dim=0).cuda()
        else:
            self.EM_2head_AE = EM_base_AE(num_classes=self.num_cluster, dim=self.dim, img_dim=input_dim, c_dim=0)
        self.loss_func1 = torch.nn.MSELoss()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.EM_2head_AE.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.finalmodel_path = save_path+"final_model.pt"
        self.final_loss1 = []
        self.final_loss2 = []
        self.final_acc = []
        self.final_nmi = []
        self.final_ari = []
        self.full_acc = []
        self.full_nmi = []
        self.full_ari = []
        self.gamma = []
        self.Embedding = []
        self.a = 5
        self.ID = torch.ones(self.num_cluster, 1)
        self.nmi = 0

    def Kmeans_cluster(self, embedding):
        kmeans = KMeans(n_clusters=self.num_cluster, n_init=20)
        y_pred = kmeans.fit_predict(embedding)
        return y_pred

    def loss_entropy(self, gamma):
        entropy = Categorical(probs=gamma).entropy().mean()
        return entropy

    def loss_func2(self, cluster, y_pred, z, alpha):

        ratio = self.ID * (y_pred.shape[0] / self.num_cluster + alpha)
        for i in range(self.num_cluster):
            ratio[i] /= np.sum(y_pred.cpu().data.numpy() == i) + alpha

        ratio = torch.unsqueeze(ratio[y_pred], 1).to(z.device)
        Dist = (z - cluster) ** 2
        loss = torch.mul(Dist, ratio).mean()
        return loss

    def train(self, epoch):
        self.EM_2head_AE.train()
        true_label = []
        pred_label = []
        Embedding = []
        t = 1 / (1 + np.exp(6 - epoch/90)) #-1e-2
        p = 1 / (1 + np.exp(-15 + epoch/10))
        """
        Train Epoch: 1000 (0%)  Loss1:0.020347  Loss2:0.072886  Loss4:21.101349 Gamma:0.962929
        Train_acc:0.7068; Train_nmi:0.7328; Train_ari:0.6454
        """
        ep1 = 1e-2  
        ep2 = 0 #5 * 1e-3 * max(t, 0)
        ep3 = self.a
        ep4 = 1e-4
        batch_size = self.batch_size
        train_data, train_targets = self.data_loader[0], self.data_loader[1]
        #train_data = train_data.transpose(0, 3, 1, 2)
        rand_idxes = np.random.choice(train_data.shape[0], train_data.shape[0], replace=False)
        train_data = train_data[rand_idxes]
        train_targets = train_targets[rand_idxes]
        for i in range(train_data.shape[0]//batch_size):
            self.EM_2head_AE.train()
            batch_idx = i
            batch_x = train_data[i*batch_size:(i+1)*batch_size] #/255.
            batch_x = torch.tensor(batch_x, dtype=torch.float)
            y_true = train_targets[i*batch_size:(i+1)*batch_size, 0] # 0:object label; 1:domain label
            y_true = torch.tensor(y_true, dtype=torch.long)
            if self.flag_cuda:
                data = batch_x.cuda()
                y_true = y_true.cuda()
            else:
                data = batch_x
            self.optimizer.zero_grad()
            z, quantized, gamma, recon_x, recon_center, mu, log_var = self.EM_2head_AE(data)
            max_gamma, y_pred = gamma.max(dim=1)
            loss1 = self.loss_func1(recon_x, data)
            #loss2 = self.loss_func2(quantized.detach(), y_pred.detach(), z, ep3)
            loss2 = self.loss(quantized.detach(), z)
            loss3 = self.loss_entropy(gamma)
            loss4 = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = loss1 + ep1 * loss2 + ep2 * loss3 + ep4*loss4
            loss.backward()
            self.optimizer.step()
            if self.use_tensorboard:
                self.tensorboard.add_scalar('loss/recon loss', loss1, i+epoch*(train_data.shape[0]//batch_size))
                self.tensorboard.add_scalar('loss/feat recon loss', loss4, i+epoch*(train_data.shape[0]//batch_size))
                self.tensorboard.add_scalar('loss/cluster loss', loss2, i+epoch*(train_data.shape[0]//batch_size))
                self.tensorboard.add_scalar('loss/gamma loss', loss3, i+epoch*(train_data.shape[0]//batch_size))
                self.tensorboard.add_scalar('gamma/gamma', torch.mean(max_gamma), i+epoch*\
                    (train_data.shape[0]//batch_size))

            if self.flag_cuda:
                y_true = y_true.cpu().data.numpy()
                true_label.append(y_true)
                y_pred = y_pred.cpu().data.numpy()
                pred_label.append(y_pred)
                Embedding.append(z.cpu().data.numpy())
            else:
                y_true = y_true.data.numpy()
                true_label.append(y_true)
                y_pred = y_pred.data.numpy()
                pred_label.append(y_pred)
                Embedding.append(z.data.numpy())

            if batch_idx % 100 == 0:
                print('Train Epoch: {} ({:.0f}%) \tLoss1:{:.6f}\tLoss2:{:.6f}\tLoss4:{:.6f}\tGamma:{:.6f}'\
                    .format(epoch, 100. * batch_idx / (train_data.shape[0]//batch_size), loss1.data, loss2.data, \
                        loss4.data, max_gamma.mean()))

        self.final_loss1.append(loss1.cpu().data.numpy())
        self.final_loss2.append(loss2.cpu().data.numpy())
        self.gamma.append(gamma.cpu().data.numpy())

        # if self.use_tensorboard:
        #     self.tensorboard.add_image('train/img||rec', tvutils.make_grid( \
        #         torch.cat([(data+1)/2, (recon_x+1)/2], dim=0), nrow=data.size(0)), epoch)
        #     self.tensorboard.add_image('train/centroid', tvutils.make_grid( \
        #         (recon_center+1)/2, nrow=recon_center.size(0)), epoch)

        if epoch % 10 == 0:
            
            self.eval(epoch)
               
        return epoch

    def eval(self, epoch):
        self.EM_2head_AE.eval()
        true_label = []
        pred_label = []
        Embedding = []

        batch_size = self.batch_size
        train_data, train_targets = self.data_loader[0], self.data_loader[1]
        for i in range(train_data.shape[0]//batch_size):
            batch_x = train_data[i*batch_size:(i+1)*batch_size]
            y_true = train_targets[i*batch_size:(i+1)*batch_size, 0] # object label
            batch_x = torch.tensor(batch_x, dtype=torch.float)
            y_true = torch.tensor(y_true, dtype=torch.long)
            if self.flag_cuda:
                data = batch_x.cuda()
                y_true = y_true.cuda()
            else:
                data = batch_x
            z, quantized, gamma, recon_x, recon_center, _, _ = self.EM_2head_AE(data)
            max_gamma, y_pred = gamma.max(dim=1)
            if self.flag_cuda:
                y_true = y_true.cpu().data.numpy()
                true_label.append(y_true)
                y_pred = y_pred.cpu().data.numpy()
                pred_label.append(y_pred)
                Embedding.append(z.cpu().data.numpy())
            else:
                y_true = y_true.data.numpy()
                true_label.append(y_true)
                y_pred = y_pred.data.numpy()
                pred_label.append(y_pred)
                Embedding.append(z.data.numpy())
            
        true_label = np.concatenate(true_label, axis=0)
        pred_label = np.concatenate(pred_label, axis=0)
        full_acc, full_nmi, full_ari = self.measure(true_label, pred_label)
        print('==========> Epoch: {} \tMinibatch: \tTrain_acc:{:.4f} \tTrain_nmi:{:.4f} \tTrain_ari:{:.4f}'.format(
            epoch, full_acc, full_nmi, full_ari))

        Embedding = np.concatenate(Embedding, axis=0)
        full_Pred_label = self.Kmeans_cluster(Embedding)
        Ftrain_acc, Ftrain_nmi, Ftrain_ari = self.measure(true_label, full_Pred_label)
        print('=========> Epoch: {} \tFullbatch: \tTrain_acc:{:.4f} \tTrain_nmi:{:.4f} \tTrain_ari:{:.4f}'.format(
            epoch, Ftrain_acc, Ftrain_nmi, Ftrain_ari))

        self.current_nmi = full_nmi
        
        if self.use_tensorboard:
            self.tensorboard.add_scalar('metric/our acc', full_acc, epoch)
            self.tensorboard.add_scalar('metric/our nmi', full_nmi, epoch)
            self.tensorboard.add_scalar('metric/our ari', full_ari, epoch)
            self.tensorboard.add_scalar('metric/kmeans acc', Ftrain_acc, epoch)
            self.tensorboard.add_scalar('metric/kmeans nmi', Ftrain_nmi, epoch)
            self.tensorboard.add_scalar('metric/kmeans ari', Ftrain_ari, epoch)


    def process(self, Iter=1001):
        Check_epoch = self.model_load(self.finalmodel_path)
        self.eval(Check_epoch)
        # self.optimizer = torch.optim.Adam(self.EM_2head_AE.parameters(), lr=self.lr, weight_decay=1e-4)
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # for epoch in range(Iter):
        #     self.train(epoch)
        #     if self.nmi < self.current_nmi: #epoch % 450 == 0:
        #         path = self.finalmodel_path #+ str(epoch) + '.pt'
        #         self.model_save(path, epoch)
        # self.model_save(self.finalmodel_path, Iter)
        # Check_epoch = self.model_load(self.finalmodel_path)
        # for epoch in range(Check_epoch, Iter):
        #     self.train(epoch)
        Embedding, label, recon_center, centroid, domain_labels = self.get_embedding()

        return Embedding, label, recon_center, centroid, domain_labels

    def model_save(self, fpath, epoch):
        state = {'epoch': epoch + 1,
                 "random_state": random.getstate(),
                 "np_random_state": np.random.get_state(),
                 "torch_random_state": torch.get_rng_state(),
                 'torch_cuda_random_state': torch.cuda.get_rng_state(),
                 'scheduler': self.scheduler.state_dict(),
                 'state_dict': self.EM_2head_AE.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, fpath)
        return state

    def model_load(self, fpath):
        checkpoint = torch.load(fpath)
        epoch = checkpoint['epoch']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.EM_2head_AE.load_state_dict(checkpoint['state_dict']) #, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return epoch

    def measure(self, y_true, y_pred):
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

    def get_embedding(self):
        self.EM_2head_AE.eval()
        hiden_embedding = []
        label = []

        batch_size = self.batch_size
        train_data, train_targets = self.data_loader[0], self.data_loader[1]
        #train_data = train_data.transpose(0, 3, 1, 2)
        for i in range(train_data.shape[0]//batch_size):
            batch_x = train_data[i*batch_size:(i+1)*batch_size]
            y_true = train_targets[i*batch_size:(i+1)*batch_size, 0] # object label
            batch_x = torch.tensor(batch_x, dtype=torch.float)
            target = torch.tensor(y_true, dtype=torch.long)
            if self.flag_cuda:
                data = batch_x.cuda()
                target = target.cuda()
            else:
                data = batch_x

            z, quantized, gamma, recon_x, recon_center, _, _ = self.EM_2head_AE(data)
            if self.flag_cuda:
                z = z.cpu().data.numpy()
                hiden_embedding.append(z)
                target = target.cpu().data.numpy()
                label.append(target)
            else:
                z = z.data.numpy()
                hiden_embedding.append(z)
                target = gamma.data.numpy()
                label.append(target)

        if self.flag_cuda:
            recon_center = recon_center.cpu().data.numpy()
            centroid = self.EM_2head_AE.Clustering.centroids.cpu().weight.data.numpy()
        else:
            recon_center = recon_center.data.numpy()
            centroid = self.EM_2head_AE.Clustering.centroids.weight.data.numpy()
        feature = np.concatenate(hiden_embedding, axis=0)
        number_label = np.concatenate(label, axis=0)
        domain_labels = train_targets[:feature.shape[0], 1]
        return feature, number_label, recon_center, centroid, domain_labels
