from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

DIM_C = 30 # just as an example. Sometimes we also have another y
num_rot_class = 6
test_label = []
for i in range(DIM_C):
    tmp = (torch.ones(num_rot_class)*i).long()
    tmp = util.one_hot(tmp, NUM_LABELS=DIM_C).cuda()
    test_label.append(tmp)
test_label = torch.cat(test_label, dim=0)

class EM_base_AE(nn.Module):
    def __init__(self, num_classes, dim, img_dim, c_dim=30):
        super(EM_base_AE, self).__init__()
        # self.encoder = nn.Sequential(
        #     #nn.Dropout(0.25),
        #     nn.Linear(3*img_shape[0]*img_shape[1], 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, dim),
        #     #nn.Dropout(0.25)
        # )
        self.encoder = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(img_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            #nn.Linear(2000, dim)
        )
        self.enc_mu = nn.Sequential(
            nn.Linear(2000, dim)
        )
        self.enc_logvar = nn.Sequential(
            nn.Linear(2000, dim)
        )
            # nn.ReLU(),
            # nn.Linear(500, 2000),
            # nn.ReLU(),
        # self.decoder = nn.Sequential(
        #     nn.Linear(dim, 1024),
        #     nn.ReLU(),
        #     # nn.Linear(2000, 500),
        #     # nn.ReLU(),
        #     # nn.Linear(500, 500),
        #     # nn.ReLU(),
        #     nn.Linear(1024, 3*img_shape[0]*img_shape[1]),
        #     nn.Sigmoid()
        # )
        self.decoder = nn.Sequential(
            nn.Linear(dim+c_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, img_dim),
            #nn.Sigmoid()
        )
        # self.encoder = nn.Sequential(
        #     nn.Dropout(0.25),
        #     nn.Linear(784, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 2000),
        #     nn.ReLU(),
        #     nn.Linear(2000, dim),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(dim, 2000),
        #     nn.ReLU(),
        #     nn.Linear(2000, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 784),
        #     nn.Sigmoid()
        # )
        self.num_classes = num_classes
        self.dim = dim
        self.Clustering = VectorQuantizerEMA(self.num_classes, self.dim)
        self.fc0 = nn.Linear(2 * self.dim, self.dim)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def Encode(self, x):
        x = self.encoder(x)
        z_mu, z_logvar = self.enc_mu(x), self.enc_logvar(x)
        return z_mu, z_logvar

    def reparametrize(self, mu, logvar):
        # eps = 0.1 * mu.data.new(mu.size()).normal_()
        # return eps.add(mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def Decode(self, z):
        x = self.decoder(z)
        #x = x.view(x.size(0), 3, self.img_shape[0], self.img_shape[1])
        return x
    
    def set_centroids(self, centroids, bs):
        self.Clustering.set_centroids(centroids, bs)

    def forward(self, x, y=None):
        x = x.contiguous().view(x.size(0), -1)
        z_mu, z_logvar = self.Encode(x)
        z = self.reparametrize(z_mu, z_logvar)
        quantized, gamma, centroid = self.Clustering(z)

        ## regular training
        double_z = torch.cat((z, quantized), 1)
        recon_z = self.fc0(double_z)
        if y is not None:
            recon_z = torch.cat([recon_z, y], dim=1)
        #recon_z = self.reparametrize(recon_z)
        recon_x = self.Decode(recon_z)

        ## Centroids reconstruction
        double_center = torch.cat((centroid, centroid), 1)
        fuse_center = self.fc0(double_center)
        if y is not None:
            fuse_center = fuse_center.repeat((DIM_C, 1))
            fuse_center = torch.cat([fuse_center, test_label], 1)
        recon_center = self.Decode(fuse_center)
        return z, quantized, gamma, recon_x, recon_center, z_mu, z_logvar

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, delta=-5, decay=0.95, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.centroids = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.centroids.weight.data = torch.nn.init.xavier_uniform_(torch.empty(self._num_embeddings, self._embedding_dim))

        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self._num_embeddings, self._embedding_dim)))

        self.delta = delta
        self._decay = decay
        self._epsilon = epsilon

    def set_centroids(self, centroids, bs):
        self._ema_w = nn.Parameter(centroids)
        cluster_size = bs/self._num_embeddings
        cluster_size = int(cluster_size)
        self._ema_cluster_size = torch.ones(self._num_embeddings, dtype=int)*cluster_size
        
    def forward(self, z):

        # Calculate distances
        Dist = torch.unsqueeze(z, 1) - self.centroids.weight
        z_dist = torch.sum(torch.mul(Dist, Dist), 2)
        gamma = F.softmax(self.delta * z_dist, dim=1)

        # Encoding
        encoding_Idx = torch.argmax(gamma, dim=1).unsqueeze(1)
        Idx = torch.zeros(z.shape[0], self._num_embeddings, device=z.device)
        Idx.scatter_(1, encoding_Idx, 1)

        # Quantize and reconstruction
        recons = torch.matmul(Idx, self.centroids.weight)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(Idx, 0)
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(Idx.t(), z)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.centroids.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Straight Through Estimator
        recons = z + (recons - z).detach()
        cluster_center = self.centroids.weight.data

        return recons, gamma, cluster_center
