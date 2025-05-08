import torch
import torch.nn as nn
import torch.nn.functional as F

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat,bias=False)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return layers

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(KANLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_dim, in_dim))
        self.biases = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = torch.sin(torch.matmul(x, self.weights.T) + self.biases)  # KAN transformation
        return x

class Generator(nn.Module):
    def __init__(self, bow_dim, hid_dim, n_topic):
        super(Generator, self).__init__()
        self.g = nn.Sequential(
            KANLayer(n_topic, hid_dim),
            KANLayer(hid_dim, bow_dim),
            nn.Softmax(dim=1)
        )

    def inference(self, theta):
        return self.g(theta)

    def forward(self, theta):
        bow_f = self.g(theta)
        doc_f = torch.cat([theta, bow_f], dim=1)
        return doc_f

class Encoder(nn.Module):
    def __init__(self, bow_dim, hid_dim, n_topic):
        super(Encoder, self).__init__()
        self.e = nn.Sequential(
            KANLayer(bow_dim, hid_dim),
            KANLayer(hid_dim, n_topic),
            nn.Softmax(dim=1)
        )

    def forward(self, bow):
        theta = self.e(bow)
        doc_r = torch.cat([theta, bow], dim=1)
        return doc_r

    def return_theta(self, bow):
        return self.e(bow)

class Discriminator(nn.Module):
    def __init__(self, bow_dim, hid_dim, n_topic):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            KANLayer(n_topic + bow_dim, hid_dim),
            KANLayer(hid_dim, 1)
        )

    def forward(self, reps):
        score = self.d(reps)
        return score