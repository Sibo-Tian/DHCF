import numpy as np
import scipy.sparse as sp
import torch

def onehot_encoder(idx, num_classes):
    identity = np.identity(num_classes)
    return identity[idx]

def load_observe(path='./data/ml-100k/', dataset='u.data'):
    raw_observe = np.genfromtxt("{}{}".format(path, dataset), dtype=np.int32)
    row_idx = raw_observe[:,0]
    col_idx = raw_observe[:,1]
    H = np.zeros((943, 1682))
    H[row_idx-1, col_idx-1] = 1
    H = torch.tensor(H)
    return H

def load_data(path='./data/ml-100k/', dataset='u.data'):
    """Load movielens dataset"""
    print('Loading {} dataset...'.format(dataset))
    #oberve(Hyper_graph)
    H_train = load_observe(path, 'u1.base')
    H_test = load_observe(path, 'u2.base')

    #user_features
    raw_occupation = np.genfromtxt("{}{}".format(path,'u.occupation'), dtype=np.dtype(str))
    occupation_num = len(raw_occupation)
    occupation_idx = {raw_occupation[i] : i for i in range(len(raw_occupation))}
    gender_idx = {'M': 0, 'F': 1}
    
    user_features = np.zeros((943, 23))
    with open('./data/ml-100k/u.user') as fp:
        for i, line in enumerate(fp):
            info = line.split('|')
            user_features[i][0] = int(info[1]) #age
            user_features[i][1] = gender_idx[info[2]] #gender
            user_features[i][2:] = onehot_encoder(occupation_idx[info[3]], occupation_num) #occupation
    user_features = torch.FloatTensor(user_features)

    #item_features
    item_features = np.zeros((1682, 19))
    with open('./data/ml-100k/u.item') as fp:
        for i, line in enumerate(fp):
            info = line.split('|')
            info = info[5:]
            info = list(map(int, info))
            item_features[i,:] = info[:]
    item_features = torch.FloatTensor(item_features)
    return H_train, H_test, user_features, item_features


def BPR_Loss(H, U, I):
    """My BPR loss(slightly different from the paper)"""
    preference = torch.matmul(U, I.t())
    #positive_sample
    D_v = H.sum(1)
    mask = (D_v >= 0.5).to(torch.int)
    positive_score = torch.sum(preference * H, 1)
    positive_score = torch.where(mask==1, positive_score/D_v, positive_score)
    #negative_sample
    D_v = H.shape[1] - H.sum(1)
    mask = (D_v >= 0.5).to(torch.int)
    negative_score = torch.sum(preference * (1 - H), 1)
    negative_score = torch.where(mask==1, negative_score/D_v, negative_score)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(positive_score-negative_score, torch.ones_like(positive_score))
    return loss