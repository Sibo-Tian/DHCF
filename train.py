from utils import load_data, BPR_Loss
from models import DHCF
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of batch_size to train.')

parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')

parser.add_argument('--num_layer', type=int, default=1,
                    help='Number of layers.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--l2', type=float, default=1e-2,
                    help='L2 loss on parameters.')

parser.add_argument('--embedding_dim', type=int, default=64,
                    help='Embedding_dim')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
H_train, H_test, U, I = load_data()

# Model and optimizer
model = DHCF(U.shape[1], I.shape[1], args.dropout, args.embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    H_train = H_train.cuda()
    H_test = H_test.cuda()
    U = U.cuda()
    I = I.cuda()

def train(H, U, I):
    model.train()
    loss_fn = BPR_Loss
    optimizer.zero_grad()

    U_out, I_out = model(H, U, I)
    loss = loss_fn(H, U_out, I_out)
    for param in model.parameters():
        loss += args.l2 * torch.sum(torch.square(param))

    loss.backward()
    optimizer.step()
    return loss.item()

def test(H, U, I):
    model.eval()
    loss_fn = BPR_Loss
    U_out, I_out = model(H, U, I)
    loss = loss_fn(H, U_out, I_out)
    return loss.item()

# Train model
best_model_loss = 1e9
t_total = time.time()
for epoch in range(args.epochs):
    loss = train(H_train, U, I)
    print('Epoch {}, loss: {:.4f}'.format(epoch, loss))
    if epoch % 5 == 0:
        loss = test(H_test, U, I)
        print('TEST -- Epoch {}, loss: {:.4f}'.format(epoch, loss))
        if loss < best_model_loss:
            best_model_loss = loss
            torch.save(model, 'best_model_{}.pt'.format(epoch))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Test model
loss = test(H_test, U, I)
print('Final TEST, loss: {:.4f}'.format(loss))

