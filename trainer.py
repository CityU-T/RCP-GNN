import torch.nn.functional as F
import torch
import numpy as np

# Reference code: https://github.com/snap-stanford/conformalized-gnn

def train(epoch, model, data, optimizer, alpha):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data, alpha, tau, target_size, size_loss = False):
    model.eval()
    if size_loss:
        pred_raw, ori_pred_raw = model(data.x, data.edge_index)
    else:
        pred_raw = model(data.x, data.edge_index)
        
    pred = pred_raw.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.valid_mask, data.calib_test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
  
    if size_loss:
        out_softmax = F.softmax(pred_raw, dim = 1)
        query_idx = np.where(data.valid_mask)[0]
        np.random.seed(0)
        np.random.shuffle(query_idx)

        train_train_idx = query_idx[:int(len(query_idx)/2)]
        train_calib_idx = query_idx[int(len(query_idx)/2):]

        n_temp = len(train_calib_idx)
        q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp

        tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
        qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')
        c = torch.sigmoid((out_softmax[train_train_idx] - qhat)/tau)
        size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))
        
        return accs, pred_raw, size_loss.item()
    else:
        return accs, pred_raw

