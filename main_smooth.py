import argparse
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
import wandb
from torch_geometric.datasets import Amazon, Coauthor, CitationFull
from torchcp.classification.scores import APS, THRRANK
from torchcp.classification.predictors import SplitPredictor_GNN
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from model.model import GNN, ConfGNN
from trainer import train, test
import copy
from torch_geometric.logging import log
import torch.nn.functional as F
from utils import conf_pred

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora_ML_CF', choices = ['Cora_ML_CF', 'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF', 'Amazon-Computers', 'Amazon-Photo', 'Coauthor-CS', 'Coauthor-Physics', 'Anaheim', 'ChicagoSketch', 'county_education_2012', 'county_election_2016', 'county_income_2012', 'county_unemployment_2012', 'twitch_PTBR'])
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.1)   
parser.add_argument('--target_size', type=int, default=0)
parser.add_argument('--conf_correct_model', type=str, default='gnn', choices = ['gnn', 'mlp', 'Calibrate', 'mcdropout', 'mcdropout_std', 'QR'])
parser.add_argument('--task', type=str, default='classification')
parser.add_argument('--verbose', action='store_true', default = False)   # False no log
parser.add_argument('--not_save_res', action='store_true', default = False)
parser.add_argument('--conformal_score', type=str, default='aps', choices = ['aps', 'thrrank'])
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--retrain', action='store_true', default = False)

# for score
parser.add_argument('--interpolation', type=str, default='higher', choices = ['higher', 'linear'])

# for base model
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.01)
# for conf model
parser.add_argument('--confgnn_num_layers', type=int, default=1)    
parser.add_argument('--confgnn_base_model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])  
parser.add_argument('--confgnn_lr', type=float, default=1e-3)   
parser.add_argument('--conftr_calib_holdout', action='store_true', default = False)
parser.add_argument('--conftr', action='store_true', default = False)
parser.add_argument('--calib_fraction', type=float, default=0.5)
parser.add_argument('--adp', type=float, default=1.0)   #
parser.add_argument('--size_loss_weight', type=float, default=1)    
parser.add_argument('--conf_epochs', type=int, default=5000)
args = parser.parse_args()

# read params.
with open('./params/optimal_param_set.pkl', 'rb') as f:
    optimal_set = pickle.load(f)
optimal_parameter = optimal_set[args.model][args.dataset]

d = vars(args)   
for i, j in optimal_parameter.items():
    d[i] = j

if args.dataset in ['Cora_CF', 'Cora_ML_CF']:
    args.tau = 1
    args.size_loss_weight = 1
    args.target_size = 0
elif args.dataset in ['Amazon-Computers', 'Amazon-Photo',  'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF']:
    args.tau = 0.01
elif args.dataset in ['Coauthor-CS']:
    args.tau = 1


def main(args):

    # Preparing a calibration data and a test data.
    if args.dataset in ['Cora_CF', 'Cora_ML_CF', 'CiteSeer_CF', 'DBLP_CF', 'PubMed_CF']:
        path = osp.join('data', 'CitationFull')
        dataset = CitationFull(path, args.dataset[:-3], transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ['Amazon-Computers', 'Amazon-Photo']:
        path = osp.join('data', 'Amazon')
        dataset = Amazon(path, args.dataset.split('-')[1], transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ['Coauthor-CS', 'Coauthor-Physics']:
        path = osp.join('data', 'coauthor')
        dataset = Coauthor(path, args.dataset.split('-')[1], transform=T.NormalizeFeatures())
        data = dataset[0]

    y = data.y.detach().cpu().numpy()
    idx = np.array(range(len(y)))
    np.random.seed(args.data_seed)
    np.random.shuffle(idx)
    split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx)), len(idx)])
    train_idx, valid, calib_test = split_res[0], split_res[1], split_res[2]

    data.train_mask = np.array([False] * len(y))
    data.train_mask[train_idx] = True

    data.valid_mask = np.array([False] * len(y))
    data.valid_mask[valid] = True

    data.calib_test_mask = np.array([False] * len(y))
    data.calib_test_mask[calib_test] = True

    print(args)

    # average GNN runs
    tau2res = {}
    for run in tqdm(range(args.num_runs)):
        result_this_run = {}

        model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1)+ '_' + args.conformal_score + '.pt'

        # Preparing a pytorch model
        # 1. base model
        if (os.path.exists(model_checkpoint)) and (not args.retrain):
            print('loading saved base model...')
            num_features = dataset.num_features
            output_dim = dataset.num_classes
            model = torch.load(model_checkpoint, map_location = args.device)
            model, data = model.to(args.device), data.to(args.device)
            model.eval()
            pred = model(data.x, data.edge_index)
            best_model = model
            best_pred = pred

                # 1
            # Options of score function: THR, APS, SAPS, RAPS
            # Define a conformal prediction algorithm. Optional: SplitPredictor, ClusteredPredictor, ClassWisePredictor
            if args.conformal_score == 'aps':
                predictor = SplitPredictor_GNN(score_function=APS(), model=model)   # _device = model._device
            elif args.conformal_score == 'thrrank':
                predictor = SplitPredictor_GNN(score_function=THRRANK(), model=model)

        else:
            print('training base model from scratch...')
            num_features = dataset.num_features
            output_dim = dataset.num_classes
            model = GNN(num_features, args.hidden_channels, output_dim, args.model, args.heads, args.aggr)
            model, data = model.to(args.device), data.to(args.device)
            optimizer = torch.optim.Adam([
                        dict(params=model.conv1.parameters(), weight_decay=5e-4),
                        dict(params=model.conv2.parameters(), weight_decay=0)
                    ], lr=args.lr)  # Only perform weight-decay on first convolution.
        
            # STAGE 1
            # Options of score function: THR, APS, SAPS, RAPS
            # Define a conformal prediction algorithm. Optional: SplitPredictor, ClusteredPredictor, ClassWisePredictor
            if args.conformal_score == 'aps':
                predictor = SplitPredictor_GNN(score_function=APS(), model=model)   # _device = model._device
            elif args.conformal_score == 'thrrank':
                predictor = SplitPredictor_GNN(score_function=THRRANK(), model=model)

            # 1.1 train
            best_val_acc = final_test_acc = 0

            for epoch in range(1, args.epochs + 1):
                loss = train(epoch, model, data, optimizer, args.alpha)
                
                (train_acc, val_acc, tmp_test_calib_acc), pred = test(model, data, args.alpha, args.tau, args.target_size)
                if val_acc > best_val_acc:
                    #torch.save(best_model, model_checkpoint)
                    best_model = copy.deepcopy(model)
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                if args.verbose:
                    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Calib_Test=tmp_test_calib_acc)
            torch.save(best_model, model_checkpoint)
            pred = best_pred

        # 1.2 test
        (train_acc, val_acc, test_acc), _ = test(best_model, data, args.alpha, args.tau, args.target_size, size_loss = False)
        log(Stage="base model", Train_acc=train_acc, Val_acc=val_acc, Test_Calib_acc=test_acc)

        # STAGE 2
        # get_logits
        # logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
        logits = pred   

        # get calibration set
        # n_base = min(1000, int(calib_test.shape[0]/2))   # modified calib size
        n_base = int(calib_test.shape[0]/2) # half of n
        smx = logits[data.calib_test_mask].detach()
        labels = data.y[data.calib_test_mask].detach()

        # correct model
        model_to_correct = copy.deepcopy(model)
        if args.conf_correct_model == 'gnn':
            confmodel = ConfGNN(model_to_correct, data, args, args.confgnn_num_layers, args.confgnn_base_model, output_dim, args.task).to(args.device)
        optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=args.confgnn_lr)  # Only perform weight-decay on first convolution.
        pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
        best_size_loss = 10000
        best_val_acc = 0

        if args.conftr_calib_holdout:
            calib_test_idx = np.where(data.calib_test_mask)[0]
            np.random.seed(run)
            np.random.shuffle(calib_test_idx)
            calib_eval_idx = calib_test_idx[:int(n_base * args.calib_fraction)]
            calib_test_real_idx = calib_test_idx[int(n_base * args.calib_fraction):]

            data.calib_eval_mask = np.array([False] * len(y))
            data.calib_eval_mask[calib_eval_idx] = True
            data.calib_test_real_mask = np.array([False] * len(y))
            data.calib_test_real_mask[calib_test_real_idx] = True
            if args.verbose:
                print('Using a separate calibration holdout...')
            calib_eval_idx = np.where(data.calib_eval_mask)[0]
            np.random.seed(run)
            np.random.shuffle(calib_eval_idx)
            train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
            train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
            train_train_idx = np.where(data.train_mask)[0]

        print('Starting topology-aware conformal correction...')
        early_stop_win = 5000
        stop_improve_count = 0
        for epoch in range(1, args.conf_epochs + 1): 
            if not args.conftr_calib_holdout:
                train_idx = np.where(data.train_mask)[0]
                np.random.seed(epoch)
                np.random.shuffle(train_idx)
                train_train_idx = train_idx[:int(len(train_idx)/2)]
                train_calib_idx = train_idx[int(len(train_idx)/2):]
                train_test_idx = train_train_idx

            confmodel.train()
            optimizer.zero_grad()
            out, ori_out = confmodel(data.x, data.edge_index)

            '''
            # for THR
            
            out_softmax = F.softmax(out, dim = 1)
            ori_out_softmax = F.softmax(ori_out, dim = 1)

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp+1)*(1-args.alpha))/n_temp

            tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
            qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher') #interpolation的改变

            c = torch.sigmoid((out_softmax[train_test_idx] - qhat)/args.tau)
            size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - args.target_size))
            
            '''

            # For RANK
            # get logits from correct model
            out_softmax = F.softmax(out, dim = 1)

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp+1)*(1-args.alpha))/n_temp

            # rank distribution
            # thrrank_score = predictor.smooth_score_function(out[train_calib_idx], data.y[train_calib_idx])    # calibration set：label 
            mu_k = out_softmax[train_calib_idx][(torch.arange(len(train_calib_idx)), data.y[train_calib_idx])]  # mu_k(x)
            mu_y = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)]   # mu_y(x)
            thrrank_scores_signal_label = torch.sum(torch.sigmoid((mu_k - mu_y)/args.tau), dim=0)  # V(xi, yi) 
            # qhat = predictor._calculate_conformal_value(thrrank_scores, args.alpha)
            qhat = torch.quantile(thrrank_scores_signal_label, 1 - q_level, interpolation='higher') # indicator -> sigmoid

            # cal c accordding to scores_all_label
            cumsum_j = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)].repeat(output_dim, 1) 
            cumsum_k = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)].repeat(1, output_dim).view(-1, len(train_calib_idx))
            thrrank_scores_all_label = torch.sum(torch.sigmoid((cumsum_j - cumsum_k)/args.tau).view(-1,output_dim,len(train_calib_idx)),dim=1)

            c = torch.sigmoid((thrrank_scores_all_label - qhat)/args.tau)
            size_loss = torch.mean(torch.relu(torch.sum(c, axis = 0) - args.target_size))

            '''
            # For APS
            # get logits from correct model
            out_softmax = F.softmax(out, dim = 1)

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp+1)*(1-args.alpha))/n_temp

            # rank distribution
            # thrrank_score = predictor.smooth_score_function(out[train_calib_idx], data.y[train_calib_idx])    # calibration set：label 
            mu_k = out_softmax[train_calib_idx][(torch.arange(len(train_calib_idx)), data.y[train_calib_idx])]
            mu_y = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)]

            aps_scores_signal_label = torch.mean(torch.sigmoid((mu_y - mu_k)/args.tau)*mu_k, dim=-1)
            # qhat = predictor._calculate_conformal_value(aps_scores, args.alpha)
            qhat = torch.quantile(aps_scores_signal_label, q_level, interpolation='higher') #indicator -> sigmoid

            # cal c accordding to scores_all_label
            cumsum_j = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)].repeat(output_dim, 1)    
            cumsum_k = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), torch.unsqueeze(torch.arange(output_dim), 1)].repeat(1, output_dim).view(-1, len(train_calib_idx))
            aps_scores_all_label = torch.sum(torch.sigmoid((cumsum_k - cumsum_j)/args.tau)*cumsum_j.view(-1,output_dim,len(train_calib_idx)),dim=1)
            # aps_scores_all_label = torch.cumsum(out_softmax, dim=-1) 
            size_loss = torch.mean(torch.sigmoid((aps_scores_all_label - qhat)/args.tau))
            '''

            pred_loss = F.cross_entropy(out[train_train_idx], data.y[train_train_idx])
            
            if args.conftr:
                if epoch <= 1000:
                    loss = pred_loss
                else:
                    loss = pred_loss + args.size_loss_weight * size_loss
            else:
                loss = pred_loss
        
            loss.backward()
            optimizer.step()
            if args.verbose:
                log(Epoch = epoch, Prediction_loss = pred_loss.item(), size_loss = size_loss.item())
            loss = float(loss)
            pred_loss_hist.append(pred_loss.item())
            size_loss_hist.append(size_loss.item())

            (train_acc, val_acc, tmp_test_calib_acc), pred, size_loss = test(confmodel, data, args.alpha, args.tau, args.target_size, size_loss = True)

            # get_correct_logits
            # correct_logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
            correct_logits = pred
            # get valid set
            smx_valid_correct = correct_logits[data.valid_mask].detach()
            labels_valid_correct = data.y[data.valid_mask].detach()
            n_valid_correct = int(len(np.where(data.valid_mask)[0])/2)

            eff_valid = conf_pred(predictor, smx_valid_correct, n_valid_correct, labels_valid_correct, args.alpha, args.interpolation, num_classes=output_dim)[1]
            
            print("eff_valid: " + str(eff_valid))
            print("best_size_loss: " + str(best_size_loss))
            eff_valid = args.adp * eff_valid 

            val_size_loss_hist.append(size_loss)
            if args.conftr:
                if eff_valid < best_size_loss: 
                    best_size_loss = eff_valid
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                    best_epoch = epoch
                elif epoch > 1000:
                    stop_improve_count += 1
                if stop_improve_count >= early_stop_win:
                    print("early stop!")
                    break
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                if args.verbose:
                    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Calib_Test=tmp_test_calib_acc)

        # STAGE 3
        # get_final_correct_logits
        # best_logits = torch.nn.Softmax(dim = 1)(best_pred).detach().cpu().numpy()
        best_logits = best_pred
        # get final correct calibration set
        n_correct = int(n_base * (1-args.calib_fraction))  # modified calib size
        
        if args.conftr_calib_holdout:
            smx_correct = best_logits[data.calib_test_real_mask].detach()
            labels_correct = data.y[data.calib_test_real_mask].detach()
        else:
            smx_correct = best_logits[data.calib_test_mask].detach()
            labels_correct = data.y[data.calib_test_mask].detach()

        result_this_run['rcp_gnn'] = {}
        result_this_run['rcp_gnn'][args.conformal_score] = conf_pred(predictor,smx_correct, n_correct, labels_correct, args.alpha, args.interpolation, num_classes=output_dim)

        print('Finished training this run!')
        name = args.dataset + '_' + args.model+'_'+args.conformal_score
        if args.conftr_calib_holdout:
            name+='_calib_holdout'
        if args.conf_correct_model == 'gnn':    
            name += '_confgnn'
        if args.alpha != 0.1:
            name += '_alpha_' + str(args.alpha)    
        
        tau2res[run] = result_this_run
        print("experiment at run " + str(run) + " results: " + str(tau2res))

    # if not os.path.exists('./pred'):
    #     os.mkdir('./pred')
    # if not args.not_save_res:
        # print('Saving results to', './pred/' + name +'.pkl')
        # with open('./pred/' + name +'.pkl', 'wb') as f:
        #     pickle.dump(tau2res, f)
    print("experiment results:")
    print(tau2res)


main(args)