import random
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform

from .TOSICA_model import scTrans_model as create_model


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:, -1])
    ct_counts = pd.Series(data[:, -1]).value_counts()
    max_total = int(os.environ.get("TOSICA_MAX_TOTAL", "2000000"))
    max_val = min(int(ct_counts.max()), max(24, int(max_total / len(ct_counts))))
    blocks = []
    for ct in ct_names:
        tmp = data[data[:, -1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        blocks.append(tmp[idx])
    return np.vstack(blocks).astype(np.float32)
  
def splitDataSet(adata,label_name='Celltype', tr_ratio= 0.7): 
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    X_arr = np.asarray(todense(adata), dtype=np.float32)
    genes = np.array(adata.var_names)
    y_enc = label_encoder.fit_transform(adata.obs[label_name].astype('str').to_numpy()).astype(np.float32)
    inverse = label_encoder.inverse_transform(range(0, int(np.max(y_enc)) + 1))
    el_data = np.column_stack([X_arr, y_enc]).astype(np.float32)
    el_data = balance_populations(data = el_data)
    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    tr_arr = el_data[train_dataset.indices]
    va_arr = el_data[valid_dataset.indices]
    exp_train = torch.from_numpy(tr_arr[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(tr_arr[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(va_arr[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(va_arr[:,-1].astype(np.int64))
    return exp_train, label_train, exp_valid, label_valid, inverse, genes

def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    gene_to_idx = {g: i for i, g in enumerate(feature_list)}
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for g in dict_pathway[k]:
            idx_g = gene_to_idx.get(g)
            if idx_g is not None:
                p_mask[idx_g, j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss() 
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=8, embed_dim=48,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    #train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse,genes = splitDataSet(adata,label_name)
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        #print(mask.shape)
        print('Mask loaded!')
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train)+1)
    #print(num_classes)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    eff_bs = min(batch_size, max(8, len(train_dataset) // 10))
    use_pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=eff_bs,
                                               shuffle=True,
                                               pin_memory=use_pin,drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=max(eff_bs, 64),
                                             shuffle=False,
                                             pin_memory=use_pin,drop_last=False)
    model = create_model(num_classes=num_classes, num_genes=len(exp_train[0]),  mask = mask,embed_dim=embed_dim,depth=depth,num_heads=num_heads,has_logits=False).to(device) 
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name) 
    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]  
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step() 
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if platform.system().lower() == 'windows':
            torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
        else:
            torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
    print('Training finished!')

#train(adata, gmt_path, pre_weights, batch_size=8, epochs=20)


