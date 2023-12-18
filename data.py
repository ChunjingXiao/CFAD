import scipy.io as sio
import glob
from utils import *
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import networkx as nx
from models import GAE

def task_generator(feature, l_list, ul_list, bs, device):
    feature_l = []
    label_l = []
    feature_l_qry = []
    label_l_qry = []

    for i in range(len(feature)//2):
        perm_l = list(range(len(l_list[i])))
        random.shuffle(perm_l)

        perm_ul = list(range(len(ul_list[i])))
        random.shuffle(perm_ul)

        # generate support set
        support_idx = np.array(l_list[i])[perm_l[:int(bs / 2)]].tolist()
        support_idx += np.array(ul_list[i])[perm_ul[:int(bs / 2)]].tolist()
        label_t = np.concatenate((np.ones(int(bs / 2)), np.zeros(int(bs / 2))))

        feature_l.append(feature[i][support_idx].to(device))
        label_l.append(torch.FloatTensor(label_t).to(device))

        # generate query set
        bs_qry = 2 * (len(l_list[i]) - int(bs / 2))
        qry_idx = np.array(l_list[i])[perm_l[-int(bs_qry / 2):]].tolist()
        qry_idx += np.array(ul_list[i])[perm_ul[-int(bs_qry / 2):]].tolist()
        label_t_qry = np.concatenate((np.ones(int(bs_qry / 2)), np.zeros(int(bs_qry / 2))))

        feature_l_qry.append(feature[i][label_t_qry].to(device))
        label_l_qry.append(torch.FloatTensor(label_t_qry).to(device))

    return feature_l, label_l, feature_l_qry, label_l_qry


def test_task_generator(feature, l_list, ul_list, bs, label, test_idx, device):
    feature_l = []
    label_l = []

    for q in range(3):
        perm_l = list(range(len(l_list)))
        random.shuffle(perm_l)

        perm_ul = list(range(len(ul_list)))
        random.shuffle(perm_ul)

        # generate support set
        support_idx = np.array(l_list)[perm_l[:int(bs / 2)]].tolist()
        support_idx += np.array(ul_list)[perm_ul[:int(bs / 2)]].tolist()
        gan_idx = [i+1 for i in support_idx]
        support_idx +=gan_idx
        label_t = np.concatenate((np.ones(int(bs / 2)), np.zeros(int(bs / 2))))
        label_t +=label_t
        feature_l.append(feature[support_idx].to(device))
        label_l.append(torch.FloatTensor(label_t).to(device))

    return feature_l, label_l, feature[test_idx].to(
        device), torch.FloatTensor(label[test_idx]).to(device)


def test_task_generator_backup(feature, l_list, ul_list, bs, label, test_idx, device):
    perm_l = list(range(len(l_list)))
    random.shuffle(perm_l)

    perm_ul = list(range(len(ul_list)))
    random.shuffle(perm_ul)

    # generate support set
    support_idx = np.array(l_list)[perm_l[:int(bs / 2)]].tolist()
    support_idx += np.array(ul_list)[perm_ul[:int(bs / 2)]].tolist()
    label_t = np.concatenate((np.ones(int(bs / 2)), np.zeros(int(bs / 2))))

    return feature[support_idx].to(device), torch.FloatTensor(label_t).to(device), feature[test_idx].to(
        device), torch.FloatTensor(label[test_idx]).to(device)


def load_yelp(file):
    data = sio.loadmat(file)
    network = data['Network'].astype(np.float)
    labels = data['Label'].flatten()
    attributes = data['Attributes'].astype(np.float)

    return network, attributes, labels


def sp_matrix_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class DataProcessor:

    def __init__(self, num_graph, degree, data_name):
        self.num_graph = num_graph  # number of auxiliary graph + 1 (target graph)
        self.degree = degree
        self.data_name = data_name
        self.feature_l, self.label_l, self.adj_l = [], [], []
        self.target, self.target_adj, self.target_feature, self.target_label= None, None, None, None
        self.target_idx_train_ano_all, self.target_idx_train_normal_all, self.target_idx_val, self.target_idx_test = None, None, None, None

        self.labeled_idx_l, self.unlabeled_idx_l = [], []
        self.target_labeled_idx, self.target_unlabeled_idx = [], []

    def data_loader(self):

        l = glob.glob("graphs/{}/*.mat".format(self.data_name))

        f_l = random.sample(l, self.num_graph)
        random.shuffle(f_l)
        for f in f_l[:-1]:
            adj, feature, label = load_yelp(f)
            adj = normalize_adjacency(adj)
            adj = sp_matrix_to_torch_sparse_tensor(adj).float()
            feature = torch.FloatTensor(feature.toarray())
            feature = sgc_precompute(feature, adj, self.degree)

            self.feature_l.append(feature)
            self.label_l.append(label)
            self.adj_l.append(adj)

        # load the target graph
        self.target = f_l[-1]
        adj, feature, label = load_yelp(self.target)
        # adj = normalize_adjacency(adj)

        self.target_adj = sp_matrix_to_torch_sparse_tensor(normalize_adjacency(adj)).float()
        self.target_feature = torch.FloatTensor(feature.toarray())
        
        G = nx.from_numpy_matrix(adj)

        data = from_networkx(G)
        edge_index = data.edge_index
        data = Data(x= torch.FloatTensor(feature.toarray()),edge_index=edge_index,y=label)
        graphs = []
        cores =[]
        for i in range(len(label)):
            node_idx = i
            k = 2 
            nodes, edges, mapping, _ = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=True)
            subgraph = Data(x=data.x[nodes], edge_index=edges,y=data.y[nodes])
            graphs.append(subgraph)
            cores.append( (nodes == i).nonzero().item())

        model = GAE(feature.shape[1], 64)
        
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(100):
            total_loss = 0
            for data in DataLoader(graphs, batch_size=32, shuffle=True):
                optimizer.zero_grad()
                x = data.x
                edge_index = data.edge_index

                output = model(x, edge_index,False)
                loss = criterion(output, x)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            average_loss = total_loss / len(graphs)
            print(f"GAN Epoch [{epoch+1}/100],GAN Loss: {average_loss:.4f}")

        model.eval()
        total_test_loss = 0
        t=0
        gan_feature=[]
        gan_label = []
        with torch.no_grad():
            for data in DataLoader(graphs, batch_size=1):
                x = data.x
                edge_index = data.edge_index
                a = cores[t]
                mask = random.sample(list(range(x.shape[0])), x.shape[0]//2)
                mask.append(a)
                output = model(x, edge_index,True,mask)
                loss = criterion(output, x)

                adj = to_dense_adj(edge_index)[0]
                target=sgc_precompute(output, adj, self.degree)
                gan_feature.append(target[a].numpy())
                
                gan_label.append(data.y[0][a]) 
                t = t+1
                total_test_loss += loss.item()
            
        average_test_loss = total_test_loss
        

        gan_feature = torch.tensor(gan_feature)
        feature_l = []
        for feature_t in self.feature_l:
            feature_l.append(torch.cat((feature_t, gan_feature), dim=0))
        self.feature_l = feature_l
        self.target_feature = sgc_precompute(self.target_feature, self.target_adj, self.degree)
        self.target_label = label
        # split the target graph into train/valid/test with 4/2/4
        idx_anomaly = np.random.permutation(np.nonzero(self.target_label == 1)[0])
        idx_normal = np.random.permutation(np.nonzero(self.target_label == 0)[0])
        split_ano = int(0.4 * len(idx_anomaly))
        split_normal = int(0.4 * len(idx_normal))
        self.target_feature= torch.cat((self.target_feature, gan_feature), dim=0)

        self.target_idx_train_ano_all = idx_anomaly[:split_ano]
        self.target_idx_train_normal_all =  idx_normal[:split_normal]
        self.target_idx_val = np.concatenate((idx_anomaly[split_ano:-split_ano], idx_normal[split_normal:-split_normal])).tolist()
        self.target_idx_test = np.concatenate((idx_anomaly[-split_ano:], idx_normal[-split_normal:])).tolist()

        print("data loading finished.")

    def sample_anomaly(self, num_labeled_ano):

        for i in range(self.num_graph - 1):
            # sampling anomalies from auxiliary graphs
            label_tmp = self.label_l[i]
            idx_anomaly = np.random.permutation(np.nonzero(label_tmp == 1)[0])
            idx_normal = np.random.permutation(np.nonzero(label_tmp == 0)[0])
            self.labeled_idx_l.append(idx_anomaly[:num_labeled_ano].tolist())
            self.unlabeled_idx_l.append(np.concatenate((idx_normal, idx_anomaly[num_labeled_ano:])).tolist())

        self.target_idx_train_ano_all = np.random.permutation(self.target_idx_train_ano_all)
        self.target_idx_train_normal_all = np.random.permutation(self.target_idx_train_normal_all)

        if num_labeled_ano <= len(self.target_idx_train_ano_all):
            self.target_labeled_idx = self.target_idx_train_ano_all[:num_labeled_ano].tolist()
            self.target_unlabeled_idx = np.concatenate((self.target_idx_train_normal_all, self.target_idx_train_ano_all[num_labeled_ano:])).tolist()

        return [self.feature_l, self.labeled_idx_l, self.unlabeled_idx_l], \
               [self.target_feature, self.target_labeled_idx, self.target_unlabeled_idx]


