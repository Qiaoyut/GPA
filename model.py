from functools import partial
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
from method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
    UniformSample, RWSample, RandomView, RawView

class GCL_model(torch.nn.Module):
    r"""A wrapped :class:`torch.nn.Module` class for the convinient instantiation of
    pre-implemented graph encoders.

    Args:
        feat_dim (int): The dimension of input node features.
        hidden_dim (int): The dimension of node-level (local) embeddings.
        n_layer (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
        pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
            (default: :obj:`sum`)
        gnn (string, optional): The type of GNN layer, :obj:`gcn`, :obj:`gin` or
            :obj:`resgcn`. (default: :obj:`gin`)
        bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
        act (string, optional): The activation function, :obj:`relu` or :obj:`prelu`.
            (default: :obj:`relu`)
        bias (bool, optional): Whether to include bias term in Linear. (default: :obj:`True`)
        xavier (bool, optional): Whether to apply xavier initialization. (default: :obj:`True`)
        node_level (bool, optional): If :obj:`True`, the encoder will output node level
            embedding (local representations). (default: :obj:`False`)
        graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
            embeddings (global representations). (default: :obj:`True`)
        edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
            compute the aggregation. (default: :obj:`False`)

    Note
    ----
    For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be
    :obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`.
    For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.

    Examples
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=5, aug_ratio=0.2, pool='sum',
                 gnn='gin', bn=True, act='relu', bias=True, proj_head='MLP', xavier=True, tau=0.5, device=None,
                 object_loss='NCE', node_level=False, graph_level=True, edge_weight=False):
        super(GCL_model, self).__init__()

        self.tau = tau
        self.proj_head = proj_head
        self.aug_ratio = aug_ratio
        self.hidden_dim = hidden_dim
        self.proj_out_dim = hidden_dim
        self.object_loss = object_loss
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        if gnn == 'gin':
            self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn, act)
        elif gnn == 'gcn':
            self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'gcn_raw':
            self.encoder = GCNRaw(feat_dim, hidden_dim, n_layers, pool, bn,
                               act, bias, xavier, edge_weight)
        elif gnn == 'resgcn':
            self.encoder = ResGCN(feat_dim, hidden_dim, num_conv_layers=n_layers,
                                  global_pool=pool)
        self.views_fn1,self.views_fn2 = self.aug_funs()
        self.node_level = node_level
        self.graph_level = graph_level
        if gnn == "gin" or gnn == "gcn":
            self.d_out = self.hidden_dim * n_layers
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim * n_layers)
        else:
            self.proj_head_g = self._get_proj(self.proj_head, self.hidden_dim)
            self.d_out = self.hidden_dim

    def aug_funs(self):
        #The first view
        views_fn1 = []
        views_fn1.append(RawView()) #raw
        views_fn1.append(UniformSample(ratio=self.aug_ratio)) #dropN
        views_fn1.append(EdgePerturbation(ratio=self.aug_ratio)) #permE
        views_fn1.append(RWSample(ratio=self.aug_ratio)) #subgraph
        views_fn1.append(NodeAttrMask(mask_ratio=self.aug_ratio)) #maskN

        #The second view
        views_fn2 = []
        views_fn2.append(RawView())  # raw
        views_fn2.append(UniformSample(ratio=self.aug_ratio))  # dropN
        views_fn2.append(EdgePerturbation(ratio=self.aug_ratio))  # permE
        views_fn2.append(RWSample(ratio=self.aug_ratio))  # subgraph
        views_fn2.append(NodeAttrMask(mask_ratio=self.aug_ratio))  # maskN

        assert (len(views_fn1) == 5)
        assert (len(views_fn2) == 5)

        return views_fn1,views_fn2

    def _get_proj(self, proj_head, in_dim):

        if callable(proj_head):
            return proj_head

        assert proj_head in ['linear', 'MLP']

        # out_dim = self.proj_out_dim
        out_dim = self.proj_out_dim

        if proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
        elif proj_head == 'MLP':
            proj_nn = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        return proj_nn

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encode_view(self, data):
        z_g = self.encoder(data)
        return z_g

    def forward(self, data):
        #views_fn1,views_fn2=self.aug_funs()
        zs1 = []
        # views = [v_fn(data).to(self.device) for v_fn in self.views_fn]
        for v_fn in self.views_fn1:
            view_data = v_fn(data).to(self.device)
            # print('**** device={}'.format(self.device))
            z = self.encode_view(view_data)
            z = self.proj_head_g(z)
            zs1.append(z)

        zs2 = []
        # views = [v_fn(data).to(self.device) for v_fn in self.views_fn]
        for v_fn in self.views_fn2:
            view_data = v_fn(data).to(self.device)
            # print('**** device={}'.format(self.device))
            z = self.encode_view(view_data)
            z = self.proj_head_g(z)
            zs2.append(z)

        return zs1,zs2

    # def compute_loss(self, zs_n1,zs_n2,atten):
    #     tot_loss=[]
    #     for i in range(0,len(zs_n1)):
    #         for j in range(i,len(zs_n2)):
    #             loss = self.NT_Xent(zs_n1[i], zs_n2[j], tau=self.tau)
    #             loss=loss.data.cpu().numpy()
    #             tot_loss.append(loss)
    #     tot_loss=np.array(tot_loss)
    #     tot_loss=tot_loss.T
    #     tot_loss=torch.from_numpy(tot_loss).to(self.device)
    #     tot_loss=torch.mul(tot_loss,atten)
    #     sum_loss=torch.sum(tot_loss,dim=1)
    #     return sum_loss

    def compute_loss(self, zs_n1,zs_n2,atten):
        tot_loss = []
        for i in range(0, len(zs_n1)):
            for j in range(i, len(zs_n2)):
                loss = self.NT_Xent(zs_n1[i], zs_n2[j], tau=self.tau)
                # loss=loss.data.cpu().numpy() # cannot transfer to numpy since it will break the computation graph
                tot_loss.append(loss)
        tot_loss = torch.stack(tot_loss, dim=1)
        tot_loss = torch.mul(tot_loss, atten)
        sum_loss = torch.sum(tot_loss, dim=1)
        return sum_loss

    def NT_Xent(self, z1, z2, tau=0.5, norm=True):
        '''
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        return loss

class GIN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True):
        super(GIN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        self.act = torch.nn.PReLU() if act == 'prelu' else torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
                            self.act,
                            Linear(hidden_dim, hidden_dim, bias=bias))
            if xavier:
                self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep
        # return global_rep, x


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep
        # return global_rep, x


class GCNRaw(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False,
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCNRaw, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        # global_rep = torch.cat(xpool, 1)
        global_rep = xpool[-1]

        return global_rep

class ResGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=False, edge_norm=True, gfn=False):
        super(ResGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weights_init()

    def weights_init(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = ResGCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class ResGCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim,
                 num_feat_layers=1,
                 num_conv_layers=3,
                 num_fc_layers=2, xg_dim=None, bn=True,
                 gfn=False, collapse=False, residual=False,
                 global_pool="sum", dropout=0, edge_norm=True):

        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.collapse = collapse
        self.bn = bn

        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(ResGCNConv, edge_norm=edge_norm, gfn=gfn)

        if xg_dim is not None:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(xg_dim)
            self.lin1_xg = Linear(xg_dim, hidden_dim)
            self.bn2_xg = BatchNorm1d(hidden_dim)
            self.lin2_xg = Linear(hidden_dim, hidden_dim)
        else:
            self.use_xg = False

        if collapse:
            self.bn_feat = BatchNorm1d(feat_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(feat_dim, feat_dim),
                    torch.nn.ReLU(),
                    Linear(feat_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden_dim))
                hidden_in = hidden_dim
        else:
            self.bn_feat = BatchNorm1d(feat_dim)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = ResGCNConv(feat_dim, hidden_dim, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    Linear(hidden_dim, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden_dim))
                self.convs.append(GConv(hidden_dim, hidden_dim))
            self.bn_hidden = BatchNorm1d(hidden_dim)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_dim))
                self.lins.append(Linear(hidden_dim, hidden_dim))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is of shape [n_graphs, feat_dim]
            xg = self.bn1_xg(data.xg) if self.bn else xg
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg) if self.bn else xg
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x) if self.bn else x
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x) if self.bn else x
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        local_rep = x
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x) if self.bn else x
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class Search_aug(torch.nn.Module):
    def __init__(self,device=None,num_augpairs=15,temperature=0.07):
        super(Search_aug, self).__init__()
        self.num_augpairs=num_augpairs
        self.temperature = temperature

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        self.a=Parameter(torch.ones(size=[1,num_augpairs], dtype=torch.float) / (self.num_augpairs**2))

    def forward(self,data):
        atten=self.a.repeat(data[0].shape[0],1)
        atten = torch.softmax(atten / self.temperature, -1)
        batch_size,c=atten.shape
        if not self.training:
            eye_atten=torch.eye(c)
            atten_value, atten_index=torch.max(atten,1)
            atten=eye_atten[atten_index]
            atten=atten.to(self.device)
        return atten

class Search_mlp(torch.nn.Module):
    def __init__(self,device=None,in_dim=128,num_augpairs=15):
        super(Search_mlp, self).__init__()
        self.num_augpairs = num_augpairs
        self.in_dim=in_dim
        self.device=device

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        self.module=nn.Sequential(nn.Linear(in_features=in_dim, out_features=in_dim),
                       nn.ReLU(inplace=True),
                      nn.Linear(in_features=in_dim, out_features=num_augpairs))
    def forward(self,data):
        x=self.module(data[0])
        x=torch.softmax(x,dim=1)
        batch_size, c=x.shape
        if not self.training:
            eye_atten=torch.eye(c)
            atten_value, atten_index=torch.max(x,1)
            x=eye_atten[atten_index]
            x=x.to(self.device)
        return x

class Search_mlp_pairs(torch.nn.Module):
    def __init__(self,device=None,in_dim=128, hid_dim=128):
        super(Search_mlp_pairs, self).__init__()
        self.in_dim=in_dim
        self.hid_dim = hid_dim
        self.device=device

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        self.module=nn.Sequential(nn.Linear(in_features=2*in_dim, out_features=hid_dim),
                       nn.ReLU(inplace=True),
                      nn.Linear(in_features=hid_dim, out_features=1))

    def forward(self,data1,data2,binary=False):
        tot_atten=[]
        for i in range(0,len(data1)):
            for j in range(i,len(data2)):
                input_data=torch.cat((data1[i],data2[j]),dim=1)
                atten=self.module(input_data)
                tot_atten.append(atten)
        tot_atten=torch.stack(tot_atten,dim=1)
        tot_atten=torch.softmax(tot_atten,dim=1)
        tot_atten=torch.squeeze(tot_atten,dim=2)
        batch_size, c = tot_atten.shape
        if not self.training:
            eye_atten = torch.eye(c)
            atten_value, atten_index = torch.max(tot_atten, 1)
            tot_atten = eye_atten[atten_index]
            tot_atten = tot_atten.to(self.device)
        if binary is True:
            eye_atten = torch.eye(c)
            atten_value, atten_index = torch.max(tot_atten, 1)
            tot_atten = eye_atten[atten_index]
            tot_atten = tot_atten.to(self.device)
        return tot_atten




        # x=self.module(data[0])
        # x=torch.softmax(x,dim=1)
        # batch_size, c=x.shape
        # if not self.training:
        #     eye_atten=torch.eye(c)
        #     atten_value, atten_index=torch.max(x,1)
        #     x=eye_atten[atten_index]
        #     x=x.to(self.device)
        # return x

