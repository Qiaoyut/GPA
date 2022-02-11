import torch
from dataset import get_dataset
import argparse
import numpy as np
from model import GCL_model,Search_mlp_pairs
import random
from eval_graph import Eval_unsuper, Eval_semi
from torch_geometric.data import DataLoader
from torch.autograd import Variable
import time

parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
parser.add_argument('--dataset', type=str, default='MUTAG', help='PROTEINS| NCI1 | DD |COLLAB | MUTAG|IMDB-BINARY|REDDIT-BINARY|REDDIT-MULTI-5K ') #PROTEINS;NCI1;COLLAB
parser.add_argument('--mode', type=str, default='unsuper', help='semi | unsuper')
parser.add_argument('--lr', default=0.0001, dest='lr', type=float,
        help='Learning rate.')

parser.add_argument('--num_gc_layers', type=int, default=2,
        help='Number of graph convolution layers before each pooling')
parser.add_argument('--hidden_dim', type=int, default=128, help='')
parser.add_argument('--hidden_dim_search', type=int, default=128, help='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_cpu', type=int, default=1, help='0: use cpu 1: use gpu')

parser.add_argument('--train_epoch', type=int, default=1)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=str, default='0', help='specify cuda devices')

parser.add_argument('--train_gnn_type', type=str, default='resgcn', help='resgcn | gin | gcn| gcn_raw') #gcn_raw is the standard gcn

parser.add_argument('--weight_decay', type=float,default=0.,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--label_rate', type=float, default=0.1,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--split_rate', type=float, default=0.1,
                    help="the ratio of spliting train and valid dataset")
parser.add_argument('--run_times', type=int, default=1,
                    help="the ratio of run times")

#Search parameter
parser.add_argument('--temperature', type=float, default=0.07,
                    help='Initial learning rate.')

def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class ValidLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        # dataset is torch.utils.data.dataset
        self.shuffle = shuffle
        self.loader = DataLoader(dataset, batch_size, shuffle=shuffle)
        self.ite_loader = iter(self.loader)

    def reset(self):
        self.ite_loader = iter(self.loader)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            batch = next(self.ite_loader)
        except StopIteration:
            self.reset()
            batch = next(self.ite_loader)
        return batch

def data_split_train_valid(dataset, ratio=0.1):
    num_class = dataset.num_classes
    y = dataset.data.y
    train_indices = []
    valid_indices = []
    all_index = torch.arange(0, len(dataset), 1)
    for i in range(num_class):
        mask_i = torch.eq(y, i)
        index_i = all_index[mask_i].numpy()
        np.random.shuffle(index_i)
        valid_len = int(len(index_i) * ratio)
        train_indices.append(torch.from_numpy(index_i[:-valid_len]))
        valid_indices.append(torch.from_numpy(index_i[-valid_len:]))

    train_indices = torch.cat(train_indices, dim=0)
    valid_indices = torch.cat(valid_indices, dim=0)
    train_dataset = dataset[train_indices.long()]
    valid_dataset = dataset[valid_indices.long()]
    return train_dataset, valid_dataset

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def hessian_vector_product(vector, dalpha_,model,vnet,data,r=1e-2):
    all_vector = vector + dalpha_
    R = r / _concat(all_vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.add_(R, v)

    #Forward
    z1,z2=model(data)
    atten=vnet(z1,z2)
    logits = model.compute_loss(z1, z2, atten)
    loss=torch.mean(logits)
    variable_list = [param for param in vnet.parameters()]
    grads_p = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.sub_(2 * R, v)

    z1, z2 = model(data)
    atten = vnet(z1,z2)
    logits = model.compute_loss(z1, z2, atten)
    loss = torch.mean(logits)
    grads_n = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    for p, v in zip(vnet.parameters(), dalpha_):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def train(feat_dim, device, data_loader, val_loader, args,save_path_model,save_path_search):
    # Original model
    model = GCL_model(feat_dim, args.hidden_dim,
                      n_layers=args.num_gc_layers, gnn=args.train_gnn_type, device=device)
    search_net = Search_mlp_pairs(device=device, in_dim=args.hidden_dim, hid_dim=args.hidden_dim_search)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': search_net.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    # Optimizer for search
    optimizer_search = torch.optim.Adam(search_net.parameters(), lr=args.lr, betas=(0.5, 0.999),
                                        weight_decay=args.weight_decay)

    # Valid model
    v_model = GCL_model(feat_dim, args.hidden_dim,
                        n_layers=args.num_gc_layers, gnn=args.train_gnn_type, device=device)
    v_search_net = Search_mlp_pairs(device=device, in_dim=args.hidden_dim, hid_dim=args.hidden_dim_search)
    v_optimizer = torch.optim.Adam([
        {'params': v_model.parameters()},
        {'params': v_search_net.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    model.to(device)
    v_model.to(device)
    search_net.to(device)
    v_search_net.to(device)

    min_loss = 1e9
    start_all_time=time.time()
    for epoch in range(args.epoch):
        start_epoch_time=time.time()
        model.train()
        search_net.train()
        epoch_loss = 0.0
        ite = 0
        for data in data_loader:
            v_model.load_state_dict(model.state_dict())
            v_search_net.load_state_dict(search_net.state_dict())

            # Get optimal W^* by one forward update via train_input
            z1, z2 = v_model(data)
            atten = v_search_net(z1,z2,binary=True)
            logits = v_model.compute_loss(z1, z2, atten)
            loss_train = torch.mean(logits)
            v_optimizer.zero_grad()
            loss_train.backward()
            v_optimizer.step()

            # v_net optimize get L_val(w^*, \alpha) based on valid_input
            val_data = val_loader.next()
            z1, z2 = v_model(val_data)
            atten = v_search_net(z1,z2,binary=False)
            logits = v_model.compute_loss(z1, z2, atten)
            loss_val = torch.mean(logits)
            v_optimizer.zero_grad()
            loss_val.backward()

            dalpha = [v.grad for v in v_search_net.parameters()]
            dalpha_ = [v.grad.data for v in v_search_net.parameters()]
            vector = [v.grad.data for v in v_model.parameters()]

            implicit_grads = hessian_vector_product(vector, dalpha_, model, search_net, data)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(args.lr, ig.data)

            # Update parameters for architecture based on Gradient_\alpha L_val(w^*, \alpha) - 2-th term
            i = 0
            for name, params in search_net.named_parameters():
                if params.requires_grad:
                    if params.grad is None:
                        params.grad = Variable(dalpha[i].data)
                    else:
                        params.grad.data.copy_(dalpha[i].data)
                    i += 1
            optimizer_search.step()

            z1, z2 = model(data)
            with torch.no_grad():
                atten = search_net(z1,z2,binary=True)
            logits = model.compute_loss(z1, z2, atten)
            loss_train = torch.mean(logits)
            optimizer.zero_grad()
            loss_train.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # nn.utils.clip_grad_norm_(search_net.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss_train.item()
            print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, ite + 1, loss_train.item()))
            ite = ite + 1
        end_epoch_time=time.time()
        print('--> Epoch %d  loss: %.3f  epoch_time: %.3f' % (epoch + 1, epoch_loss/ite,end_epoch_time-start_epoch_time))
    end_all_time=time.time()
#    print('--> Finish Searching! Total search time per epoch: ',(end_all_time-start_all_time)/epoch)

    torch.save(model.state_dict(), save_path_model)
    torch.save(search_net.state_dict(), save_path_search)



if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    if torch.cuda.is_available() and args.use_cpu==1:
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
    else:
        device = torch.device("cpu")
        print("--> No GPU")
    print_configuration(args)
    # for semi-supervised learning with label ratio 0.01
    if args.mode == "semi":
        dataset, dataset_pretrain = get_dataset(args.dataset, task='semisupervised')
        train_dataset, valid_dataset = data_split_train_valid(dataset_pretrain, ratio=args.split_rate)
        data_loader=DataLoader(dataset, args.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = ValidLoader(valid_dataset, args.batch_size)
        evaluator = Eval_semi(dataset, label_rate=args.label_rate, device=device)  # for unsupervised
        args.epoch = args.train_epoch
    else:
        dataset = get_dataset(args.dataset, task='unsupervised')
        train_dataset, valid_dataset = data_split_train_valid(dataset, ratio=args.split_rate)
        data_loader = DataLoader(dataset, args.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = ValidLoader(valid_dataset, args.batch_size)
        evaluator = Eval_unsuper(dataset, device=device)  # for unsupervised
        args.epoch = args.train_epoch
    print('*** {} statistics: num_instances={} num_class={}'.format(args.dataset, len(dataset), dataset.num_classes))

    feat_dim = dataset[0].x.shape[1]
    save_path_model = "./weights/h21_graphcl_model_" + args.mode + "_%s_" % args.dataset + "_train_encoder-"+args.train_gnn_type+ "_nlayer-%d" % args.num_gc_layers + "_train_epoch_"+"{}".format(args.train_epoch) +"_learning_rate_"+"{}".format(args.lr)+'_dim%d_' % args.hidden_dim \
                     + "search_dim-{}-".format(args.hidden_dim_search) + "{}".format(args.label_rate) + '.pth'
    save_path_search = "./weights/h21_graphcl_search_" + args.mode + "_%s_" % args.dataset + "_train_encoder-"+args.train_gnn_type+ "_nlayer-%d" % args.num_gc_layers + "_train_epoch_"+"{}".format(args.train_epoch) +"_learning_rate_"+"{}".format(args.lr)+ '_dim%d_' % args.hidden_dim \
                       + "search_dim-{}-".format(args.hidden_dim_search) + "{}".format(args.label_rate) + '.pth'
    print(save_path_model)
    print(save_path_search)
    #Train
    print('-> Start Training')
    print('-> Training Encoder: ',args.train_gnn_type)
    train(feat_dim, device, train_loader, val_loader, args,save_path_model,save_path_search)
    #Evaluate
    model = GCL_model(feat_dim, args.hidden_dim,
                      n_layers=args.num_gc_layers, gnn=args.train_gnn_type, device=device)
    model.to(device)
    model.load_state_dict(torch.load(save_path_model))

    if args.mode == "semi":
        acc, std=evaluator.evaluate_run(run_times=args.run_times,model=model)
        acc1,std1=evaluator.gridsearch_run(run_times=args.run_times,model=model)
    else:
        acc,std=evaluator.evaluate_run(run_times=args.run_times,model=model)
