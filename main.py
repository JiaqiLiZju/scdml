# Main
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import logging
from cycler import cycler
import record_keeper
import pytorch_metric_learning

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import scanpy as sc

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(8.7,6.27)})
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# Logs
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)

class EmbeddingNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super(EmbeddingNet, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        if ps == 0:
          ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]),
                  actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(n_in=emb_szs[i], n_out=emb_szs[i+1], bn=use_bn, p=ps[i], actn=actn)
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        
    def bn_drop_lin(self, n_in:int, n_out:int, bn:bool=True, p:float=0., actn:nn.Module=None):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers
              
    def forward(self, x):
        output = self.fc(x)
        return output

# This will be used to create train and val sets
class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get data and format
adata = sc.read_h5ad("./data/DMCA.h5ad")
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata

# create dictionary of label map
label_map = dict(enumerate(adata.obs['CellType'].cat.categories))
# extract of some of the most representative clusters for training/testing
clusters = [0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,21,22,23,24,25,26]
indices = adata.obs['OldCellType'].cat.codes.isin(clusters)

data, labels = adata.X[indices], adata.obs[indices]['OldCellType'].cat.codes.values

# extract holdout clusters for projection
hld_data, hld_labels = adata.X[~indices], adata.obs[~indices]['OldCellType'].cat.codes.values

X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                  labels,
                                                  stratify=labels,
                                                  test_size=0.1,
                                                  random_state=77)
X_train = data[X_train_idx]
X_val= data[X_val_idx]

# Training, validation, holdout set
train_dataset = BasicDataset(X_train, y_train)
val_dataset = BasicDataset(X_val, y_val)
hld_dataset = BasicDataset(hld_data, hld_labels)

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
model = EmbeddingNet(in_sz=len(adata.var),
                     out_sz=25,
                     emb_szs=[1000, 500, 250, 100],
                     ps=0,
                     use_bn=False,
                     actn=nn.ReLU())
model = nn.DataParallel(model).to(device)

# Set optimizers
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

# Set the loss function
loss = losses.TripletMarginLoss(margin=0.1,
                                distance_norm=2, 
                                power=1, 
                                swap=False, )

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(y_train.flatten(), m=4, length_before_new_iter=len(train_dataset))

# Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value

# Set other training parameters
batch_size = 64

# Package the above stuff into dictionaries.
models = {"trunk": model}
optimizers = {"trunk_optimizer": model_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}


# Remove logs if you want to train with new parameters
# !rm -rf example_logs/ example_saved_models/ example_tensorboard/

record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"train": train_dataset, "val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook,
                                            dataloader_num_workers = 32,
                                            use_trunk_output=True)


end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                            dataset_dict, 
                                            model_folder, 
                                            test_interval = 1,
                                            test_collate_fn=torch.utils.data._utils.collate.default_collate,
                                            patience = 5)


trainer = trainers.MetricLossOnly(models,
                                  optimizers,
                                  batch_size,
                                  loss_funcs,
                                  mining_funcs,
                                  train_dataset,
                                  sampler=sampler,
                                  dataloader_num_workers = 32,
                                  collate_fn=torch.utils.data._utils.collate.default_collate,
                                  end_of_iteration_hook = hooks.end_of_iteration_hook,
                                  end_of_epoch_hook = end_of_epoch_hook)


trainer.train(num_epochs=100)

# Get a dictionary mapping from loss names to lists
loss_histories = hooks.get_loss_history() 

# The first argument is the tester object. The second is the split name.
# Get a dictionary containing the keys "epoch" and the primary metric
# Get all accuracy histories
acc_histories = hooks.get_accuracy_history(tester, "val", return_all_metrics=True)
acc_histories = hooks.get_accuracy_history(tester, "train", return_all_metrics=True)



####################################################################################
## inference
# extract embeddings
train_emb, train_lab = tester.get_all_embeddings(train_dataset, model, collate_fn=torch.utils.data._utils.collate.default_collate,)
val_emb, val_lab = tester.get_all_embeddings(val_dataset, model, collate_fn=torch.utils.data._utils.collate.default_collate,)
# Visualize embeddings using tSNE 
# combine validation and holdout embeddings
comb_emb = np.concatenate((train_emb, val_emb))
comb_lab = np.concatenate((train_dataset.labels, val_dataset.labels))
comb_src = np.concatenate((np.repeat("TRAIN", len(train_emb)),
                           np.repeat("VAL", len(val_emb))))

# get tsne coords
comb_tsne = TSNE().fit_transform(comb_emb)

sns.scatterplot(x=comb_tsne[:,0], 
                y=comb_tsne[:,1], 
                hue=comb_src)
plt.title('Training & Val Embeddings tSNE')
plt.show()
sns.scatterplot(x=comb_tsne[:,0], 
                y=comb_tsne[:,1], 
                hue=[label_map[i] for i in comb_lab],
                style=comb_src,
                palette='Paired')
plt.title('Training & Val Embeddings tSNE')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

####################################################################################
# set adata pca
comb_src_df = pd.DataFrame(comb_src, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
adata.varm['PCs'] = comb_src_df.loc[adata.obs.index].values
# config params 
adata.uns['pca'] = {}
# adata.uns['pca']['variance'] = pca_.explained_variance_
# adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

# set adata tsne
comb_tsne_df = pd.DataFrame(comb_tsne, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
adata.obsm['X_tsne'] = comb_tsne_df.loc[adata.obs.index].values

####################################################################################
# transfer to held out datasets
# set the reference features list
adata_TM = sc.read_h5ad("./data/TM_Lung.h5ad")
adata_TM.var['gene_ids'] = adata_TM.var['gene_ids'].astype('category')
adata_TM.var['gene_ids'].cat.set_categories(adata.var.index.to_list(), inplace=True)
idx = adata_TM.var.sort_values('gene_ids', ascending=True).index
adata_TM[:, idx]
mask_1 = adata.var.index.isin(adata_TM.var.index)
mask_2 = adata_TM.var.index.isin(adata.var.index)

hld_data = np.zeros((1748, 16726))
hld_data[:, mask_1] = adata_TM.X.A[:,mask_2]
hld_labels = adata_TM.obs['cell_ontology_class'].cat.codes.values

hld_dataset = BasicDataset(hld_data, hld_labels)

hld_emb, hld_lab = tester.get_all_embeddings(hld_dataset, model, collate_fn=torch.utils.data._utils.collate.default_collate,)
comb_tsne = TSNE().fit_transform(hld_emb)

# set adata
adata_TM.obsm['X_tsne'] = comb_tsne
sc.pl.tsne(adata_TM, color='cell_ontology_class', size=50)


####################################################################################
## learning with classifier
# adata_raw.write_h5ad("./data/EC_Atlas.h5ad")
adata_raw = sc.read_h5ad("./data/EC_Atlas.h5ad")
# adata
adata_raw.obs.head()
np.unique(adata_raw.obs['Cluster'], return_counts=True)
# create dictionary of label map
label_map = dict(enumerate(adata_raw.obs['Cluster'].cat.categories))
label_map
# extract of some of the most representative clusters for training/testing
clusters = list(range(24))
# clusters = np.unique(adata_raw.obs['Cluster'].cat.codes)[np.unique(adata_raw.obs['Cluster'].cat.codes, return_counts=True)[-1] > 100]
# clusters = clusters[clusters!=0]
indices = adata_raw.obs['Cluster'].cat.codes.isin(clusters)
data, labels = adata_raw.X[indices], adata_raw.obs[indices]['Cluster'].cat.codes.values

# extract holdout clusters for projection
# hld_data, hld_labels = adata_raw.X[~indices], adata.obs[~indices]['Type'].cat.codes.values
np.unique(labels, return_counts=True)
X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                  labels,
                                                  stratify=labels,
                                                  test_size=0.1,
                                                  random_state=77)
X_train = data[X_train_idx]
X_val= data[X_val_idx]
X_train_idx[:5], X_val_idx[:5], X_train[:5], X_val[:5], y_train[:5], y_val[:5]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training, validation, holdout set
train_dataset = BasicDataset(X_train, y_train)
val_dataset = BasicDataset(X_val, y_val)
# hld_dataset = BasicDataset(hld_data, hld_labels)

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = EmbeddingNet(in_sz=len(adata_raw.var),
                     out_sz=64,
                     emb_szs=[2000, 1000, 500, 100],
                     ps=[0.1, 0.1, 0.1, 0.1],
                     use_bn=False,
                     actn=nn.ReLU())

# embedder = ResDenseEmbeddingNet(in_sz=len(adata_raw.var),
#                          out_sz=64,
#                          emb_szs=[5000, 1000, 500, 128],
#                          ps=0,
#                          use_bn=False,
#                          actn=nn.ReLU())

# Set the classifier. The classifier will take the embeddings and output a 50 dimensional vector.
# (Our training set will consist of the first 50 classes of the CIFAR100 dataset.)
# We'll specify the classification loss further down in the code.
classifier = EmbeddingNet(in_sz=64, out_sz=24, emb_szs=[24], ps=0)

# embedder = nn.DataParallel(embedder).to(device)
# classifier = nn.DataParallel(classifier).to(device)
embedder = embedder.to(device)
classifier = classifier.to(device)

# Set optimizers
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=0.0001)

# Set the loss function
loss = losses.TripletMarginLoss(margin=0.1)

# Set the classification loss:
classification_loss = torch.nn.CrossEntropyLoss()

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(y_train.flatten(), m=4, length_before_new_iter=len(train_dataset))

# Set other training parameters
batch_size = 512

# Package the above stuff into dictionaries.
# models = {"trunk": model}
# optimizers = {"trunk_optimizer": model_optimizer}
# loss_funcs = {"metric_loss": loss}
# mining_funcs = {"tuple_miner": miner}

# Package the above stuff into dictionaries.
models = {"trunk": embedder, "classifier": classifier}
optimizers = {"trunk_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
mining_funcs = {"tuple_miner": miner}

# We can specify loss weights if we want to. This is optional
loss_weights = {"metric_loss": 1, "classifier_loss": 1}

import umap

record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"train": train_dataset, "val": val_dataset}
model_folder = "example_saved_models"

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20,15))
    plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)   
    plt.show()

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, 
                                            reference_set="compared_to_self", 
                                            normalize_embeddings=True, 
                                            use_trunk_output=True,
                                            visualizer = umap.UMAP(), 
                                            visualizer_hook = visualizer_hook,
                                            dataloader_num_workers = 32)

end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                            dataset_dict, 
                                            model_folder, 
                                            test_interval = 1,
                                            test_collate_fn=torch.utils.data._utils.collate.default_collate,
                                            patience = 1)

trainer = trainers.TrainWithClassifier(models,
                                  optimizers,
                                  batch_size,
                                  loss_funcs,
                                  mining_funcs,
                                  train_dataset,
                                       
                                  sampler=sampler,
                                  dataloader_num_workers = 0,
                                  loss_weights = loss_weights,
                                  collate_fn = torch.utils.data._utils.collate.default_collate,
                                  end_of_iteration_hook = hooks.end_of_iteration_hook,
                                  end_of_epoch_hook = end_of_epoch_hook)

trainer.train(num_epochs=100)

loss_history = hooks.get_loss_history()
plt.plot(loss_history['metric_loss'], 'r', alpha=0.5, label='metric_loss')
plt.plot(loss_history['classifier_loss'], 'b', alpha=0.5, label='classifier_loss')
# plt.plot(loss_history['total_loss'], 'y', alpha=0.5, label='total_loss')
plt.legend()

plt.figure(facecolor='w')
for c,ds in zip(['r', 'b'], ['train', 'val']):
    accuracies = hooks.get_accuracy_history(tester, ds, return_all_metrics=True)
    plt.plot(accuracies['epoch'], accuracies['AMI_level0'], '{}x-'.format(c), label=ds)
#     plt.plot(accuracies['epoch'], accuracies['r_precision_level0'], '{}o-'.format(c), label=ds)
    
plt.legend()
plt.title("Adjusted Mutual Info")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

####################################################################################
## importance interpret
# load model
model.load_state_dict(torch.load("./example_saved_models/trunk_best25.pth"))
# model.to(torch.device('cpu'))

test_input_tensor = torch.from_numpy(X_val).type(torch.FloatTensor).to(torch.device('cpu'))

ig = IntegratedGradients(model)

test_input_tensor.requires_grad_()
attr, delta = ig.attribute(test_input_tensor, target=1, return_convergence_delta=True)
attr = attr.detach().numpy()

test_input_tensor.requires_grad_()
attr_0, delta = ig.attribute(test_input_tensor, target=0, return_convergence_delta=True)
attr_0 = attr_0.detach().numpy()

