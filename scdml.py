# Data manipulation
import numpy as np
import pandas as pd
import scanpy as sc

# ML
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

# torch
import torch
import torch.nn as nn

# pytorch_metric_learning
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets

# Logs
import logging
logging.getLogger().setLevel(logging.INFO)


from models import DenseEmbeddingNet
from utils import *


def scdml(adata, obs_label="Celltype"):

    assert obs_label in adata.obs.columns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dictionary of label map
    label_map = dict(enumerate(adata.obs[obs_label].cat.categories))

    # get data and format
    data, labels = adata.X, adata.obs[obs_label].cat.codes.values

    # train_test_split
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                    labels,
                                                    stratify=labels,
                                                    test_size=0.1,
                                                    random_state=77)
    X_train = data[X_train_idx]
    X_val= data[X_val_idx]

    logging.info(len(X_train_idx))
    logging.info(len(X_val_idx))

    # Training, validation, holdout set
    train_dataset = BasicDataset(X_train, y_train)
    val_dataset = BasicDataset(X_val, y_val)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    model = EmbeddingNet(in_sz=len(adata.var),
                        out_sz=25,
                        emb_szs=[1000, 500, 250, 100],
                        ps=0,
                        use_bn=False,
                        actn=nn.ReLU())
    model = nn.DataParallel(model).to(device)
    # adata = model(adata)
    # Matrix adata: M cells * out_sz genes // M*25

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
    # early stopping

    # Get a dictionary mapping from loss names to lists
    loss_histories = hooks.get_loss_history() 

    # The first argument is the tester object. The second is the split name.
    # Get a dictionary containing the keys "epoch" and the primary metric
    # Get all accuracy histories
    acc_histories = hooks.get_accuracy_history(tester, "val", return_all_metrics=True)
    acc_histories = hooks.get_accuracy_history(tester, "train", return_all_metrics=True)

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

    # scanpy api
    # set adata pca
    comb_src_df = pd.DataFrame(comb_src, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
    # adata.varm['PCs'] = comb_src_df.loc[adata.obs.index].values
    # config params 
    # adata.uns['pca']['type'] = "scdml"
    # adata.uns['pca']['variance'] = pca_.explained_variance_
    # adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    # get tsne coords
    comb_tsne = TSNE().fit_transform(comb_emb)

    # set adata tsne
    comb_tsne_df = pd.DataFrame(comb_tsne, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
    adata.obsm['X_tsne'] = comb_tsne_df.loc[adata.obs.index].values

    return adata


def scdml_clf(adata, obs_label="Celltype"):

    assert obs_label in adata.obs.columns
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dictionary of label map
    label_map = dict(enumerate(adata.obs[obs_label].cat.categories))

    # get data and format
    data, labels = adata.X, adata.obs[obs_label].cat.codes.values

    X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                    labels,
                                                    stratify=labels,
                                                    test_size=0.1,
                                                    random_state=77)
    X_train = data[X_train_idx]
    X_val= data[X_val_idx]
    X_train_idx[:5], X_val_idx[:5], X_train[:5], X_val[:5], y_train[:5], y_val[:5]

    # Training, validation, holdout set
    train_dataset = BasicDataset(X_train, y_train)
    val_dataset = BasicDataset(X_val, y_val)
    # hld_dataset = BasicDataset(hld_data, hld_labels)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = EmbeddingNet(in_sz=len(adata.var),
                        out_sz=64,
                        emb_szs=[2000, 1000, 500, 100],
                        ps=[0.1, 0.1, 0.1, 0.1],
                        use_bn=False,
                        actn=nn.ReLU())

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
    models = {"trunk": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": 1, "classifier_loss": 0.5}

    record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"train": train_dataset, "val": val_dataset}
    model_folder = "example_saved_models"

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, 
                                                reference_set="compared_to_self", 
                                                normalize_embeddings=True, 
                                                use_trunk_output=True,
                                                # visualizer = umap.UMAP(), 
                                                # visualizer_hook = visualizer_hook,
                                                dataloader_num_workers = 0)

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

    # Get a dictionary mapping from loss names to lists
    loss_histories = hooks.get_loss_history() 

    # The first argument is the tester object. The second is the split name.
    # Get a dictionary containing the keys "epoch" and the primary metric
    # Get all accuracy histories
    acc_histories = hooks.get_accuracy_history(tester, "val", return_all_metrics=True)
    acc_histories = hooks.get_accuracy_history(tester, "train", return_all_metrics=True)

    ## inference
    # extract embeddings
    train_emb, train_lab = tester.get_all_embeddings(train_dataset, embedder, collate_fn=torch.utils.data._utils.collate.default_collate,)
    val_emb, val_lab = tester.get_all_embeddings(val_dataset, embedder, collate_fn=torch.utils.data._utils.collate.default_collate,)
    # Visualize embeddings using tSNE 
    # combine validation and holdout embeddings
    comb_emb = np.concatenate((train_emb, val_emb))
    comb_lab = np.concatenate((train_dataset.labels, val_dataset.labels))
    comb_src = np.concatenate((np.repeat("TRAIN", len(train_emb)),
                            np.repeat("VAL", len(val_emb))))

    # scanpy api
    # set adata pca
    comb_src_df = pd.DataFrame(comb_src, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
    # adata.varm['PCs'] = comb_src_df.loc[adata.obs.index].values
    # config params 
    # adata.uns['pca']['type'] = "scdml"
    # adata.uns['pca']['variance'] = pca_.explained_variance_
    # adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    # get tsne coords
    comb_tsne = TSNE().fit_transform(comb_emb)

    # set adata tsne
    comb_tsne_df = pd.DataFrame(comb_tsne, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
    adata.obsm['X_tsne'] = comb_tsne_df.loc[adata.obs.index].values

    # find_markers
    markers, marker_importance = find_important_markers(embedder, classifier, adata, X_val, y_val)

    return adata, markers, marker_importance


def inference_pretrained(model, adata):
    # transfer to complex held out datasets
    # set the reference features list
    # features = ordered genes list
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
