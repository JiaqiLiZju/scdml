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

from .models import DenseEmbeddingNet
from .utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def scdml(adata, obs_label="Celltype",
            device_used="cpu",
            test_size=0.1,
            out_sz=25, emb_szs=[1000, 500, 250, 100], ps=0, use_bn=False, actn=nn.ReLU(),
            lr=0.00001, weight_decay=0.0001,
            margin=0.1, distance_norm=2, 
            miner_m=4,
            batch_size = 64,
            dataloader_num_workers=0,
            test_interval = 1, 
            patience = 5,
            num_epochs=100,
            embedding_on_tsne=True,
):

    assert obs_label in adata.obs.columns

    # assign device
    if device_used == "cpu":
        device = torch.device("cpu")
        logging.info("using device cpu")
        if torch.cuda.is_available():
            logging.warning("using device cpu, cuda is available!")
    elif device_used == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("using device cuda")
        else:
            device = torch.device("cpu")
            logging.warning("using device cpu")

    # create dictionary of label map
    label_map = dict(enumerate(adata.obs[obs_label].cat.categories))

    # get data and format
    data, labels = adata.X, adata.obs[obs_label].cat.codes.values

    # train_test_split
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                    labels,
                                                    stratify=labels,
                                                    test_size=test_size,
                                                    random_state=77)
    X_train = data[X_train_idx]
    X_val= data[X_val_idx]
    logging.info("train data size %d;\t test data size" % (len(X_train_idx, len(X_val_idx))))

    # Training, validation, holdout set
    train_dataset = BasicDataset(X_train, y_train)
    val_dataset = BasicDataset(X_val, y_val)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    model = DenseEmbeddingNet(in_sz=len(adata.var),
                        out_sz=out_sz,
                        emb_szs=emb_szs,
                        ps=ps,
                        use_bn=use_bn,
                        actn=actn)

    # model = nn.DataParallel(model).to(device)
    # adata = model(adata)
    # Matrix adata: M cells * out_sz genes // M*25

    # Set optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=margin,
                                    distance_norm=distance_norm, 
                                    power=1, 
                                    swap=False, )

    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(y_train.flatten(), m=miner_m, length_before_new_iter=len(train_dataset))

    # Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value

    # Set other training parameters
    batch_size = batch_size
    logging.info("setting batch size = %d" % batch_size)

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
                                                dataloader_num_workers = dataloader_num_workers,
                                                use_trunk_output=True)


    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                                dataset_dict, 
                                                model_folder, 
                                                test_interval = test_interval,
                                                test_collate_fn=torch.utils.data._utils.collate.default_collate,
                                                patience = patience)


    trainer = trainers.MetricLossOnly(models,
                                    optimizers,
                                    batch_size,
                                    loss_funcs,
                                    mining_funcs,
                                    train_dataset,
                                    sampler=sampler,
                                    dataloader_num_workers = dataloader_num_workers,
                                    collate_fn=torch.utils.data._utils.collate.default_collate,
                                    end_of_iteration_hook = hooks.end_of_iteration_hook,
                                    end_of_epoch_hook = end_of_epoch_hook)

    # early stopping
    trainer.train(num_epochs=num_epochs)
    
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
    adata.varm['PCs'] = comb_src_df.loc[adata.obs.index].values
    # config params 
    adata.uns['pca'] = {}
    adata.uns['pca']['type'] = "scdml"
    # adata.uns['pca']['variance'] = pca_.explained_variance_
    # adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    if embedding_on_tsne:
        # get tsne coords
        comb_tsne = TSNE().fit_transform(comb_emb)

        # set adata tsne
        comb_tsne_df = pd.DataFrame(comb_tsne, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
        adata.obsm['X_tsne'] = comb_tsne_df.loc[adata.obs.index].values

    return adata


def scdml_clf(adata, obs_label="Celltype",
            test_size=0.1,
            out_sz=25, emb_szs=[1000, 500, 250, 100], ps=0, use_bn=False, actn=nn.ReLU(),
            clf_output_size=None,
            embedder_lr=0.00001, embedder_weight_decay=0.0001,
            classifier_lr=0.00001, classifier_weight_decay=0.0001,
            margin=0.1, distance_norm=2, 
            miner_m=4,
            batch_size = 64,
            metric_loss_weight=1, classifier_loss_weight=0.5,
            dataloader_num_workers=0,
            test_interval = 1, 
            patience = 5,
            num_epochs=100,
            embedding_on_tsne=True,
):

    assert obs_label in adata.obs.columns
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dictionary of label map
    label_map = dict(enumerate(adata.obs[obs_label].cat.categories))

    # get data and format
    data, labels = adata.X, adata.obs[obs_label].cat.codes.values

    X_train_idx, X_val_idx, y_train, y_val = train_test_split(range(len(data)),
                                                    labels,
                                                    stratify=labels,
                                                    test_size=test_size,
                                                    random_state=77)
    X_train = data[X_train_idx]
    X_val= data[X_val_idx]

    # Training, validation, holdout set
    train_dataset = BasicDataset(X_train, y_train)
    val_dataset = BasicDataset(X_val, y_val)
    # hld_dataset = BasicDataset(hld_data, hld_labels)

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = DenseEmbeddingNet(in_sz=len(adata.var),
                            out_sz=out_sz,
                            emb_szs=emb_szs,
                            ps=ps,
                            use_bn=use_bn,
                            actn=actn)

    # Set the classifier. The classifier will take the embeddings and output a 50 dimensional vector.
    # (Our training set will consist of the first 50 classes of the CIFAR100 dataset.)
    # We'll specify the classification loss further down in the code.
    if clf_output_size is None:
        clf_output_size = len(np.unique(labels))

    classifier = DenseEmbeddingNet(in_sz=out_sz, out_sz=clf_output_size, emb_szs=[out_sz], ps=0)

    # embedder = nn.DataParallel(embedder).to(device)
    # classifier = nn.DataParallel(classifier).to(device)
    embedder = embedder.to(device)
    classifier = classifier.to(device)

    # Set optimizers
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=embedder_lr, weight_decay=embedder_weight_decay)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=classifier_weight_decay)

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=margin,
                                    distance_norm=distance_norm, 
                                    power=1, 
                                    swap=False, )

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(y_train.flatten(), m=miner_m, length_before_new_iter=len(train_dataset))

    # Set other training parameters
    batch_size = batch_size

    # Package the above stuff into dictionaries.
    models = {"trunk": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": metric_loss_weight, "classifier_loss": classifier_loss_weight}

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
                                                dataloader_num_workers = dataloader_num_workers)

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                                dataset_dict, 
                                                model_folder, 
                                                test_interval = test_interval,
                                                test_collate_fn=torch.utils.data._utils.collate.default_collate,
                                                patience = patience)

    trainer = trainers.TrainWithClassifier(models,
                                    optimizers,
                                    batch_size,
                                    loss_funcs,
                                    mining_funcs,
                                    train_dataset,
                                        
                                    sampler=sampler,
                                    dataloader_num_workers = dataloader_num_workers,
                                    loss_weights = loss_weights,
                                    collate_fn = torch.utils.data._utils.collate.default_collate,
                                    end_of_iteration_hook = hooks.end_of_iteration_hook,
                                    end_of_epoch_hook = end_of_epoch_hook)


    trainer.train(num_epochs=num_epochs)

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
    adata.varm['PCs'] = comb_src_df.loc[adata.obs.index].values
    # config params 
    adata.uns['pca'] = {}
    adata.uns['pca']['type'] = "scdml"
    # adata.uns['pca']['variance'] = pca_.explained_variance_
    # adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    if embedding_on_tsne:
        # get tsne coords
        comb_tsne = TSNE().fit_transform(comb_emb)

        # set adata tsne
        comb_tsne_df = pd.DataFrame(comb_tsne, index=adata.obs.index[np.concatenate([X_train_idx, X_val_idx])])
        adata.obsm['X_tsne'] = comb_tsne_df.loc[adata.obs.index].values

    # find_markers
    markers, marker_importance = find_important_markers(embedder, classifier, adata, X_val, y_val)

    return adata, markers, marker_importance


def inference_pretrained(model, pretrained_features, label_map, adata_new, batch_size=128, embedding_on_tsne=True):
    assert isinstance(model, embedder_clf)
    
    # transfer to complex held out datasets
    # set the reference features list
    # features = ordered genes list
    pretrained_features = pd.Index(pretrained_features)

    adata_new.var['gene_ids'] = adata_new.var.index.astype('category')
    adata_new.var['gene_ids'].cat.set_categories(pretrained_features.to_list(), inplace=True)
    idx = adata_new.var.sort_values('gene_ids', ascending=True).index

    mask_1 = pretrained_features.isin(adata_new.var.index)
    mask_2 = adata_new.var.index.isin(pretrained_features)
    
    # new_obs * old_vars
    hld_data = np.zeros((adata_new.obs.shape[0], len(pretrained_features)))
    hld_data[:, mask_1] = adata_new.X[:,mask_2]
    # hld_data[:, mask_1] = adata_new.X.A[:,mask_2]
    # hld_labels = adata_new.obs['cell_ontology_class'].cat.codes.values
    hld_labels = np.zeros((adata_new.obs.shape[0], 1))

    hld_dataset = BasicDataset(hld_data, hld_labels)
    hld_dataloader = torch.utils.data.DataLoader(hld_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # register embedding hook 
    activations = ActivateFeaturesHook(model.embedder)
    probs, _ = evaluate(model, hld_dataloader, device)
    hld_emb = activations.get_total_features()
    activations.close()

    label = np.argmax(probs, axis=-1)
    label = [label_map[x] for x in label]

    adata_new.obs["scdml_annotation"] = label

    # scanpy api
    # set adata pca
    adata_new.obsm['scdml'] = hld_emb
    # config params 
    adata_new.uns['scdml'] = {}
    adata_new.uns['scdml']['type'] = "scdml"
    # adata.uns['pca']['variance'] = pca_.explained_variance_
    # adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    if embedding_on_tsne:
        # get tsne coords
        comb_tsne = TSNE().fit_transform(hld_emb)
        # set adata tsne
        adata_new.obsm['X_tsne'] = comb_tsne

    return adata_new