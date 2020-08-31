import os
import tqdm
import logging

import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

import torch
from torch.utils.data import Dataset

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

from .models import embedder_clf

def assign_device(device_used):
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
    return device


def save_checkpoint(model, features_name, label_map, model_path):
    model.to(torch.device("cpu"))
    checkpoint = {
        'model': model,
        'features_name': features_name,
        'label_map': label_map
    }

    torch.save(checkpoint, model_path)
    logging.info("model saved")


def load_checkpoint(model_path):
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    features_name = checkpoint['features_name']
    label_map = checkpoint['label_map']
    return model, features_name, label_map
    

# hook
class ActivateFeaturesHook():
    def __init__(self, module):
        self.features = []
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features.append(output.cpu().data.numpy())
    def get_total_features(self):
        return np.vstack(self.features)
    def close(self):
        self.hook.remove()


# This will be used to create train and val sets
class BasicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


def evaluate(model, data, device, do_normalize=True):
    eval_loader = data
    # batch_losses, all_predictions, all_targets = [], [], []
    all_predictions, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm.tqdm(eval_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            # test_loss = criterion(output, targets)
            # batch_losses.append(test_loss.cpu().item())
            all_predictions.append(output.cpu().data.numpy())
            all_targets.append(targets.cpu().data.numpy())
    # average_loss = np.average(batch_losses)
    # return average_loss, all_predictions, all_targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    if do_normalize:
        logging.info("performing normalization...")
        all_predictions = normalize(all_predictions)
    return all_predictions, all_targets
    

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


def show_loss(hooks):
    loss_history = hooks.get_loss_history()
    plt.plot(loss_history['metric_loss'], 'r', alpha=0.5, label='metric_loss')
    if 'classifier_loss' in loss_history.keys():
        plt.plot(loss_history['classifier_loss'], 'b', alpha=0.5, label='classifier_loss')
    plt.plot(loss_history['total_loss'], 'y', alpha=0.5, label='total_loss')
    plt.legend()


def show_accuracy(hooks, tester):
    plt.figure(facecolor='w')
    for c,ds in zip(['r', 'b'], ['train', 'val']):
        accuracies = hooks.get_accuracy_history(tester, ds, return_all_metrics=True)
        plt.plot(accuracies['epoch'], accuracies['AMI_level0'], '{}x-'.format(c), label=ds)
        plt.plot(accuracies['epoch'], accuracies['r_precision_level0'], '{}o-'.format(c), label=ds)
                
    plt.legend()
    plt.title("Adjusted Mutual Info")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")


# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)


def find_important_markers(model, adata, X_val, y_val, reduce_by="mean", internal_batch_size=128):
    device = torch.device('cpu')
    logging.info("using device cpu in find_important_markers")

    # construct model    
    assert isinstance(model, embedder_clf)
    model.to(device)
    model.eval()

    # generate data
    test_input_tensor = torch.from_numpy(X_val).type(torch.FloatTensor).to(device)

    ig = IntegratedGradients(model)

    # process labels
    attr_l = []
    for i in np.unique(y_val):
        attr, delta = ig.attribute(test_input_tensor, target=int(i), n_steps=100, internal_batch_size=internal_batch_size, return_convergence_delta=True)
        attr = attr.detach().numpy()
        attr_l.append(attr)

    attr_reduce_l = []
    markers = []
    if reduce_by == "mean":
        for attr in attr_l:
            attr_mean = np.mean(attr, axis=0)
            attr_reduce_l.append(np.sort(attr_mean)[::-1][:100])
            markers.append(adata.var.index[np.argsort(attr_mean)[::-1][:100]])
    elif reduce_by == "median":
        for attr in attr_l:
            attr_median = np.median(attr, axis=0)
            attr_reduce_l.append(np.sort(attr_median)[::-1][:100])
            markers.append(adata.var.index[np.argsort(attr_median)[::-1][:100]])
    markers = pd.DataFrame(markers, index=np.unique(adata.obs['Cluster'])).T
    marker_importance = pd.DataFrame(attr_reduce_l, index=np.unique(adata.obs['Cluster'])).T

    return markers, marker_importance
