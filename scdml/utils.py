
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

from .models import embedder_clf

# This will be used to create train and val sets
class BasicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


def evaluate(model, data, device):
    eval_loader = data
    # batch_losses, all_predictions, all_targets = [], [], []
    all_predictions, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in eval_loader:
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


def find_important_markers(embedder, classifier, adata, X_val, y_val):
    # construct model
    model = embedder_clf(embedder, classifier)
    model.to(torch.device('cpu'))
    model.eval()

    # generate data
    test_input_tensor = torch.from_numpy(X_val).type(torch.FloatTensor).to(torch.device('cpu'))

    ig = IntegratedGradients(model)

    # process labels
    attr_l = []
    for i in np.unique(y_val):
        attr, delta = ig.attribute(test_input_tensor, target=int(i), n_steps=100, internal_batch_size=1024, return_convergence_delta=True)
        attr = attr.detach().numpy()
        attr_l.append(attr)

    attr_mean_l = []
    markers = []
    for attr in attr_l:
        attr_mean = np.mean(attr, axis=0)
        attr_mean_l.append(np.sort(attr_mean)[::-1][:100])
        markers.append(adata.var.index[np.argsort(attr_mean)[::-1][:100]])

    markers = pd.DataFrame(markers, index=np.unique(adata.obs['Cluster'])).T
    marker_importance = pd.DataFrame(attr_mean_l, index=np.unique(adata.obs['Cluster'])).T

    return markers, marker_importance