import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class ComputeTSNE:
    
    MOUSE_10X_COLORS = {
        0: "#FFFF00",
        1: "#1CE6FF",
        2: "#FF34FF",
        3: "#FF4A46",
        4: "#008941",
        5: "#006FA6",
        6: "#A30059",
        7: "#FFDBE5",
        8: "#7A4900",
        9: "#0000A6",
        10: "#63FFAC",
        11: "#B79762",
        12: "#004D43",
        13: "#8FB0FF",
        14: "#997D87",
        15: "#5A0007",
        16: "#809693",
        17: "#FEFFE6",
        18: "#1B4400",
        19: "#4FC601",
        20: "#3B5DFF",
        21: "#4A3B53",
        22: "#FF2F80",
        23: "#61615A",
        24: "#BA0900",
        25: "#6B7900",
        26: "#00C2A0",
        27: "#FFAA92",
        28: "#FF90C9",
        29: "#B903AA",
        30: "#D16100",
        31: "#DDEFFF",
        32: "#000035",
        33: "#7B4F4B",
        34: "#A1C299",
        35: "#300018",
        36: "#0AA6D8",
        37: "#013349",
        38: "#00846F",
    }
    
    def preprocess_for_tsne(self, encodings, labels, encodings_pred = None, labels_pred = None):
        
        if (encodings_pred is not None):
            encodings = torch.cat((encodings, encodings_pred))
            del encodings_pred
            gc.collect()
        if (labels_pred is not None):
            labels.append(labels_pred[0])
        
        encodings = encodings.detach().numpy()
        labels = np.array(labels)
        
        return encodings, labels
    
    
    def compute_tsne(self, encodings, labels):
        
        pca = PCA(n_components=50)
        encodings = pca.fit_transform(encodings)
        
        tsne = TSNE(
            perplexity=30,
            n_components=2, 
            learning_rate=200, 
            early_exaggeration=50,
            verbose=3,
            metric='euclidean',
            init = 'pca',
            n_iter = 500
        )
        
        tsne_embedding = tsne.fit_transform(encodings)
        
        del encodings
        gc.collect()
        
        X = {
            'x': tsne_embedding[:,0],
            'y': tsne_embedding[:,1],
            'labels': labels
        }
        
        del tsne_embedding
        gc.collect()
        
        tsne_to_plot = pd.DataFrame(data = X)
        
        return tsne_to_plot