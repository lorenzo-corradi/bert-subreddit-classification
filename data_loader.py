import pandas as pd
import torch
import os
import config
import glob
import pickle

class DataLoader:
    
    def save_text_inputs(self, post_title, post_body):
                
        if not post_title:
            raise ValueError('Title of post cannot be empty. Retry.')
        
        post = post_title + ' ' + post_body
        
        return post
    
    
    def load_encodings(self, keyword = '*tsne*'):
        
        glob_filepath = os.path.join(config.ENCODINGS_DIR, keyword)
        
        filepath_encodings = glob.glob(glob_filepath)[0]
        filepath_labels = glob.glob(glob_filepath)[1]
            
        encodings = torch.load(filepath_encodings, map_location=torch.device('cpu'))
        
        with open(filepath_labels, 'rb') as filepath_labels_pickle:
            labels = pickle.load(filepath_labels_pickle)
        
        return encodings, labels