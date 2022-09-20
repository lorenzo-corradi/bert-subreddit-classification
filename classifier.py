from classifier_downloader import BertClassifierDownloader
import config
import os.path
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from singleton import Singleton

class BertClassifier:
    
    def __init__(self, base_classifier_name = 'bert-base-uncased'):
        
        self.base_classifier_name = base_classifier_name
        self.base_classifier_path = os.path.join(config.MODELS_DIR, self.base_classifier_name)
        
        if (not os.path.exists(self.base_classifier_path)):
            print('Classifier not found. Downloading...')
            classifier_downloader = BertClassifierDownloader(filename=base_classifier_name)
            print('Download completed in folder:\n{}'.format(self.base_classifier_path))
            
        self.base_classifier = SentenceTransformer(self.base_classifier_path)


class CustomBertClassifier(BertClassifier, metaclass = Singleton):
    
    def __init__(self, 
        base_classifier_name = 'bert-base-uncased', 
        trained_classifier_name = 'subreddit_classification_model.pt'
    ):
        self.trained_classifier_name = trained_classifier_name
        self.trained_classifier_path = os.path.join(config.MODELS_DIR, self.trained_classifier_name)
        
        super().__init__(base_classifier_name)
        
        DIM_HIDDEN1 = 384
        DIM_HIDDEN2 = 64
        DIM_HIDDEN3 = 512
        N_CLASSES = 9
        
        self.trained_classifier = BERTClass(DIM_HIDDEN1, DIM_HIDDEN2, DIM_HIDDEN3, N_CLASSES, bert = self.base_classifier_name)
        
        checkpoint = torch.load(
            self.trained_classifier_path, 
            map_location=torch.device('cpu')
        )
        self.trained_classifier.load_state_dict(checkpoint['model_state_dict'])
        
    def encode(self, sentence):
        return self.trained_classifier(sentences = sentence)


# NOTE: helper class to allow successful load_state_dict
class BERTClass(nn.Module):
    def __init__(self, dim_hidden1, dim_hidden2, dim_hidden3, n_class, bert):
        super(BERTClass, self).__init__()
        
        self.dim_hidden1 = dim_hidden1
        self.dim_hidden2 = dim_hidden2
        self.dim_hidden3 = dim_hidden3
        self.n_class = n_class

        self.BERT = BertModel.from_pretrained(bert)
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        
        self.dropout = nn.Dropout(0.15)
        # NOTE: BERT creates 768-dimension embeddings
        self.linear1 = nn.Linear(768, self.dim_hidden1)
        self.linear2 = nn.Linear(self.dim_hidden1, self.dim_hidden2)
        self.linear3 = nn.Linear(self.dim_hidden2, self.dim_hidden3)
        self.linear4 = nn.Linear(self.dim_hidden3, self.n_class)
        self.activation = nn.GELU()
        
    def forward(self, sentences):
        tokenized_sentences = self.tokenizer(sentences, padding = True, truncation = True, return_tensors = 'pt')
        x = self.BERT(input_ids = tokenized_sentences['input_ids'], attention_mask = tokenized_sentences['attention_mask'])
        
        encodings = self.mean_pooling(x = x, attention_mask = tokenized_sentences['attention_mask'])
        
        x = self.activation(self.dropout(self.linear1(encodings)))
        x = self.activation(self.linear2(x))
        x = self.activation(self.dropout(self.linear3(x)))
        x = self.activation(self.linear4(x))
        
        return x, encodings
        
    def mean_pooling(self, x, attention_mask):
        token_embeddings = x[0] # First element of BERT output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)