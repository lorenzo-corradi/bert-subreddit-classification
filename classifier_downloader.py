import os.path
import config
from sentence_transformers import SentenceTransformer

class BertClassifierDownloader():
    
    def __init__(self, filename):
    
        path = os.path.join(config.MODELS_DIR, filename)
        
        bert = SentenceTransformer(filename)
        
        bert.save(path)
        
        return