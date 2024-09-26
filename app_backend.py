from data_loader import DataLoader
from classifier import CustomBertClassifier
from compute_tsne import ComputeTSNE
from prediction_handling import PredictionHandling

class AppBackend:
    
    def run(self, post_title, post_body, plot_tsne: bool):
        
        data_loader = DataLoader()
        
        post = data_loader.save_text_inputs(post_title=post_title, post_body=post_body)
        
        classifier = CustomBertClassifier()
        
        # improve this to return embeddings only if used by plot_tsne
        post_predictions, post_encoding = classifier.encode(post)
        
        prediction_handling = PredictionHandling()
        
        predictions_labels, predictions_confidence = prediction_handling.prediction_handling(predictions = post_predictions)
        
        tsne_to_plot = []
        
        if plot_tsne:
            
            encodings_tsne, labels_tsne = data_loader.load_encodings()
            
            compute_tsne = ComputeTSNE()
            
            encodings_tsne, labels_tsne = compute_tsne.preprocess_for_tsne(
                encodings = encodings_tsne, 
                labels = labels_tsne, 
                encodings_pred = post_encoding, 
                labels_pred = predictions_labels
            )

            tsne_to_plot = compute_tsne.compute_tsne(encodings=encodings_tsne, labels=labels_tsne)
            
        return predictions_labels, predictions_confidence, tsne_to_plot