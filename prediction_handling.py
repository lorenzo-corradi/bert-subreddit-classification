from torch.nn import Softmax
import config
import torch

class PredictionHandling:
    
    def prediction_handling(self, predictions):
        
        softmax = Softmax(dim = 1)
        
        probabilities = softmax(predictions)
        
        best_predictions = torch.topk(input = probabilities, k = 3)
        
        predictions_confidence = best_predictions[0].detach().numpy().flatten() ## values
        predictions_values = best_predictions[1].detach().numpy().flatten() ## indices
        
        predictions_labels = [config.LABELS_DICT[prediction_value] for prediction_value in predictions_values]
        predictions_confidence = ['%.2f' % (prediction_confidence * 100) for prediction_confidence in predictions_confidence]
        
        return predictions_labels, predictions_confidence