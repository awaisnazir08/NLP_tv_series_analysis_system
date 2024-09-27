import torch
import huggingface_hub

class JutsuClassifier():
    def __init__(self, model_path, data_path=None, text_column_name='text', label_column_name='jutsu', model_name='distilbert/distilbert-base-uncased', test_size=0.2, num_labels=3,huggingface_token = None):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.text_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.huggingface_token = huggingface_token
        
        if self.huggingface_token is not None:
            huggingface_hub.login(token=self.huggingface_token)
            
    