import spacy
from nltk import sent_tokenize
import os
import pandas as pd
from ast import literal_eval
from ..utils.data_loader import load_subtitles_dataset

class NamedEntityRecognizer():
    def __init__(self):
        self.nlp_model = self.load_model()
        pass
    def load_model(self):
        nlp = spacy.load('en_core_web_trf')
        return nlp
    
    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)
        ner_output = []
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == 'PERSON':
                    full_name = entity.text
                    first_name = full_name.split(' ')[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)
        return ner_output

    def get_ners(self, dataset_path, save_path = None):
        if save_path is not None and os.path.exists(save_path):
            # read_csv does not read lists as column values, instead saves them as strings
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        
        # load the dataset
        df = load_subtitles_dataset(dataset_path)
        # df = df.head(10)
        # Run the inference 
        df['ners'] = df['script'].apply(self.get_ners_inference)
        if save_path is not None:
            df.to_csv(save_path, index = False)
        return df

if __name__ == '__main__':
    named_entity_recognizer = NamedEntityRecognizer()
    named_entity_recognizer.get_ners(r'D:\VS Code Folders\NLP_Analysis\data\subtitles')
