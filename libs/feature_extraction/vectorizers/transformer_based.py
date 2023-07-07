from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from libs.feature_extraction.vectorizers.base import BaseVectorizer

class BertVectorizer(BaseVectorizer):
    name = "bert"
    trainable = False
    take_tokenized_data = False

    def __init__(self, path = None):
        if path:
            print(f"Loading model({path})...")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModel.from_pretrained(path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def fit(self, x):
       return self

    def transform(self, x):
        code_embeddings = []
        tokens = self.__get_tokens(x)
        rng = tokens["input_ids"].shape[0]
        for i in tqdm(range(rng)):
            selected_token = {"input_ids": tokens["input_ids"][i].view(1, -1).to(self.device), 'attention_mask': tokens["attention_mask"][i].view(1, -1).to(self.device)}
            embedding = self.__get_embedding(selected_token)
            code_embeddings.append(embedding)
        arr = np.array(code_embeddings)
        return arr.reshape(arr.shape[0], -1)

    def __get_tokens(self, x):
        print("Tokenizing...")
        token = {'input_ids': [], 'attention_mask': []}
        for sentence in tqdm(x):
            new_token = self.tokenizer.encode_plus(sentence, max_length=300,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')
            token['input_ids'].append(new_token['input_ids'][0])
            token['attention_mask'].append(new_token['attention_mask'][0])
        token['input_ids'] = torch.stack(token['input_ids'])
        token['attention_mask'] = torch.stack(token['attention_mask'])
        return token

    def __get_embedding(self, token):
        with torch.no_grad():
            output = self.model(**token)
            embeddings = output.last_hidden_state
            att_mask = token['attention_mask']
            mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
            mask_embeddings = embeddings * mask
            summed = torch.sum(mask_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / summed_mask
            mean_pooled = mean_pooled.detach().cpu().numpy()
            return mean_pooled.reshape(-1, 1)

    @classmethod
    def save_vectors(cls, x, path):
        """Save the vectors to a file with .npy format"""
        return np.save(path, x)

    @classmethod
    def load_vectors(cls, path) -> np.ndarray:
        """Load the vectors from a file with .npy format"""
        return np.load(path)