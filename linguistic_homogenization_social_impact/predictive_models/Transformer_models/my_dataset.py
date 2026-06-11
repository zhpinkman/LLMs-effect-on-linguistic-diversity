from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class EssayDataset(Dataset):
    def __init__(self, csv_path, scaler, tokenizer):
        df = pd.read_csv(csv_path)
        df.dropna(inplace=True)
        self.texts = df['text'].tolist()
        clabels = df[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].values.tolist()
        self.clabels = np.array([[1 if label=='y' else 0 for label in labels] for labels in clabels])
        self.zlabels_orig = df[['zEXT', 'zNEU', 'zAGR', 'zCON', 'zOPN']].values.tolist()
        self.zlabels = scaler.fit_transform(df[['zEXT', 'zNEU', 'zAGR', 'zCON', 'zOPN']])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        encodings = self.tokenizer(self.texts[index],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=2048, 
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encodings['input_ids'].squeeze()
        local_attention_mask = encodings['attention_mask'].squeeze()
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
        global_attention_mask[0] = 1
        clabels = self.clabels[index]
        zlabels = self.zlabels[index]
        return {
            "input_ids": input_ids,
            'global_attention_mask': global_attention_mask,
            "local_attention_mask": local_attention_mask,
            'clabels': torch.tensor(clabels, dtype=torch.long),
            'zlabels': torch.tensor(zlabels, dtype=torch.float),
            'zlabels_orig': torch.tensor(self.zlabels_orig[index], dtype=torch.float)
        }