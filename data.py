from datasets import load_dataset
import pandas as pd

class MyDataset():
    def __init__(self):
    
        self.dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
        self.record = self.dataset[0]
        
    
    def __getitem__(self, index):
        return pd.DataFrame(self.dataset[index]['notes'])
    
    def __len__(self):
        return len(self.df)


