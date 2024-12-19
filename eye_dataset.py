import glob
import torch
import pandas as pd
import numpy as np


class EyeDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.fnames = sorted(glob.glob(r'../../datasets/eyetracking/Eye-tracking-Kaggle/*.csv'))            
        print(self.fnames)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        data = pd.read_csv(self.fnames[idx],  low_memory=False)
        idx_good = (data['Category Left'] == data['Category Right'])
        data = data[idx_good]
        pupil_l = pd.to_numeric(data['Pupil Diameter Left [mm]'].values[1:], errors='coerce', downcast='float')
        pupil_r = pd.to_numeric(data['Pupil Diameter Right [mm]'].values[1:], errors='coerce', downcast='float')
        x_l = pd.to_numeric(data['Point of Regard Left X [px]'].values[1:], errors='coerce', downcast='float')
        x_r = pd.to_numeric(data['Point of Regard Right X [px]'].values[1:], errors='coerce', downcast='float')
        y_l = pd.to_numeric(data['Point of Regard Left Y [px]'].values[1:], errors='coerce', downcast='float')
        y_r = pd.to_numeric(data['Point of Regard Right Y [px]'].values[1:], errors='coerce', downcast='float')
        category_l = data['Category Left']
        category_r = data['Category Right']
        sample = [x_l, x_r, y_l, y_r, pupil_l, pupil_r]
        return sample
    

    def encode(unique_labels, labels):    
        unique_labels = {label: idx for idx, label in enumerate(unique_labels)}
        integer_data = np.array([unique_labels[label] for label in labels])

        return integer_data
    

if __name__ == "__main__":
    D = EyeDataset()
    s = D[0]
    print(s)