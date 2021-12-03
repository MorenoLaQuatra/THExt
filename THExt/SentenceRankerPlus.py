import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

from transformers import get_linear_schedule_with_warmup



from tqdm import tqdm
import os

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)



class SentenceRankerPlus():
    def __init__(self, train_set=None, eval_set=None, base_model_name = None, model_name_or_path=None, 
                epochs = 4, batch_size=32, lr = 2e-5, max_length=384):
        print ("Full implementation release upon paper acceptance")