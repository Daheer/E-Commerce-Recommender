import torch
import pandas as pd
import numpy as np
import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import logging
import warnings
import os
import gdown

warnings.filterwarnings('ignore')
torch_device = 'cpu'
logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.DEBUG)

logging.info('Searching tensors in /variables folder')

if os.path.exists('./variables/weights.pt') & os.path.exists('./variables/all_items.pt') & os.path.exists('./variables/df.pt'):
  pass
else:
  
  logging.info('Tensors not found, downloading from Google Drive..')

  gdown.download('https://drive.google.com/uc?export=download&id=1OqBpVHvm32kZh_47nYJ3D2EN6fTnbKIY', 'variables/weights.pt', quiet = True)
  gdown.download('https://drive.google.com/uc?export=download&id=1IzTUuk6blrdMgrbDGMvLPfRyoHtgKK69', 'variables/df.pt', quiet = True)
  gdown.download('https://drive.google.com/uc?export=download&id=1-CXOf71ubA51MBhPpXoa32qZlTrp0eGG', 'variables/all_items.pt', quiet = True)

weights = torch.load('variables/weights.pt', map_location = torch.device('cpu'))
df = torch.load('variables/df.pt', map_location = torch.device('cpu'))
all_items = torch.load('variables/all_items.pt', map_location = torch.device('cpu'))

logging.info('Done loading tensors in /variables folder')

logging.info('Downloading CLIP Tokenizer + Text Model...')

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

logging.info('Done downloading CLIP Tokenizer + Text Model')

def get_embeds(x):
  token = tokenizer(x, padding = "max_length", max_length = tokenizer.model_max_length, truncation = True, return_tensors = "pt")
  embed = text_encoder(token.input_ids)[0]
  return embed

def search_similar_items(idx: int, kind: str, all_items = all_items, weights = weights):
  if kind == 'clip':
    item = get_embeds(df.iloc[idx]['Description']).to(torch_device)
    scores = torch.einsum("abc,dbc->a", all_items.to(torch_device), item)
    return torch.argsort(scores, descending = True)[1:6]
  elif kind == 'dot_nn':
    item = weights[idx]
    scores = ((weights * torch.tensor(item)).sum(dim = 1, keepdim = True)).argsort(axis = 0)[0:5]
    return scores

def idx_to_desc(idx: int, df = df):
  if isinstance(idx, torch.Tensor):
    return df.iloc[idx.cpu().numpy().reshape(-1)]['Description']
  else:
    return df.iloc[idx]['Description']