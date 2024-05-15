import os
import torch
import math
import torch
import pandas as pd
import heapq
from torch.utils.data import Dataset, random_split
import pickle
import logging

from utils.peter import now_time
from tokenizer import load


# De fet si consideres que els historial d'alguna mena de columna t'aporta moltes coses i no són coses molt
# llargues, crec que considerar l'historial probably millori bastant els resultats de qualsevol cosa que facis,
# tot i que potser el model tardarà més en entrenar. Per exemple:
# - average rating till now
# - average dels last 10 ratings
# - most common summary

# Però de moment no afegiré més coses, simplement canviaré lo del template pq no tinc el template per tots,
# i posaré el summary no la review sencera, pq dona més informació en poques paraules, i les paraules crec
# que em limiten bastant. De moment la tasca de predir el context la deixaré de moment que predigui només
# el context del summary, i de moment no afegiré tot l'historial pq considero que és una mica fer trampes

# De fet fer predir el context també es podria tenir per cada item les paraules més utilitzades
# i.e. es podria tenir un:
# - user context
# - item context
# - user-item context

# i després l'explicació que utilitzi tot això, crec que ajudaria realment molt a generar millors explicacions
# sense haver de fer tantes èpoques

# I si inclús es considerés la diversitat a nivell de batch tmb es podria evitar una mica generar exactament
# el mateix per a tothom


# El Python l'iterator és l'object itself, i només hi pot haver un active iterator. Si vols multiples iterators
# independents, ho has de fer fent una còpia, però en general no es sol comprovar ni passa.
# Les propietats són @property


# podria tmb instal·lar torchtext, però com que és curt i és útil prefereix-ho tenir-jo jo manualment
class EntityDict:
    def __init__(self):
        self.entity_to_idx = {}
        self.entity_count = {}
        self.idx_to_entity = []
   
    def add_entity(self, entity): # entity can be anything hashable
        if entity not in self.entity_to_idx:
            self.entity_to_idx[entity] = len(self.idx_to_entity)
            self.idx_to_entity.append(entity)
            self.entity_count[entity] = 1
        else:
            self.entity_count[entity] += 1
        # return self.entity_to_idx[entity]

    def __len__(self):
        return len(self.idx_to_entity)
    
    @property
    def total_count(self):
        return sum(self.entity_count.values())
    
    # Deixo implementat pq és útil però crec que mai no voldré limitar les entitats així
    # En tot cas simplement hauries d'utilitzar un tokenitzador millor, no perdre vocabulari
    def keep_most_frequent(self, vocab_size, always_keep_entities=[]):
        if len(self) - len(always_keep_entities) > vocab_size:
            non_special_count = {k: v for k, v in self.entity_count.items() if k not in always_keep_entities}
            frequent_entities = heapq.nlargest(vocab_size, non_special_count, key=non_special_count.get)
            
            self.idx_to_entity = always_keep_entities + frequent_entities
            self.entity_to_idx = {t: i for i, t in enumerate(self.idx_to_entity)}
            self.entity_count = {k: self.entity_count[k] for k in self.idx_to_entity}
    

# entity = token (as string), idx = token id
class TokenDict(EntityDict):
    def __init__(self, special_tokens, max_tokens):
        super().__init__()
        self.special_tokens = special_tokens # bos, eos, pad, unk
        for token in special_tokens:
            self.add_entity(token)
        self.max_tokens = max_tokens
        self.bos, self.eos, self.pad, self.unk = [self.entity_to_idx[token] for token in special_tokens]

    def add_sentence(self, tokens):
        for token in tokens[:self.max_tokens]:
            self.add_entity(token)
    
    def encode(self, tokens):
        content = [self.entity_to_idx.get(token, self.unk) for token in tokens[:self.max_tokens]]
        return [self.bos, *content, self.eos] + [self.pad] * (self.max_tokens - len(content))
    
    def decode(self, token_ids, raw=False):
        if not raw:
            assert token_ids[0] == self.bos
            eos_idx = token_ids.index(self.eos)
            return [self.idx_to_entity[token_id] for token_id in token_ids[1:eos_idx]]
        
        return [self.idx_to_entity[token_id] for token_id in token_ids]
    
    
    def keep_most_frequent(self, vocab_size):
        super().keep_most_frequent(vocab_size, self.special_tokens)



# user, item, rating, text
class MyDataset(Dataset):

    def __init__(self, data_path, tokenizer, context_window):

        self.context_window = context_window
        # self.text_vocab_size = text_vocab_size

        # 1, load the users, items, ratings and text dataset
        original_data = pd.read_csv(data_path + '/reviews.csv')

        # 2, load the train/valid/test split doesn't need to be done here. It will be done when batching

        # 3, tokenize the text with a tokenizer
        original_data["tokenized_text"], self.tokenize, self.untokenize = load(data_path, tokenizer)

        # 4, transform users, items and text to ID's
        print(f'{now_time()}Creating entities and transforming data')

        print("Creating entities")
        self.user_dict = EntityDict()
        self.item_dict = EntityDict()
        original_data["user"].apply(self.user_dict.add_entity)
        original_data["item"].apply(self.item_dict.add_entity)

        special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>"]
        self.token_dict = TokenDict(special_tokens, self.context_window)
        original_data["tokenized_text"].apply(self.token_dict.add_sentence)

        # here the optative vocab size would be applied. Però crec que MAI no m'interessa limitar el vocabulari de res
        # en tot cas el que cal és un tokenitzador millor, o passar-li menys dades al model si no tens prou recursos
        print(f"There's {len(self.user_dict)} users")
        print(f"There's {len(self.item_dict)} items")
        print(f"There's {len(self.token_dict)} tokens")

        self.user_encode = lambda x: self.user_dict.entity_to_idx.get(x)
        self.item_encode = lambda x: self.item_dict.entity_to_idx.get(x)
        self.user_decode = lambda x: self.user_dict.idx_to_entity[x]
        self.item_decode = lambda x: self.item_dict.idx_to_entity[x]
        self.text_encode = lambda x: self.token_dict.encode(x)
        self.text_decode = lambda x, raw=False: self.token_dict.decode(x, raw)

        self.text_vectorize = lambda x: self.text_encode(self.tokenize(x))
        self.text_unvectorize = lambda x, raw=False: self.untokenize(self.text_decode(x, raw))

        self.users = torch.tensor(original_data["user"].apply(self.user_encode))
        self.items = torch.tensor(original_data["item"].apply(self.item_encode))
        self.texts = torch.tensor(original_data["tokenized_text"].apply(self.text_encode))
        
        self.ratings = torch.tensor(original_data["rating"], dtype=torch.float32) # sense especificar float32 peta el backwards
        self.max_rating = self.ratings.max()
        self.min_rating = self.ratings.min()


    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx], self.texts[idx]


# It doesn't load the dataset, just creates splits based on the indices of the length at the folder
class MySplitDataset:

    # could be cleaned up a bit
    def __init__(self, data_path, data_length, split_id, load_split=False,
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):

        self.split_dir = f"{data_path}/splits/{split_id}"
        self.names = ['train', 'valid', 'test']

        if load_split:
            assert os.path.exists(self.split_dir)
            self.load_split()
            lengths = [len(self.train), len(self.valid), len(self.test)]
            ratios = [length / data_length for length in lengths]
            self.train_ratio, self.valid_ratio, self.test_ratio = ratios
        else:
            self.train_ratio, self.valid_ratio, self.test_ratio = train_ratio, valid_ratio, test_ratio
            self.create_split(seed, data_length)
            assert not os.path.exists(self.split_dir)
            os.makedirs(self.split_dir)
            self.save_split()
    
    def load_split(self):
        split = {}
        for name in self.names:
            with open(f"{self.split_dir}/{name}.index", 'r') as f:
                split[name] = [int(x) for x in f.readline().split()]
        self.train, self.valid, self.test = split['train'], split['valid'], split['test']

    def save_split(self):
        split = {'train': self.train, 'valid': self.valid, 'test': self.test}
        for name in self.names:
            with open(f"{self.split_dir}/{name}.index", 'w') as f:
                f.write(' '.join(map(str, split[name])))

    def create_split(self, seed, data_length):
        assert math.isclose(self.train_ratio + self.valid_ratio + self.test_ratio, 1, rel_tol=1e-7)
        fixed_generator = torch.Generator().manual_seed(seed)
        split_generator = random_split(range(data_length), [self.train_ratio, self.valid_ratio, self.test_ratio],
                                       generator=fixed_generator)
        splits = [sorted(s) for s in split_generator] # sort just because it's simpler to read the file this way
        self.train, self.valid, self.test = splits


def setup_logger(name, log_file, stdout=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))    # Log to file
    if stdout:
        logger.addHandler(logging.StreamHandler())      # Log to stdout
    return logger


def move_to_device(content, device):
    return tuple(element.to(device) for element in content)
