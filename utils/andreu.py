import os
import torch
import math
import torch
import pandas as pd
import heapq
from torch.utils.data import Dataset, random_split
import pickle
import logging

from .peter import now_time


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


# no thing torchtext instal·lat, però si és tant simple ho puc fer jo
class EntityDict:
    def __init__(self):
        self.idx_to_entity = []
        self.entity_to_idx = {}
        self.entity_count = {}

    def add_entity(self, e):
        if e not in self.entity_to_idx:
            self.entity_to_idx[e] = len(self.idx_to_entity)
            self.idx_to_entity.append(e)
            self.entity_count[e] = 1
        else:
            self.entity_count[e] += 1

    def __len__(self):
        return len(self.idx_to_entity)
    
    @property
    def total_count(self):
        return sum(self.entity_count.values())
    

class TokenDictionary(EntityDict):
    def __init__(self, special_tokens):
        super().__init__()
        self.special_tokens = special_tokens
        for token in special_tokens:
            self.add_entity(token)

    def keep_most_frequent(self, vocab_size):
        if len(self) > vocab_size:
            non_special_count = {k: v for k, v in self.entity_count.items() if k not in self.special_tokens}
            frequent_tokens = heapq.nlargest(vocab_size - len(self.special_tokens), non_special_count, key=non_special_count.get)
            self.idx_to_entity = self.special_tokens + frequent_tokens # Es canvia l'indexing dels tokens
            self.entity_to_idx = {t: i for i, t in enumerate(self.idx_to_entity)}
            self.entity_count = {k: self.entity_count[k] for k in self.idx_to_entity}
    
    def add_sentence(self, sentence, tokenizer=None):
        if not tokenizer:
            for token in sentence.split(): # Naive white space tokenizer
                self.add_entity(token)
        else:
            for token in tokenizer(sentence):
                self.add_entity(token)


# user, item, rating, text
class MyDataset(Dataset):
    def __init__(self, data_path, text_max_words, vocab_size):

        self.data_path = data_path
        self.text_max_words = text_max_words
        self.vocab_size = vocab_size

        self.user_dict = EntityDict()
        self.item_dict = EntityDict()
        special_tokens = "<bos>", "<eos>", "<pad>", "<unk>"
        self.word_dict = TokenDictionary(special_tokens) # word_dict pq tokenitzo amb espais
        self.bos, self.eos, self.pad, self.unk = (self.word_dict.entity_to_idx[x] for x in special_tokens)

        # 0, carregar csv original
        print(f'{now_time()}Loading csv...')
        original_data = pd.read_csv(data_path + '/reviews.csv') # No atribut de classe pq no el serialitzi
        original_data["text"] = original_data["text"].fillna('nan') # some text might be literally "nan" as string

        # 1, tallar tot el text a un determinat nombre de paraules
        print(f'{now_time()}Cutting text')
        original_words = original_data["text"].str.split().apply(len).sum()
        original_data["text"] = original_data["text"].str.split().str[:self.text_max_words].str.join(' ')
        new_words = original_data["text"].str.split().apply(len).sum()
        lost_percentage = 100*(original_words - new_words)/original_words
        print(f"{now_time()}lost {lost_percentage:.2f}% of words")

        # 2, inicialitzar entitats
        print(f'{now_time()}Creating entities')
        original_data["user"].apply(self.user_dict.add_entity)
        original_data["item"].apply(self.item_dict.add_entity)
        original_data["text"].apply(self.word_dict.add_sentence)
        self.max_rating = original_data["rating"].max()
        self.min_rating = original_data["rating"].min()

        # 3, deixar només les paraules més freqüents
        old_count = self.word_dict.total_count
        print(f'{now_time()}Keeping only most frequent words')
        self.word_dict.keep_most_frequent(vocab_size)
        new_count = self.word_dict.total_count
        lost_percentage = 100*(old_count - new_count)/old_count
        print(f"{now_time()}lost {lost_percentage:.2f}% of words")

        # 4, transformar strings a ID's
        print(f'{now_time()}Transforming data')
        data = original_data.apply(self.transform_review, axis=1)
        self.users, self.items, self.ratings, self.texts = map(torch.tensor, zip(*data))
        print(f'{now_time()}Data ready')
    

    # review -> (user, item, rating, text)
    def transform_review(self, review):
        user = self.user_dict.entity_to_idx[review["user"]]
        item = self.item_dict.entity_to_idx[review["item"]]
        rating = float(review["rating"])
        text = self.tokenize_text(review["text"])
        return user, item, rating, text

    # the reverse
    def untransform_review(self, review):
        user = self.user_dict.idx_to_entity[review[0]]
        item = self.item_dict.idx_to_entity[review[1]]
        rating = review[2].item() # si no posava el item() hi havia un tensor
        text = self.untokenize_text(review[3])
        return {"user": user, "item": item, "rating": rating, "text": text}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx], self.texts[idx]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    @staticmethod
    def load_or_create_and_save(data_path, text_max_words, vocab_size):
        filename = f'{data_path}/MyDataset_{text_max_words}_{vocab_size}.pkl'
        if os.path.exists(filename):
            print("exists, loading it")
            return MyDataset.load(filename)
        else:
            print("doesn't exist, creating it")
            dataset = MyDataset(data_path, text_max_words, vocab_size)
            print("created, saving it")
            dataset.save(filename)
            print("saved")
            return dataset
    
    @staticmethod
    def create_override(data_path, text_max_words, vocab_size):
        filename = f'{data_path}/MyDataset_{text_max_words}_{vocab_size}.pkl'
        if os.path.exists(filename):
            print('will override the previous pkl')
        dataset = MyDataset(data_path, text_max_words, vocab_size)
        print("created, saving it")
        dataset.save(f'{data_path}/MyDataset_{text_max_words}_{vocab_size}.pkl')
        print("saved")
        return dataset
    
    def tokenize_text(self, text): # Naive white space tokenizer
        text_tokens = [self.word_dict.entity_to_idx.get(w, self.unk) for w in text.split()]
        output = [self.bos, *text_tokens, self.eos] + [self.pad] * (self.text_max_words - len(text_tokens)) 
        # freaking copilot it's stupid he put len(text) instead of len(text_tokens)
        if len(output) != 17:
            print('ERROR:', len(output))
            print(output)
        return output
    
    def untokenize_text(self, tokenized_text, wrong=False):
        if not wrong: # MUST have <bos> and <eos>
            assert tokenized_text[0] == self.bos
            eos_idx = None
            for idx, token in enumerate(tokenized_text):
                if token == self.eos:
                    eos_idx = idx
                    break
            else:
                assert(False)
            return ' '.join(self.word_dict.idx_to_entity[idx] for idx in tokenized_text[1:eos_idx])
        
        print('WARNING: untokenizing text without checking <bos> and <eos>')
        return ' '.join([self.word_dict.idx_to_entity[idx] for idx in tokenized_text])


# It doesn't load the dataset, just creates splits based on the indices of the length at the folder
class MySplitDataset:

    def __init__(self, data_path, data_length, index_dir, load_split=False,
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):

        self.data_path = data_path
        self.data_length = data_length
        self.index_dir = str(index_dir)

        self.names = ['train', 'valid', 'test']

        if load_split:
            assert os.path.exists(os.path.join(self.data_path, self.index_dir))
            self.load_split()
            lengths = [len(self.train), len(self.valid), len(self.test)]
            ratios = [length / data_length for length in lengths]
            self.train_ratio, self.valid_ratio, self.test_ratio = ratios
        else:
            self.train_ratio, self.valid_ratio, self.test_ratio = train_ratio, valid_ratio, test_ratio
            self.create_split(seed)
            split_folder = os.path.join(self.data_path, self.index_dir)
            assert not os.path.exists(split_folder)
            os.makedirs(split_folder)
            self.save_split()
    
    def load_split(self):
        split = {}
        for name in self.names:
            with open(f"{self.data_path}/{self.index_dir}/{name}.index", 'r') as f:
                split[name] = [int(x) for x in f.readline().split()]
        self.train, self.valid, self.test = split['train'], split['valid'], split['test']

    def save_split(self):
        split = {'train': self.train, 'valid': self.valid, 'test': self.test}
        for name in self.names:
            with open(f"{self.data_path}/{self.index_dir}/{name}.index", 'w') as f:
                f.write(' '.join(map(str, split[name])))

    def create_split(self, seed):
        assert math.isclose(self.train_ratio + self.valid_ratio + self.test_ratio, 1, rel_tol=1e-7)
        fixed_generator = torch.Generator().manual_seed(seed)
        split_generator = random_split(range(self.data_length), [self.train_ratio, self.valid_ratio, self.test_ratio],
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


def move_to_device(content, device, transpose_seq=True):

    user, item, rating, seq = content

    user = user.to(device)
    item = item.to(device)
    rating = rating.to(device)

    if transpose_seq:
        seq = seq.t()
    seq = seq.to(device)
    
    return user, item, rating, seq
