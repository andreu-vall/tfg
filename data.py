import os
import torch
import math
import torch
import pandas as pd
import heapq
from torch.utils.data import Dataset, random_split, DataLoader, Subset
import sys
import logging

from utils.peter import now_time, root_mean_square_error, mean_absolute_error
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
        self.special_tokens = special_tokens # bos, eos, pad, unk, cut
        self.max_tokens = max_tokens
        for token in special_tokens:
            self.add_entity(token)
        
        self.bos, self.eos, self.pad, self.unk, self.cut = [self.entity_to_idx[token] for token in special_tokens]

        # Crec que pot ser interessant pel model distingir entre frases que realment han acabat <eos>
        # i les frases que les he tallat jo pq tots tinguin la mateixa longitud <cut>

        # self.bos -> serveix simplement per començar a construir el text a partir de algo
        # self.eos -> serveix per indicar que s'ha acabat la frase i no s'ha de generar ja res més
        # self.pad -> serveix només per fer que totes les frases tinguin exactament els mateixos tokens
        # self.unk -> significa que és un token desconegut no part del vocabulari
        # self.cut -> significa que s'ha tallat la frase pq era massa llarga, per diferenciar-ho del <eos> normal

    def add_sentence(self, tokens):
        self.add_entity(self.special_tokens[0]) # per si li passo els weights al text_criterion
        self.add_entity(self.special_tokens[1] if len(tokens) <= self.max_tokens else self.special_tokens[4])
        for token in tokens[:self.max_tokens]:
            self.add_entity(token)
    
    def encode(self, tokens):
        tokens_size = len(tokens)
        content = [self.entity_to_idx.get(token, self.unk) for token in tokens[:self.max_tokens]]
        if tokens_size <= self.max_tokens:
            return [self.bos, *content, self.eos] + [self.pad] * (self.max_tokens - tokens_size)
        else:
            return [self.bos, *content, self.cut]
    

    def decode(self, token_ids, mode):

        valid_modes = [
            'correct', # <bos> al principi i <eos> + padding o <cut> al final si era més llarga
            'sceptic', # <bos> al principi però no necessàriament <eos> o <cut>
            'literal'  # els caràcters especials no importen per res
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode}. Expected one of: {valid_modes}")
        
        if mode == 'literal':
            return [self.idx_to_entity[token_id] for token_id in token_ids]
        
        assert token_ids[0] == self.bos
        
        if self.eos in token_ids:
            termination_idx = token_ids.index(self.eos)
        
        elif token_ids[-1] == self.cut:
            termination_idx = len(token_ids)

        else:
            if mode == 'correct':
                raise ValueError("It was not correct")
            else:
                termination_idx = len(token_ids)

        return [self.idx_to_entity[token_id] for token_id in token_ids[1:termination_idx]]
    
    
    def keep_most_frequent(self, vocab_size):
        super().keep_most_frequent(vocab_size, self.special_tokens)



# user, item, rating, text
class MyDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_tokens, vocab_size=None):

        self.max_tokens = max_tokens # sense comptar el <bos> i <eos>

        # 1, load the users, items, ratings and text dataset
        original_data = pd.read_csv(data_path + '/reviews.csv')

        # 2, load the train/valid/test split doesn't need to be done here. It will be done when batching

        # Ara necessito un space tokenizer només

        # 3, tokenize the text with a tokenizer
        original_data["tokenized_text"], self.tokenize, self.untokenize = load(data_path, tokenizer, False)

        # 4, transform users, items and text to ID's

        self.user_dict = EntityDict()
        self.item_dict = EntityDict()
        original_data["user"].apply(self.user_dict.add_entity)
        original_data["item"].apply(self.item_dict.add_entity)

        special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<cut>"]
        self.token_dict = TokenDict(special_tokens, self.max_tokens)
        original_data["tokenized_text"].apply(self.token_dict.add_sentence)

        # here the optative vocab size would be applied. Però crec que MAI no m'interessa limitar el vocabulari de res
        # en tot cas el que cal és un tokenitzador millor, o passar-li menys dades al model si no tens prou recursos
        print(f"There's {len(self.user_dict)} users, {len(self.item_dict)} items and {len(self.token_dict)} tokens")

        if vocab_size is not None:
            self.token_dict.keep_most_frequent(vocab_size)
            print(f"Reduced to {len(self.token_dict)} tokens")

        self.user_encode = lambda x: self.user_dict.entity_to_idx.get(x)
        self.item_encode = lambda x: self.item_dict.entity_to_idx.get(x)
        self.user_decode = lambda x: self.user_dict.idx_to_entity[x]
        self.item_decode = lambda x: self.item_dict.idx_to_entity[x]
        self.text_encode = lambda x: self.token_dict.encode(x)
        self.text_decode = lambda x, mode: self.token_dict.decode(x, mode)

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

            # Per si necessito la seed amb la qual s'ha creat més tard
            with open(f"{self.split_dir}/seed.txt", 'w') as f:
                f.write(f"Seed: {seed}")

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
            with open(f"{self.split_dir}/{name}.index", 'w') as f: # same format that PETER used
                f.write(' '.join(map(str, split[name])))

    def create_split(self, seed, data_length):
        assert math.isclose(self.train_ratio + self.valid_ratio + self.test_ratio, 1, rel_tol=1e-7)
        fixed_generator = torch.Generator().manual_seed(seed)
        split_generator = random_split(range(data_length), [self.train_ratio, self.valid_ratio, self.test_ratio],
                                       generator=fixed_generator)
        splits = [sorted(s) for s in split_generator] # sort just because it's simpler to read the file this way
        self.train, self.valid, self.test = splits


# on hauria de posar els utils de andreu?

def setup_logger(name, log_file, stdout=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))    # Log to file
    if stdout:
        logger.addHandler(logging.StreamHandler())      # Log to stdout
    return logger


def record_execution(path):
    history_logger = setup_logger('history_logger', f'{path}/history.log')
    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")




def decode_batch_results(user, item, rating, text, predicted_rating, predicted_context, predicted_text, data:MyDataset):

    batch_size = user.size(0)

    decoded_user = [data.user_decode(u) for u in user]
    decoded_item = [data.item_decode(i) for i in item]
    decoded_text = [data.text_decode(list(t), mode='correct') for t in text] # needs a list to call the .index
    untokenized_text = [data.untokenize(t) for t in decoded_text]

    # context: predir les possibles paraules de tot el text en qualsevol ordre
    decoded_predicted_context = [data.text_decode(list(c), mode='literal') for c in predicted_context]
    untokenized_predicted_context = [data.untokenize(c) for c in decoded_predicted_context]

    decoded_predicted_text = [data.text_decode(list(t), mode='sceptic') for t in predicted_text]
    untokenized_predicted_text = [data.untokenize(t) for t in decoded_predicted_text]

    batch_results, batch_metrics = [], []
    for i in range(batch_size):
        batch_results.append({
            'user': decoded_user[i],
            'item': decoded_item[i],
            'predicted_rating': predicted_rating[i].item(), # cal l'item pq si no és tensor i no és serialitzable
            'real_rating': rating[i].item(),
            'predicted_context': untokenized_predicted_context[i], # en realitat n'hi ha més, simplement mostro els més alts
            # real_context: no té sentit pq simplement son les paraules més freqüents del text
            'predicted_text': untokenized_predicted_text[i],
            'real_text': untokenized_text[i]
            # Quan vaig copiar vectors directe vaig tenir un munt de problemes amb memòria GPU i memòria RAM
        })
        batch_metrics.append({
            'tokens_predicted_text': decoded_predicted_text[i],
            'tokens_real_text': decoded_text[i],
            'predicted_text': untokenized_predicted_text[i],
            'real_text': untokenized_text[i]
        })
    return batch_results, batch_metrics


def get_RMSE_MAE(results, max_rating, min_rating):

    predicted_ratings = [result['predicted_rating'] for result in results]
    real_ratings = [result['real_rating'] for result in results]
    real_predicted_rating = [(r, p) for (r, p) in zip(real_ratings, predicted_ratings)]

    RMSE = root_mean_square_error(real_predicted_rating, max_rating, min_rating)
    MAE = mean_absolute_error(real_predicted_rating, max_rating, min_rating)
    if torch.is_tensor(MAE): # a vegades és tensor i a vegades no
        MAE = MAE.item()

    return RMSE, MAE
    



def get_batch_for_test_purposes(data_path='data/amz-beauty-review', split_id='split_id_1',
                                tokenizer='tokenizer-bert-base-uncased', max_tokens=10, batch_size=128):
    
    data = MyDataset(data_path, tokenizer, max_tokens)
    split = MySplitDataset(data_path, len(data), split_id, load_split=True)
    train_data = Subset(data, split.train)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    first_batch = next(iter(train_dataloader))
    return first_batch, data

