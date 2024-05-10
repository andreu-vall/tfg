import os
import torch
import pickle
import math
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split

from .peter import WordDictionary, EntityDictionary, sentence_format, ids2tokens


# Comentaris de l'antic codi:
# Wait l'Alejandro va començar a usar la REV_COL la review sencera enlloc de predir només el template!
# Això és molt més pro! Per tant els resultats de l'Alejandro potser tenen més valor dels de PETER,
# tot i que potser es va complicar una mica introduint tot l'historial. Potser afegir tot l'historial
# té sentit per coses curtes, com l'historial de notes, maybe inclús l'average que has donat till now,
# també l'historial de features de les quals has comentat, pq és possible que repeteixis features en
# nous comentaris, inclús es podria considerar l'historial del item en altres reviews, tot i que fer-ho
# manualment és una mica de feina i en principi més lleig que si no ho fas. Més senzill per exemple seria
# l'average till now de notes donades per l'usuari, seria in fact quite useful saber-lo a l'hora de predir
# la nota per exemple i no serien molts valors si agafessis per exemple les últimes 10 notes

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


class MyDataset(Dataset):
    # Yikes si trec d'aquí els paràmetres per defecte ho hauria de canviar a tot arreu, a bit of a pain
    def __init__(self, data_path, text_length, vocab_size, user_col='user', item_col='item',
                 rating_col='rating', text_col='sentence', feature_col='feature'):
        
        # LOL l'he fet aquesta classe tant bé que funciona tant pel dataset de PETER com pel de Amazon?
        # I'm a boss ngl. Calm down he canviat simplement entre 2 datasets del PETER, no he canviat encara
        # al dataset d'Amazon

        # Crec que és millor guardar la data transformada a disc un cop feta per no haver-la de fer
        # cada cop que executo el model per fer proves ràndom. Així acceleria una part de totes
        # les proves que faci a partir d'ara amb el mateix dataset

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.text_col = text_col
        self.feature_col = feature_col

        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')

        # Cal guardar les dades en el format original
        # ? Not sure, així almenys no l'hauré de carregar 2 cops

        # with open(os.path.join(data_path, 'reviews.pickle'), 'rb') as f:
        #     self.original_data = pickle.load(f)

        # He canviat el source
        self.original_data = pd.read_csv(data_path + '/reviews.csv')

        # 1
        self.initialize_entities()

        # 2
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set() # ara mateix no faig res amb el feature_set. Is it even rellevant?

        self.text_length = text_length
        self.bos = self.word_dict.word2idx['<bos>']
        self.eos = self.word_dict.word2idx['<eos>']
        self.pad = self.word_dict.word2idx['<pad>']

        # 3
        self.transformed_data = self.transform_data() # 3

        # En aquesta classe NO faig el train/valid/test split, ho faig en una altra

        # 1. Llegir tot el dataset i incialitzar els diccionaris de entitats
        # 2. Escollir un num de words de vocabulary i retallar tota la resta
        # 3. Fer la conversió dels ID's originals als nous
        # 4. Permetre la conversió back dels ID's nous als originals, sobretot per de cara
        #    a quan hagi de descodificar paraules noves


        # Not completely sure si hauria de guardar les dades originals
    
        # Cal fer el padding del text ja aquí, si no el torch tindrà problemes
    
    # review -> (user, item, rating, text, [feature])
    def transform_review(self, review):
        user = self.user_dict.entity2idx[review[self.user_col]]
        item = self.item_dict.entity2idx[review[self.item_col]]
        rating = float(review[self.rating_col])     # it needs to be a float so that torch can predict a float
        text = self.seq2ids(review[self.text_col])  # and compute a loss that's for floats not integers
        padded_text = sentence_format(text, self.text_length, self.pad, self.bos, self.eos)
        # setic fent aquí el padding del text
        assert(len(padded_text) == self.text_length + 2) # here everything looks correct

        data_tuple = (user, item, rating, padded_text)
        if self.feature_col:
            # Si una feature codifica com a unkown vaya gràcia de feature xD
            feature = self.word_dict.word2idx.get(review[self.feature_col], self.__unk)
            #feature = self.word_dict.word2idx.get(self.get_maybe_internal(review, self.feature_col), self.__unk)
            data_tuple += (feature,)
            
            self.feature_set.add(feature) # crec que aquí encara que sigui unkown funcionaria

            # potser hauria de posar coses en el feature_set, que després l'utilitzaven des de fora en el test
            # per donar algunes mètirques especifiques de les features i del coverage de features
        
        return data_tuple
    
    # the reverse
    def untransform_review(self, review):
        user = self.user_dict.idx2entity[review[0]]
        item = self.item_dict.idx2entity[review[1]]
        rating = review[2].item()
        tokens = ids2tokens(review[3], self.word_dict.word2idx, self.word_dict.idx2word)
        assert tokens[0] == '<bos>' # and tokens[-1] == '<eos>' # no necessàriament hi ha el <eos>!!!
        
        text = ' '.join(tokens[1:])

        data_dic = {'user': user, 'item': item, 'rating': rating, 'text': text}
        if self.feature_col:
            feature = self.word_dict.idx2word[review[4]]
            data_dic['feature'] = feature
        
        return data_dic
    

    # Ara ja no em transposa ningú el text, és així de simple
    def untransform_batch(self, batch):
        return [self.untransform_review(items) for items in zip(*batch)]

    
    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        return self.transformed_data[idx]
    

    # def get_maybe_internal(self, review, column):
    #     if type(column) == list:
    #         assert len(column) >= 2 # no té sentit una llista si només és 1?
    #         if column[0] not in review: # Pel P5 pot ser que la sentence no aparegui en la review
    #             return None
    #         first = review[column[0]]
    #         second = first[column[1]]
    #         if len(column) == 2:
    #             return second
    #         else:
    #             assert len(column) == 3
    #             third = second[column[2]]
    #             return third
    #     else:
    #         return review[column]


    # Inicialitzar les entitats de user, item; i el diccionari de words per comptar les més freqüents
    # per veure quines es descarten abans de convertir les dades a ids (pq inicialment no es pot saber)
    def initialize_entities(self):
        for _, review in self.original_data.iterrows():
            
            self.user_dict.add_entity(review[self.user_col])
            self.item_dict.add_entity(review[self.item_col])
            self.word_dict.add_sentence(review[self.text_col])
            if self.feature_col: # Faig algo amb la feature de moment? Might as well
                self.word_dict.add_word(review[self.feature_col])

            rating = review[self.rating_col]
            if rating > self.max_rating:
                self.max_rating = rating
            elif rating < self.min_rating:
                self.min_rating = rating


    def transform_data(self):
        transformed_data = []
        for _, review in self.original_data.iterrows():
            transformed_data.append(self.transform_review(review))
        return transformed_data


    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]


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
