import os
import torch
import pickle
import math
import torch
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
                 rating_col='rating', text_col=['template', 2], feature_col=['template', 0]):

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

        # Cal guardar les dades en el format original? Not sure, així almenys no l'hauré de carregar 2 cops
        with open(os.path.join(data_path, 'reviews.pickle'), 'rb') as f:
            self.original_data = pickle.load(f)

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
        rating = float(review[self.rating_col]) # it needs to be a float, because the predictions will be floats
        # as it would be weirder to only predict integers here
        text = self.seq2ids(self.get_maybe_internal(review, self.text_col))
        padded_text = sentence_format(text, self.text_length, self.pad, self.bos, self.eos)
        assert(len(padded_text) == self.text_length + 2) # here everything looks correct

        data_tuple = (user, item, rating, padded_text)
        if self.feature_col:
            # Si una feature codifica com a unkown vaya gràcia de feature xD
            feature = self.word_dict.word2idx.get(self.get_maybe_internal(review, self.feature_col), self.__unk)
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
    

    def get_maybe_internal(self, review, column):
        if type(column) == list:
            assert len(column) == 2
            first = review[column[0]]
            return first[column[1]]
        else:
            return review[column]


    # Inicialitzar les entitats de user, item; i el diccionari de words per comptar les més freqüents
    # per veure quines es descarten abans de convertir les dades a ids (pq inicialment no es pot saber)
    def initialize_entities(self):
        for review in self.original_data:
            self.user_dict.add_entity(review[self.user_col])
            self.item_dict.add_entity(review[self.item_col])

            text = self.get_maybe_internal(review, self.text_col)
            self.word_dict.add_sentence(text)

            # Faig algo amb la feature de moment? Might as well
            if self.feature_col:
                feature = self.get_maybe_internal(review, self.feature_col)
                self.word_dict.add_word(feature)

            rating = review[self.rating_col]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating


    def transform_data(self):
        transformed_data = []
        for review in self.original_data:
            transformed_data.append(self.transform_review(review))
        return transformed_data


    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]



class MySplitDataset:

    def __init__(self, data_path, data_length, index_dir, load_split=False,
                 train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42):

        self.data_path = data_path
        self.data_length = data_length
        self.index_dir = str(index_dir)

        self.names = ['train', 'validation', 'test']

        if load_split:
            assert os.path.exists(os.path.join(self.data_path, self.index_dir))
            self.load_split()
            lengths = [len(self.train), len(self.validation), len(self.test)]
            ratios = [length / data_length for length in lengths]
            self.train_ratio, self.validation_ratio, self.test_ratio = ratios
        else:
            self.train_ratio, self.validation_ratio, self.test_ratio = train_ratio, validation_ratio, test_ratio
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
        self.train, self.validation, self.test = split['train'], split['validation'], split['test']

    def save_split(self):
        split = {'train': self.train, 'validation': self.validation, 'test': self.test}
        for name in self.names:
            with open(f"{self.data_path}/{self.index_dir}/{name}.index", 'w') as f:
                f.write(' '.join(map(str, split[name])))

    def create_split(self, seed):
        assert math.isclose(self.train_ratio + self.validation_ratio + self.test_ratio, 1, rel_tol=1e-7)
        fixed_generator = torch.Generator().manual_seed(seed)
        split_generator = random_split(range(self.data_length), [self.train_ratio, self.validation_ratio, self.test_ratio],
                                       generator=fixed_generator)
        splits = [sorted(s) for s in split_generator] # sort just because it's simpler to read the file this way
        self.train, self.validation, self.test = splits
