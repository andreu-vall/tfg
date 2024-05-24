import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch

# Aquest és el de torch. M'és útil el codi si el miro, si no és millor usar simplement
# l'oficial i ja. Però crec que precisament tenir aquestes coses m'ajuden molt per veure
# jo com funcionen els transformers aquests
# from torch.nn import TransformerEncoder
# en C:\Users\Andreu Vall\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\transformer.py

from utils.module import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, MLP


class PETER(nn.Module):
    # Crec que aquí hi ha masses arguments? O potser els hauria de mirar tots?
    def __init__(self, max_tokens, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, pad_idx, recommender_type,
                 bert_embeddings, token_dict_idx_to_entity):
        
        super(PETER, self).__init__()

        # print('emsize is', emsize) # 512 (transformers hidden space size)
        # print('nhead is', nhead) # 2
        # print('nhid is', nhid) # 2048
        # print('dropout is', dropout) # 0.2 # ojo el dropout fa que sigui no deteministic?
        # print('nlayers is', nlayers) # 2

        self.pad_idx = pad_idx
        self.emsize = emsize

        # PETER: nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above


        # ---------- EMBEDDINGS ----------

        # Sembla que amb els transformers es poden combinar diferents embeddings en diferents posicions.
        # La única restricció és que tots han de tenir la mateixa mida, i en aquest cas és emsize (512).
        # També tots els outputs dels transformers hidden tindran la mateixa mida, que també és emsize (512)

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
       
        # També es podria estudiar carregar els embeddings de tokens d'un altre model, però no crec que m'aporti res pel treball
        if bert_embeddings:
            self.token_embeddings = self.get_bert_embeddings(token_dict_idx_to_entity)
        else:
            self.token_embeddings = nn.Embedding(ntoken, emsize)

        self.hidden2token = nn.Linear(emsize, ntoken)

        # Wait hi haurien més possiblitats. Es podria fer servir igualment com el tenen en el PETER però usant-lo tmb
        # el rating autoregressivament pel model de la predicció del text.

        # Possiblement li hauria de donar un nom més descriptiu del que és cadascun. La diferència és que el recomanador
        # de PETER, per la part de text, només és un regularitzador que modifica lleugerament els embeddings de user i item.
        

        if recommender_type not in ['PETER', 'andreu']:
            raise ValueError(f"Recommender type {recommender_type} not recognized. Must be 'PETER' or 'andreu'")
        self.recommender_type = recommender_type

        if self.recommender_type == 'PETER':
            self.recommender = MLP(emsize, emsize)
            self.always_visible_input = 2 # user, item (en el PETER li deien src_len)
        else:
            self.recommender = MLP(2*emsize, emsize) # li passaré directament user i item embedding
            self.always_visible_input = 2 + 1 # user, item + rating (el PETER només l'augmentava amb les features)
        
        # text[:-1] and text[1:] size. Meaning +2 of <bos>, <eos> and -1 of the shift right for fully causal
        self.context_window = max_tokens + 1
        print(f'max_tokens is {max_tokens}, so context_window is {self.context_window}')

        self.attn_mask = PETER.generate_andreu_mask(self.always_visible_input, self.context_window)

        # Aquí és on es podria inicialitzar els embeddings de tokens pre-entrenats o d'un altre dataset
        self.init_weights(bert_embeddings)


    # És una attention_mask. Això vol dir que en els llocs on posis True, es bloquejarà l'atenció, per tant no es consderarà.
    # Per tant els False significa que són les posicions que sí que es tindran en compte (NO es bloquejaran)
    # rows: input, cols: ouput
    # True en mask[i, j] -> significa que el output token a la posició j NO pot veure el input token a la posició i
    # False en mask[i, j] -> significa que el output token a la posició j SÍ pot veure el input token a la posició i
    @staticmethod
    def generate_causal_mask(size):
        mask = torch.tril(torch.ones(size, size))
        mask = mask == 0
        return mask 
    
    # @staticmethod
    # def generate_peter_mask(size):
    #     mask = PETER.generate_causal_mask(size)
    #     mask[0, 1] = False
    #     return mask

    # always generate_peter_mask(size) == generate_andreu_mask(2, size-2),
    # so it's a generalization of the peter mask, to possibly have more always visible inputs
    @staticmethod
    def generate_andreu_mask(always_visible_input, context_window):
        mask_size = always_visible_input + context_window
        mask = torch.full((mask_size, mask_size), True) # block the text in the unused hidden states
        mask[:, :always_visible_input] = False # input part is always visible (if I want to use it in the unused states)
        causal_mask = PETER.generate_causal_mask(context_window) # usual causal mask, context_window x context_window
        mask[always_visible_input:, always_visible_input:] = causal_mask
        return mask


    # En què influeix la inicialització dels pesos? Usa la seed aquí si s'ha posat en un altre lloc?
    def init_weights(self, bert_embeddings):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        if not bert_embeddings: # ups crec que quan vaig provar els embeddings de bert no havia comentat això xD
            self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()


    def get_log_prob(self, hidden):
        token_prob = self.hidden2token(hidden)
        log_token_prob = func.log_softmax(token_prob, dim=-1)
        return log_token_prob # si es fa servir la CrossEntropyLoss no s'ha de fer el log_softmax


    def forward(self, user, item, rating, text, mode):
        
        assert mode in ['parallel', 'sequential'], "Mode must be either 'parallel' or 'sequential'"

        device = user.device
        batch_size = user.size(0)
        text_size = text.size(0)

        user_embed = self.user_embeddings(user)
        item_embed = self.item_embeddings(item)


        if self.recommender_type == 'andreu':

            x = torch.cat((user_embed, item_embed), 1) # uso directament els embeddings de user i item per predir el rating
            predicted_rating = self.recommender(x)
            
            if mode=='parallel': # En model paral·lel pel transformer s'usa el rating real
                assert rating is not None
                transformer_rating = rating
            else:
                transformer_rating = predicted_rating

            # Tinc certs dubtes sobre si fer això és més útil que només utilitzar el rating com un tasca per modificar els embeddings
            normalized_rating = (transformer_rating - 1) / 4 # rating by users is in [1, 5], predicted could be outside this
            rating_src = normalized_rating.view(1, batch_size, -1).expand(-1, -1, self.emsize) # l'he afegit jo de input, els de PETER
            # només els servia el rating per modificar lleugerament els embedding de user, item i ja està


        user_src = user_embed.unsqueeze(0)
        item_src = item_embed.unsqueeze(0)
        text_src = self.token_embeddings(text)

        if self.recommender_type=='andreu': # li poso el rating com a input del transformers també
            src = torch.cat([user_src, item_src, rating_src, text_src], 0)
        else:
            src = torch.cat([user_src, item_src, text_src], 0)

        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        my_size = self.always_visible_input + text_size # mida que ara mateix tinc disponible a mirar
        attn_mask = self.attn_mask[:my_size, :my_size].to(device)

        key_padding_mask = torch.full((batch_size, my_size), False, dtype=torch.bool, device=device)
        key_padding_mask[:, self.always_visible_input:] = text.t() == self.pad_idx
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)


        # Cal veure si realment la tasca de predicció de context és útil o no
        # Si no es realment molt útil preferia borrar-la, o com a mínim donar-li menys pes.
        # Pel que estic veient ara l'únic que serveix és per predir les paraules més comunes...
        # Si s'utilitza tant sols així la veritat és que sembla molt poc útil. Si realment vulgúes
        # que fos útil hauria de predir les paraules clau més important de la review.
        # Més aviat està predint només stop words...
        log_context_dis = self.get_log_prob(hidden[1])

        if self.recommender_type=='PETER':
            predicted_rating = self.recommender(hidden[0])

        if mode=='parallel':
            used_hidden = hidden[self.always_visible_input:] # es descodifiquen tots els tokens de text
        else:
            used_hidden = hidden[-1] # només cal descodificar l'últim token de text
        
        log_token_prob = self.get_log_prob(used_hidden)
        
        return log_token_prob, log_context_dis, predicted_rating, attns
    


    def get_bert_embeddings(self, token_dict_idx_to_entity):
        
        # Upsies les proves que havia fet crec que sobreescrivia els embeddings, per tant no tinc resultats d'això encara

        # Carrego els embeddings de només els tokens que apareixen en el meu dataset.
        # El vocabulari sencer del bert-base-uncased sembla ser de 30522 tokens.

        # Els embeddings de BERT són de 768, però els truncaré a 512 perquè és la mida
        # que utilitzaven sempre en el PETER i em va bé per comparar resultats. A més,
        # si augmentés el emsize a 768, se'm duplica el temps d'execució de tot...
        # Es podria fer amb PCA però em fa molta mandra fer-ho i no em sobra el temps.

        # només cal importar en aquest cas, així amb el codi normal no cal instal·lar els transformers
        from transformers import BertModel, BertTokenizer
        
        print('loading pretrained embeddings from Bert')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = model.get_input_embeddings()

        print(f'truncating embeddings size from {embeddings.weight.shape[1]} to {self.emsize}')
        vocab = torch.tensor(tokenizer.convert_tokens_to_ids(token_dict_idx_to_entity))
        truncated_embeddings = embeddings(vocab)[:, :self.emsize]

        return nn.Embedding.from_pretrained(truncated_embeddings)
    

