import math
import torch
import torch.nn as nn
import torch.nn.functional as func

# Aquest és el de torch. M'és útil el codi si el miro, si no és millor usar simplement
# l'oficial i ja. Però crec que precisament tenir aquestes coses m'ajuden molt per veure
# jo com funcionen els transformers aquests
# from torch.nn import TransformerEncoder
# en C:\Users\Andreu Vall\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\transformer.py

from utils.module import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, MLP


class PETER(nn.Module):
    # Crec que aquí hi ha masses arguments? O potser els hauria de mirar tots?
    def __init__(self, max_tokens, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, pad_idx, recommender_type):
        super(PETER, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)

        # print('emsize is', emsize) # 512 (transformers hidden space size)
        # print('nhead is', nhead) # 2
        # print('nhid is', nhid) # 2048
        # print('dropout is', dropout) # 0.2 # ojo el dropout fa que sigui no deteministic?
        # print('nlayers is', nlayers) # 2

        # PETER: nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above

        # Sembla que amb els transformers es poden combinar diferents embeddings en diferents posicions
        # La única restricció és que tots han de tenir la mateixa mida, i en aquest cas és emsize (512)
        # També tots els outputs dels transformers hidden tindran la mateixa mida, que també és emsize (512)

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.token_embeddings = nn.Embedding(ntoken, emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)

        if recommender_type not in ['PETER', 'andreu']:
            raise ValueError(f"Recommender type {recommender_type} not recognized. Must be 'PETER' or 'andreu'")
        self.recommender_type = recommender_type

        if self.recommender_type == 'PETER':
            self.recommender = MLP(emsize, emsize)
            self.always_visible_input = 2 # user, item
        else:
            self.recommender = MLP(2*emsize, emsize) # li passaré directament user i item embedding
            self.always_visible_input = 2 + 1 # user, item + rating (el PETER només l'augmentava amb les features)

        self.pad_idx = pad_idx
        self.emsize = emsize
        
        # text[:-1] and text[1:] size. Meaning +2 of <bos>, <eos> and -1 of the shift right for fully causal
        self.context_window = max_tokens + 1
        print(f'max_tokens is {max_tokens}, so context_window is {self.context_window}')

        self.attn_mask = PETER.generate_andreu_mask(self.always_visible_input, self.context_window)

        self.init_weights() # Aquí és on es podria inicialitzar els embeddings de tokens pre-entrenats o d'un altre dataset


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


    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
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

