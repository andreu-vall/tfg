import math
import torch
import torch.nn as nn
import torch.nn.functional as func

# Sembla que havia desgraciat algo somehow en en peter_model

# Aquest és el de torch. M'és útil el codi si el miro, si no és millor usar simplement
# l'oficial i ja. Però crec que precisament tenir aquestes coses m'ajuden molt per veure
# jo com funcionen els transformers aquests
# from torch.nn import TransformerEncoder
# en C:\Users\Andreu Vall\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\transformer.py

from utils.module import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, MLP


# He de simplificar el codi i els arguments. Ecara ho he de fer
# De moment només havia mirat lo de la màscara

class PETER(nn.Module):
    # Crec que aquí hi ha masses arguments?
    def __init__(self, max_tokens, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, pad_idx):
        super(PETER, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size


        # ojo el dropout fa que sigui no deteministic?

        # # why am I only using 2 heads?
        # print('emsize is', emsize) # 512
        # print('nhead is', nhead) # 2
        # print('nhid is', nhid) # 2048
        # print('dropout is', dropout) # 0.2
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
        self.recommender = MLP(emsize, emsize) # old: MLP(emsize)

        self.pad_idx = pad_idx
        self.emsize = emsize

        self.always_visible_input = 2 + 1 # user, item + rating
        
        # text[:-1] and text[1:] size. Meaning +2 of <bos>, <eos> and -1 of the shift right for fully causal
        self.context_window = max_tokens + 1
        print(f'max_tokens is {max_tokens}, so context_window is {self.context_window}')

        self.attn_mask = PETER.generate_andreu_mask(self.always_visible_input, self.context_window)

        self.init_weights() # Aquí és on es podria inicialitzar els embeddings de tokens pre-entrenats o d'un altre dataset


    # És una attention_mask. Això vol dir que en els llocs on posis True, es bloquejarà l'atenció, per tant no es consderarà
    # Per tant els False significa que són les posicions que sí que es tinran en compte
    # rows: input, cols: ouput

    # True en mask[i, j] -> significa que el output token a la posició j NO pot veure el input token a la posició i
    # False en mask[i, j] -> significa que el output token a la posició j SÍ pot veure el input token a la posició i
    @staticmethod
    def generate_causal_mask(size):
        mask = torch.tril(torch.ones(size, size)) # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL
        mask = mask == 0 # triu enlloc de tril del copilot m'ha fet perdre més de 2 hores inútilment!!!!
        return mask # hauria de vigilar més amb les coses omplertes pel copilot, i seguir provar que tot funcioni més sovint
    # i sobretot abans de posar-me a fer canvis molt grans que em poden trencar tot i que després no tindré ni idea d'on ve l'error
    
    # @staticmethod
    # def generate_peter_mask(size):
    #     mask = PETER.generate_causal_mask(size)
    #     mask[0, 1] = False
    #     return mask

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

    # ara ja podré comparar si tenir això realment aporta gaire. Si fos molt útil, potser valdria la pena
    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[0]) # podria usar qualsevol dels 3 primers que són inútils
        # crec que l'únic pel que serveix aquesta tasca és per modificar els embeddings per predir més les paraules comunes?
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    # el recomanador també podria comparar les 2 versions
    # def predict_rating(self, hidden):
    #     rating = self.recommender(hidden[0])  # (batch_size,)
    #     return rating

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.always_visible_input:])
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob


    def forward(self, user, item, rating, text, mode):
        
        assert mode in ['parallel', 'sequential'], "Mode must be either 'parallel' or 'sequential'"

        device = user.device
        batch_size = user.size(0)
        text_size = text.size(0)

        user_embed = self.user_embeddings(user)
        item_embed = self.item_embeddings(item)

        predicted_rating = self.recommender(user_embed, item_embed)
        
        # En model paral·lel pel transformer s'usa el rating real
        if mode=='parallel':
            assert rating is not None
            transformer_rating = rating
        else:
            transformer_rating = predicted_rating
        

        # Cal veure si realment la tasca de predicció de context és útil o no
        # Si no es realment molt útil preferia borrar-la, o com a mínim donar-li menys pes.
        # Pel que estic veient ara l'únic que serveix és per predir les paraules més comunes...
        # Si s'utilitza tant sols així la veritat és que sembla molt poc útil. Si realment vulgúes
        # que fos útil hauria de predir les paraules clau més important de la review

        # ----------------------- El transformer -----------------------

        u_src = user_embed.unsqueeze(0)
        i_src = item_embed.unsqueeze(0)
        r_src = transformer_rating.view(1, batch_size, -1).expand(-1, -1, self.emsize)
        w_src = self.token_embeddings(text)

        src = torch.cat([u_src, i_src, r_src, w_src], 0)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        my_size = self.always_visible_input + text_size # mida que ara mateix tinc disponible a mirar
        attn_mask = self.attn_mask[:my_size, :my_size].to(device)

        key_padding_mask = torch.full((batch_size, my_size), False, dtype=torch.bool, device=device)
        key_padding_mask[:, self.always_visible_input:] = text.t() == self.pad_idx

        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)

        # encara he de veure si el context realment és útil per algo. Ara més aviat està predint només stop words...
        log_context_dis = self.predict_context(hidden)

        # És molt similar, l'única diferència és que el predict_seq ho fa per tots i el generate_token només per l'últim
        if mode=='parallel':
            log_word_prob = self.predict_seq(hidden)
        else:
            log_word_prob = self.generate_token(hidden)
        
        return log_word_prob, log_context_dis, predicted_rating, attns

