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
    def __init__(self, src_len, tgt_len, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, pad_idx):
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
        self.recommender = MLP(emsize)

        # quina diferència hi ha amb ui_len???
        # src_len, s'usa per: crear màscara atenció (de src_len + tgt_len), en el predict_seq per saber a partir de on es descodificaran tokens
        # tgt_len, s'usa per: crear màscara atenció (de src_len + tgt_len) i ja està!!!!!
        # LOL pràcticament només li crees la màscara d'atenció i el torch transformers ja t'ho fan tot xD

        # should this work with indices or with entities??? i'm not sure
        # i think with indicies as the embeddings work with indices

        self.ui_len = 2
        self.src_len = src_len # source_length, és sempre 2 pq el que tinc sempre (source) és 2: user i item
        # tgt_len target_length, la longitud del que vull predir, només ho usen per crear la màscara i ja!
        self.pad_idx = pad_idx
        self.emsize = emsize

        # Encara he de mirar lo de les màscares. Aviam ja no es dirà aquest fitxer peter.py

        # LES 2 COSES QUE S'UTILITZEN PER CREAR LA MÀSCARA: src_len i tgt_len
        print('src_len is', src_len) # 2 always
        print('tgt_len is', tgt_len) # 6 (context_window +1)

        self.attn_mask = PETER.generate_peter_mask(src_len + tgt_len) # el +1 és pq ells tenien els paràmetres mal?

        self.init_weights() # Aquí és on es podria inicialitzar els embeddings de tokens pre-entrenats
    

    @staticmethod
    def generate_causal_mask(size):
        mask = torch.tril(torch.ones(size, size)) # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL
        mask = mask == 0 # triu enlloc de tril m'ha fet perdre més de 1 hora inútilment!!!!
        return mask # maleït copilot!!!

    @staticmethod
    def generate_peter_mask(size):
        mask = PETER.generate_causal_mask(size)
        mask[0, 1] = False # allow the first hidden position to attend both user and item for rating prediction
        return mask


    def init_weights(self):
        initrange = 0.1 # aquí es podria plantejar de inicar els token embeddings pre-entrenats d'un altre model
        self.user_embeddings.weight.data.uniform_(-initrange, initrange) # o inclús del mateix entrenat per un altre dataset
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        rating = self.recommender(hidden[0])  # (batch_size,)
        return rating

    # en realitat aquesta funció és pràcticament el mateix. La única diferència és que en la 2a no importa tota la
    # resta i només vols el que et dona la última hidden, la qual he vist que era de mida variable
    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        # print('generate_token, right now hidden has shape', hidden.shape) # [3, 128, 512] oh it do seems to be adapted?
        # # [?, batch_size, emsize]
        # assert False
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob


    def forward(self, user, item, text, seq_prediction=True, context_prediction=True, rating_prediction=True):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''

        device = user.device
        batch_size = user.size(0)

        # self.ui_len = 2
        # text.size(0) mira quants tokens tens de l'input ja generats
        # per tant total_len és la suma de les coses pots mirar en qualsevol punt (perquè més endvant no tindria sentit mirar si no tens)
        total_len = self.ui_len + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask

        # La màscara d'atenció és per dir-li a totes les coses que hi ha generades fins ara, a quines pot atendre en cada punt
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)

        # crec que la key_padding_mask bàscicament serviex per dir-li que no es fixi en els paddings per fer cap càlcul?
        left = torch.zeros(batch_size, self.ui_len).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        # src és molt senzill, és posar-li lo de la posició i canviar els ID's per els embeddings,
        # on al principi són coses aleatòries però es van entrenant junt amb el model per intentar
        # minimitzar la funció de loss que vulguis definir, la qual el torch va guardant-se les
        # coses i steps i fa la derivada automàticament, niceee
        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize) # ups havia borrat aquest línia
        # sense volguer, ojo que no me l'hagi carregat que no crec pq era molt senzilla
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        w_src = self.token_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        # Aquest és el primer pas clau de tots, aplicar el transformer a tot arreu. Perquè totes les altres coses
        # les treuen a partir d'aquest encoding produït pel transformer. Sembla que és el tipus de transformer
        # que codifica, i.e. el encoder only, no el encoder + decoder
        # CREC QUE HE DE CANVIAR ALGO DE AQUÍ. I JA ESTÀ BÉ PQ SEGUEIXO APRENENT I AVANÇANT
        # ARA JA HE DE MIRAR LO DE LA ATTN_MASK YAY. L'HE DE PROBABLY AMPLIAR EN 1?
        # w_src ha augmentat la mida en 1
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)
        # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)

        # Això és el més important de tot el model. Entendre aquesta línia és entendre els transformers
        # i en què estan basats tots els avenços que s'estan fent en LLM i altres coses similars


        # this code is really ugly ngl
        # i tampoc s'entén del tot exactament el que estan fent

        # train, test: model(user, item, text)
        # -> Li p

        # generate first token: model(user, item, text, False) (és el seq_prediction)
        # generate an additional token: model(user, item, text, False, False, False) seq_predction, context_prediction, rating_prediction

        # per mi no acaba de tenir sentit tot això

        if rating_prediction:
            rating = self.predict_rating(hidden)  # (batch_size,)
        else:
            rating = None
        
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        
        # usa tots els hidden alhora i descodifica totes les paraules en paral·lel. Per tant, si es descodifiqués
        # totes les paraules inidivualment probablement no tindrien coherència juntes, ja que s'han generat de
        # manera independent
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)

        else: # La qüestió és que la mida de hidden va canviant!
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns
