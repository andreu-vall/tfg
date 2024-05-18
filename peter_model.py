import math
import torch
import torch.nn as nn
import torch.nn.functional as func

# Aquest és el de torch. M'és útil el codi si el miro, si no és millor usar simplement
# l'oficial i ja. Però crec que precisament tenir aquestes coses m'ajuden molt per veure
# jo com funcionen els transformers aquests
# from torch.nn import TransformerEncoder
# en C:\Users\Andreu Vall\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\transformer.py

from utils.module import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, MLP, \
    generate_peter_mask, generate_square_subsequent_mask


# He de simplificar el codi i els arguments

class PETER(nn.Module):
    # Crec que aquí hi ha masses arguments?
    def __init__(self, peter_mask, src_len, tgt_len, nuser, nitem, ntoken, emsize,
                 nhead, nhid, nlayers, dropout, bos_idx, eos_idx, pad_idx, unk_idx):
        super(PETER, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size

        # # why am I only using 2 heads?
        # print('emsize is', emsize) # 512
        # print('nhead is', nhead) # 2
        # print('nhid is', nhid) # 2048
        # print('dropout is', dropout) # 0.2
        # print('nlayers is', nlayers) # 2

        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above


        # Sembla que amb els transformers es poden combinar diferents embeddings en diferents posicions
        # La única restricció és que tots han de tenir la mateixa mida, i en aquest cas és emsize (512)

        # També tots els outputs dels transformers hidden tindran la mateixa mida, que també és emsize (512)

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.token_embeddings = nn.Embedding(ntoken, emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.recommender = MLP(emsize)

        # should this work with indices or with entities??? i'm not sure
        # i think with indicies as the embeddings work with indices

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = emsize

        # Encara he de mirar lo de les màscares. Aviam ja no es dirà aquest fitxer peter.py

        # LES 2 COSES QUE S'UTILITZEN PER CREAR LA MÀSCARA: src_len i tgt_len
        print('src_len is', src_len)
        print('tgt_len is', tgt_len)

        #1234 tret un  + 1
        self.attn_mask = PETER.generate_andreu_mask(src_len + tgt_len) # el +1 és pq ells tenien els paràmetres mal?
        
        # if peter_mask: # Crec que hauria de començar ja a tocar les màscares d'atenció, i sobretot hauria ja
        #     # d'anar avançant la memòria, ni que siguin versions tontes en brut, perquè sinó arribaran les
        #     # pròximes reunions i no tindré res fet i m'estressaré i tot anirà malament. No cal que faci un
        #     # treball de 10, de fet és possible que ja no pugui aconseguir un treball de 10, però com a mínim
        #     # he de fer un treball que de moment no tinc res fet. De fet la nota no m'importa massa, però
        #     # hauria d'acabar el treball que és el que volen en la uni fer tenir tots els graus
        #     self.attn_mask = generate_peter_mask(src_len, tgt_len)
        #     print("s'ha creat la màscara ja, amb el generate_peter_mask")
        # else:
        #     assert False
        #     self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len)
        #     print("ups, era en square???")
        
        

        self.init_weights()

    @staticmethod
    def generate_andreu_mask(size):
        mask = torch.tril(torch.ones(size, size))
        mask = mask == 0
        mask[0, 1] = False # allow to attend for user and item
        return mask


    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
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

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
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

        # Entendre aquesta funció és entendre exactament que estan fent a PETER
        # Ja que he invertit tant temps en el PETER, òbviament hauria d'entendre això,
        # pq el meu TFG consistirà exactament en explicar això que se suposa que hauria
        # d'haver entès i dominat molt bé

        # Aquí s'agafa el deivce de les dades, però evidentment les dades i el model han d'estar en el mateix device

        device = user.device
        batch_size = user.size(0)
        # ojo he canviat la línia posterior a aquesta i crec que estic tenint problemes amb les transposicions en general

        # Sembla que la màscara depèn de quan text has generat ja i estàs passant-lo doncs pels paràmetres

        total_len = self.ui_len + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask

        # Totes les coses que venen de input pots atendre a elles

        # Sembla que l'atenció és exactament per les coses que ja hi ha generades fins ara,
        # i possiblement amb un sol step es generi totes les paraules indepdendentment del template
        # de fixed size a generar, l'únic que després pq hi hagi consistència entre el text generat
        # el que es va fent es una estratègia de decoding, on la més simple que usaven en PETER
        # és anar en cada step simplement generant una paraula més, la més probable fins on havies
        # generat fins ara amb una més la que afegeixes ara
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)

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
        
        # utilitza NOMÉS l'últim hidden token per genera 1 únic token més
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns
