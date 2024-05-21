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
    def __init__(self, context_window, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, pad_idx):
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

        # quina diferència hi ha amb ui_len???
        # src_len, s'usa per: crear màscara atenció (de src_len + tgt_len), en el predict_seq per saber a partir de on es descodificaran tokens
        # tgt_len, s'usa per: crear màscara atenció (de src_len + tgt_len) i ja està!!!!!
        # LOL pràcticament només li crees la màscara d'atenció i el torch transformers ja t'ho fan tot xD

        # should this work with indices or with entities??? i'm not sure
        # i think with indicies as the embeddings work with indices

        self.pad_idx = pad_idx
        self.emsize = emsize

        # Ara amb això puc teòricament modificar-ho com vulgui. Puc canviar el input i/o el output
        # self.attn_mask = PETER.generate_andreu_mask(2, 2, context_window) # aquest funciona
        # PETER: generate_peter_mask(2 + context_window)

        # modificació 1: considerar el rating com a input enlloc de output
        # AIXÒ SEMPRE EM FA UNA MÀSCARA QUADRADA
        self.input_only_size = 2 + 1 # user, item + rating
        self.output_only_size = 2 - 1 # rating, context - rating
        self.context_window = context_window # text[:-1] and text[1:]
        
        self.falsos_hidden = 2
        self.attn_mask = PETER.generate_andreu_mask(self.input_only_size, self.input_only_size, self.context_window)

        self.init_weights() # Aquí és on es podria inicialitzar els embeddings de tokens pre-entrenats


    @staticmethod
    def generate_causal_mask(size):
        mask = torch.tril(torch.ones(size, size)) # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL
        mask = mask == 0 # triu enlloc de tril m'ha fet perdre més de 2 hores inútilment!!!!
        return mask # maleït copilot!!!
    
    @staticmethod
    def generate_peter_mask(size):
        mask = PETER.generate_causal_mask(size)
        mask[0, 1] = False
        return mask

    @staticmethod
    def generate_andreu_mask(input_only_size, output_only_size, context_window):
        mask = torch.full((input_only_size + context_window, output_only_size + context_window), True) # block everything
        mask[:, :input_only_size] = False # input part is always visible
        causal_mask = PETER.generate_causal_mask(context_window) # usual causal mask, context_window x context_window
        mask[input_only_size:, output_only_size:] = causal_mask
        return mask


    # input size: 8
    # context_window: 5
    # extra: user, item, context

    # output size: 8
    # hidden[0]: rating
    # hidden[1]: context
    # hidden[2:7]: tokens 2-6 (5) N'HI HA 1 EXTRA?
    # hidden[7] WTF is it?
    


    def init_weights(self):
        initrange = 0.1 # aquí es podria plantejar de inicar els token embeddings pre-entrenats d'un altre model
        self.user_embeddings.weight.data.uniform_(-initrange, initrange) # o inclús del mateix entrenat per un altre dataset
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[self.falsos_hidden])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    # def predict_rating(self, hidden):
    #     rating = self.recommender(hidden[0])  # (batch_size,)
    #     return rating

    # en realitat aquesta funció és pràcticament el mateix. La única diferència és que en la 2a no importa tota la
    # resta i només vols el que et dona la última hidden, la qual he vist que era de mida variable
    def predict_seq(self, hidden):
        #print('predict_seq, right now hidden has shape', hidden.shape) # [6, 128, 512]
        word_prob = self.hidden2token(hidden[self.falsos_hidden+1:])
        #print('word_prob shape is', word_prob.shape) # [4, 128, 8631]
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob


    def generate_token(self, hidden):
        # print('generate_token, right now hidden has shape', hidden.shape) # [3, 128, 512] oh it do seems to be adapted?
        # # [?, batch_size, emsize]
        # assert False
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob


    # PETER
    # input: user, item, text (minus last token) [1 + 1 + context_window - 1]
    # output: rating, context, text (shifted right, minus first token) [1 + 1 + context_window - 1]
    # user s'alinea amb rating, item amb context i per això la màscara és quadrada
    # Per tant [context_window + 1, context_window + 1] de attn_mask

    # Step intermig: borrar la predicció del context (la hidden que la prediu, per tant modificant 1 dimensió del output).
    # Perquè quedi tot ben alineat potser tb hauria de borrar user, item?

    # Andreu
    # Per tant hi ha 2 extres que només tenen sentit en el input: user i item

    # input: user, item, rating, context, text (minus last token) [1 + 1 + 1 + context_window + 1 - 1]
    # output: rating, context, text (shifted right, minus first token) [1 + 1 + context_window + 1 - 1]
    # Per taint [context_window + 3, context_window + 1] de attn_mask

    # Podria treure 1 de input i un de output i tenir la màscara de mida 1, 1 menys per exemple primer de tot?
    # O 

    # Per tant el step 1 serà deixar de predir rating i context i predir directament el text

    def forward(self, user, item, rating, text, mode):
        device = user.device
        batch_size = user.size(0)
        text_size = text.size(0)

        user_embed = self.user_embeddings(user)
        item_embed = self.item_embeddings(item)

        # Sempre en qualsevol cas primer faré la predicció de rating 
        # LOL donar tot 5's només és una loss de 1.9793450832366943
        # i donant tot 5's dona 1.3801511526107788

        # WTF amb 1 època ha augmentat el rating_loss???? Quin sentit té, es contraposa usar els embeddings
        # en un lloc i un altre? Kinda estrany que vagi a pitjor, no?
        predicted_rating = self.recommender(user_embed, item_embed)
        #predicted_rating = torch.full((batch_size,), 4, dtype=torch.float32).to(device)

        # La diferència és només que en mode parallel posaré de input del transformer el rating real
        transformer_rating = rating if mode=='parallel' else predicted_rating

        value = self.falsos_hidden + 1 + text_size

        attn_mask = self.attn_mask[:value, :value].to(device)

        r_src = transformer_rating.view(1, batch_size, -1).expand(-1, -1, 512).to(device) # suggerit by copilot (expand it, repeat gasta memòria)
        w_src = self.token_embeddings(text)
        
        u_src = user_embed.unsqueeze(0)
        i_src = item_embed.unsqueeze(0)

        src = torch.cat([u_src, i_src, r_src, w_src], 0)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        # ara el falta la key_padding_mask, yikes. De moment no la poso
        # Ara segurament ja seria hora de fer-ho!

        left = torch.zeros(batch_size, 2).bool().to(device)
        right = text.t() == self.pad_idx
        key_padding_mask = torch.cat([left, right], 1)

        # lo important era el: value = self.falsos_hidden (2) + 1 + text_size
        # simplement és marcar en quines posicions hi ha padding pq no influeixin en el càlcul del transformer


        hidden, attns = self.transformer_encoder(src, attn_mask)

        log_context_dis = self.predict_context(hidden)

        if mode=='parallel':
            log_word_prob = self.predict_seq(hidden)
        elif mode=='sequential':
            log_word_prob = self.generate_token(hidden)
        else:
            raise ValueError('mode should be either parallel or sequential')
        
        return log_word_prob, log_context_dis, predicted_rating, attns

        

    def forward2(self, user, item, text, seq_prediction=True, context_prediction=True, rating_prediction=True):

        device = user.device
        batch_size = user.size(0)
        
        total_len = 2 + text.size(0)

        # ara he de triar la màscara diferent

        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)

        left = torch.zeros(batch_size, 2).bool().to(device)
        right = text.t() == self.pad_idx
        key_padding_mask = torch.cat([left, right], 1)



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

        # attn_mask, 1=block the attention
        # key_padding_mask, 1=block the attention, due to padding reasons

        # Documentació de transformer_encoder:
        # src: the sequence to the encoder (required).
        # mask: the mask for the src sequence (optional).
        # src_key_padding_mask: the mask for the src keys per batch (optional).

        # en la traducció o sumarització evidentment pots mirar a tot arreu de l'input
        # però en la tasca de generació de text només pots mirar a les paraules anteriors,
        # perquè quan generis nou text només tindràs les posicions anteriors

        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)


        if rating_prediction:
            rating = self.predict_rating(hidden)  # (batch_size,)
        else:
            rating = None

        # log_context_dis = None

        # He comentat fora la tasca de predir el context pq la volia borrar a veure què passava
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
