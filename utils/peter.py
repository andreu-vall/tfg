import math
import datetime

from .rouge import rouge
from .bleu import compute_bleu

# https://github.com/lileipisces/PETER/blob/master/utils.py


# I think I will completely delete this whole file as it quite useless
# Or at least delete a lot of things from here


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def content(context_loss, text_loss, rating_loss):
    exp_context_loss = math.exp(context_loss)
    exp_text_loss = math.exp(text_loss)
    return f"context ppl {exp_context_loss:4.4f} | text ppl {exp_text_loss:4.4f} | rating loss {rating_loss:4.4f}"




# pendent: borrar-la
def loss(predicted, real, context_reg, text_reg, rating_reg, text_criterion, rating_criterion, ntokens, tgt_len):

    # de predicted només s'utilitza: 
    # log_word_prob les probabilitats en cada query sobre tots els tokens del vocabulari
    # log_context_dis les probabilitas exactament igual que en paraula però sense tenir en compte ordre paraules?
    # rating_p el rating predit per comparar-lo amb el real
    # NO les attns

    # de real només s'utilitza:
    # rating per comparar-lo amb el rating predit
    # text (en la versió de shifted right)
    

    user, item, rating, text = real

    #log_word_prob, log_context_dis, rating, attns = predicted

    log_word_prob, log_context_dis, rating_p, _ = predicted  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

    # el context es compara amb text[1:-1] directament, i què vindria a ser?

    # El text és tot el text
    # i li va treient coses del principi i del final. El primer potser és el start token, el del final el
    # que estàs intentant predir en el text i per tant és millor no utilitzar-lo???
    # I'm still not fully sure about this

    # print('text shape is', text.shape)
    # print('text is', text)
    # assert(False)

    
    c_loss = text_criterion(context_dis.view(-1, ntokens), text[1:-1].reshape((-1,))) # This is a bit ugly
    #1234 [:-1] crec aquí hi havia. Crec que el leakage be de treure aquí el [1: de fet]
    # Què significa aquí un [1:]? Que no miris el primer token, que crec que well obviously és el <bos>
    # Potser per això a ells no els genera el <bos>? Però no justifica que el meu hi hagi leakage això només?
    
    # el [1:] és per fer el shift right del output
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), text[1:].reshape((-1,))) # ara modificant la mask cal canviar,
    # abans era text[1:]
    r_loss = rating_criterion(rating_p, rating)

    loss = c_loss * context_reg + t_loss * text_reg + r_loss * rating_reg # ordre més normal ara

    return c_loss, t_loss, r_loss, loss


# Això és el que feien en el PETER. Per la validació s'ignora el loss de context (i maybe tmb el de rating)
# Sembla bastant estrany el que imprimeixen, pq no és pas la loss real en cap lloc
def peter_validation_msg(val_losses, rating_reg):
    c_loss, t_loss, r_loss, real_loss = val_losses
    printed_loss = t_loss
    if rating_reg != 0: # what even is rating_reg?
        printed_loss += r_loss
    # Crec que aquí explota i no pinta res l'exponencial???
    return f"{now_time()}{content(c_loss, t_loss, r_loss)} | valid loss {printed_loss:4.4f} on validation"

