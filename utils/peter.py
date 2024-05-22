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


# def content(context_loss, text_loss, rating_loss):
#     exp_context_loss = math.exp(context_loss)
#     exp_text_loss = math.exp(text_loss)
#     return f"context ppl {exp_context_loss:4.4f} | text ppl {exp_text_loss:4.4f} | rating loss {rating_loss:4.4f}"


# # Això és el que imprimien en PETER. No és realment la loss de la cosa que s'està optimitzant, not sure pq ho posen així
# # A més la cosa del raing_reg està malament clarament
# # Això és el que feien en el PETER. Per la validació s'ignora el loss de context (i maybe tmb el de rating)
# # Sembla bastant estrany el que imprimeixen, pq no és pas la loss real en cap lloc
# def peter_validation_msg(val_losses, rating_reg):
#     c_loss, t_loss, r_loss, real_loss = val_losses
#     printed_loss = t_loss
#     if rating_reg != 0: # what even is rating_reg?
#         printed_loss += r_loss
#     # Crec que aquí explota i no pinta res l'exponencial??? Actually sí pq he aplicat logaritmes a algun lloc
#     return f"{now_time()}{content(c_loss, t_loss, r_loss)} | valid loss {printed_loss:4.4f} on validation"


    # if args.rating_reg == 0:
    #     val_loss = val_t_loss
    # else:
    #     val_loss = val_t_loss + val_r_loss
    # print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
    #     math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))

