import math
import datetime

from .rouge import rouge
from .bleu import compute_bleu

# https://github.com/lileipisces/PETER/blob/master/utils.py


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



# Això és la clau de l'entrenament. Depenen del que si posis loss aprendrà a fer una cosa o altra el model
def loss(predicted, real, context_reg, text_reg, rating_reg, text_criterion, rating_criterion, ntokens, tgt_len):

    user, item, rating, seq = real

    log_word_prob, log_context_dis, rating_p, _ = predicted  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

    
    c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,))) # This is a bit ugly
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
    r_loss = rating_criterion(rating_p, rating)

    loss = c_loss * context_reg + t_loss * text_reg + r_loss * rating_reg # ordre més normal ara

    return c_loss, t_loss, r_loss, loss