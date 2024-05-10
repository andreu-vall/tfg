import math
import heapq
import datetime

from .rouge import rouge
from .bleu import compute_bleu


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


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


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


class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    # naive space splitting
    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20_000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)



def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


# A partir del <eos> ja no es torna res més. Si es fes tot en paral·lel sense la list comprehension
# potser seria més fàcil, però per 15 words potser no val la pena i perdo més temps en passar a pandas
def ids2tokens(ids, word2idx, idx2word, untrained=False):
    if not untrained:
        bos = word2idx['<bos>'] # El model entrenat 0 èpoques en general no em genera ni <bos> ni <eos>!!
        eos = word2idx['<eos>']
        assert bos == ids[0] # yikes el model sense entrenar res no genera el <bos>!!
        eos_index = ids.index(eos) # Si no trobés el <eos> petaria, crec que ja és el que s'espera
        return [idx2word[i] for i in ids[:eos_index+1]]
    
    return [idx2word[i] for i in ids]
