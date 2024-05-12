import argparse
import os
import torch
import torch.nn as nn
import logging
import json
import sys

from torch.utils.data import DataLoader, Subset

from utils.peter import rouge_score, bleu_score, root_mean_square_error, mean_absolute_error, \
    unique_sentence_percent, now_time, content, loss

from utils.data import MyDataset, MySplitDataset, setup_logger, move_to_device


def test(dataloader: DataLoader, model, loss_fn, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = torch.zeros(4)
    
    with torch.no_grad():
        
        for real in dataloader:

            real = move_to_device(real, device, transpose_seq=True) # en el test tmb ho transposaven
            user, item, rating, seq = real
            batch_size = user.size(0)

            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

            predicted = model(user, item, text)

            losses = loss_fn(predicted, real) # [c_loss, r_loss, t_loss, loss]

            total_losses += torch.tensor(losses) * batch_size

    return (total_losses / len(dataloader.dataset)).tolist()


# Aquí és on descodifica el context
def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


# Ara estic tornant a fer servir variables globals, que és molt lleig i porta to unexpected behaviour
# Encara no he fet pràcticament res en aquesta funció
# Ara mateix ja hauria d'aprendre com funciona aquesta funció, pq sinó no tinc ni idea de com es generen les paraules...
# És com si ho fes del tot manual. Si ho fes tot d'una tirada potser li sortirien 16 paraules que no estan connectades,
# perquè les genera totes en paral·lel
def generate(data, dataloader, model, device, words, word2idx, idx2word, max_rating, min_rating, ratings, sequences):

    peter_logger = logging.getLogger("peter_logger")
    # andreu_logger = logging.getLogger("andreu_logger") # Falta acabar de definir les coses que vull fer log jo

    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():

        for real in dataloader:

            real = move_to_device(real, device, transpose_seq=False) # aquí específicament NO ho transposaven
            user, item, rating, seq = real

            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size) # Maybe no funca?

            text = bos  # (src_len - 1, batch_size)

            start_idx = text.size(0)
            for idx in range(words):
                # produce a word at each step

                # Aquí és el pas clau on genera múltiples coses executant el model iterativament a partir
                # de les coses generades. Encara no ho acabo d'entendre però em sembla una manera més bona
                # de generar recomanacions que no tant el P5. M'agraden més les bones arquitecutres que no
                # simplement els bons prompts, perquè fer bons prompts és més un art i fer bones arquitectures
                # és un procés científic

                # COm es genera aquí un context tant llarg, si només es crida una vegada pel context?
                # Ja ho vaig entendre, simplement el context són les 15 paraules més probables (en paral·lel)

                if idx == 0:
                    # En la línia de sota peta per algun motiu, però el model crec que no l'he canviat?
                    # És simplement la primera vegada que es crida el forward amb el seq_predictions=False,
                    # tots els anteriors no ho havien fet mai

                    log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.extend(rating_p.tolist())
                    context = predict(log_context_dis, topk=words)  # (batch_size, words)
                    context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)

                # L'única cosa que es modifica és el text, cada cop se li va afegint una nova paraula
                # És el greedy decoding, que en cada step posa la paraula més probable.
                # Per això en el PETER probably genera kind of coses genèriques, perquè s'utilitza
                # una estratègia de decoding de text molt senzilla. Crec que seria fàcil i interessant
                # estudiar altres, totes, les estratègies bàsiques de decoding

                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

    # rating
    # canviat corpus per data
    # tinc problemes amb data i dataloader en aquí, no tinc clar exactament què haurien de ser...
    # Per obtenir els reals ho fa d'una manera molt estranya, accedint directament al .rating o .seq
    predicted_rating = [(r, p) for (r, p) in zip(ratings, rating_predict)] # he canviat data per dataloader
    RMSE = root_mean_square_error(predicted_rating, max_rating, min_rating)
    peter_logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, max_rating, min_rating)
    peter_logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    text_test = [data.untokenize_text(ids) for ids in sequences] # he canviat data per dataloader
    text_predict = [data.untokenize_text(ids, wrong=True) for ids in idss_predict] # he posat que si
    # és untrained no salti l'error si es genera una sèrie de tokens sense el <bos> ni <eos>


    # Ho necessita en formats de tokens per calcular aquesta mena de coses? Quite weird ngl
    BLEU1 = bleu_score(sequences, idss_predict, n_gram=1, smooth=False)
    peter_logger.info(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(sequences, idss_predict, n_gram=4, smooth=False)
    peter_logger.info(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(idss_predict)
    peter_logger.info(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        peter_logger.info(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    return text_out


# Encara he de netejar una mica això del main que ho tinc realment molt lleig
# primer cop que utilitzo voluntàriament pq el necessito el name main xD
if __name__ == "__main__":

    # This could be cleaned up a bit
    # El meu modus operandis és agafar un codi que funciona i executar-lo jo i anar-lo editant per entendre'l

    # Parse command line arguments. Primer hem de fet aquest pq és el que dona la id
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('id', type=str, help='model id')
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA')
    cmd_parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')

    cmd_args = cmd_parser.parse_args()

    path = os.path.join('out', cmd_args.id)
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')
    

    mylogs = os.path.join(path, 'logs')
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log', True) # Per tant de moment mostro per pantalla els logs de PETER
    #andreu_logger = setup_logger('andreu_logger', f'{mylogs}/andreu.log', True) # De moment pel test no he fet cap log yet
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')

    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    # Load arguments from file
    with open(f'out/{cmd_args.id}/train.json', 'r') as f:
        train_args = json.load(f)

    # Merge the two sets of arguments
    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot

    # Convert the merged dictionary back to a Namespace object
    args = argparse.Namespace(**merged_args)

    model_path = os.path.join(path, 'model.pt')

    if torch.cuda.is_available():
        if args.cpu:
            peter_logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')



    # Load the best saved model.
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice) # Simplement he carregat el meu model enlloc del seu


    mydata = MyDataset.load_or_create(args.data_path, args.words, args.vocab_size)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.index_dir, True)

    test_data = Subset(mydata, mysplitdata.test)
    # té sentit fer el shuffle en el test???
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    pad_idx = mydata.pad


    tgt_len = args.words + 1  # added <bos> or <eos>
    ntokens = len(mydata.word_dict)
    myword2idx = mydata.word_dict.entity_to_idx

    myidx2word = mydata.word_dict.idx_to_entity

    mytext_criterion = nn.NLLLoss(ignore_index=pad_idx)  # És això duplicació de codi?
    myrating_criterion = nn.MSELoss()

    peter_loss = lambda predicted, real: loss(predicted, real, args.context_reg, args.text_reg, args.rating_reg,
                                              mytext_criterion, myrating_criterion, ntokens, tgt_len)

    # Run on test data.
    test_losses = test(test_dataloader, mymodel, peter_loss, mydevice)
    c_loss, t_loss, r_loss, loss = test_losses
    peter_logger.info('=' * 89)
    peter_logger.info(f"{now_time()}{content(c_loss, t_loss, r_loss)} on test")

    prediction_path = os.path.join(path, args.outf)

    peter_logger.info(now_time() + 'Generating text')

    ratings = [content[2] for content in test_data]
    sequences = [content[3] for content in test_data]

    text_o = generate(mydata, test_dataloader, mymodel, mydevice, args.words, myword2idx, myidx2word,
                      mydata.max_rating, mydata.min_rating, ratings, sequences)
    
    # real,
    # context (top most probable words for this task in this order),
    # predicted (greedy generated left to right)

    # A part del format del PETER, potser puc fer el meu propi. Amb un json seria més fàcil d'interpretar per exemple

    with open(prediction_path, 'w', encoding='utf-8') as f:
        f.write(text_o)
    peter_logger.info(now_time() + 'Generated text saved to ({})'.format(prediction_path))
