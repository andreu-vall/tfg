import argparse
import os
import torch
import torch.nn as nn
import logging
import json
import sys

from utils.peter import rouge_score, bleu_score, DataLoader, Batchify, root_mean_square_error, mean_absolute_error, ids2tokens, \
    unique_sentence_percent, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity, now_time

from utils.andreu import move_content_to_device, peter_content, peter_loss_good, setup_logger


def test(dataloader, model, loss_fn, device, use_feature):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = torch.zeros(4)
    
    with torch.no_grad():
        for content in dataloader:

            content = move_content_to_device(content, device)

            user, item, rating, seq, feature = content
            batch_size = user.size(0)

            if use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

            pred = model(user, item, text)

            losses = loss_fn(pred, content) # [c_loss, r_loss, t_loss, loss] # al revés?

            total_losses += torch.tensor(losses) * batch_size

    return (total_losses / dataloader.total_elements).tolist() # Crec q amb el tolist s'hauria de solucionar les referències


# Aquí és on descodifica el context
def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


# Encara no he fet pràcticament res en aquesta funció
def generate(dataloader, model, device, use_feature, words):

    peter_logger = logging.getLogger("peter_logger")
    # andreu_logger = logging.getLogger("andreu_logger") # Falta acabar de definir les coses que vull fer log jo

    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        for content in dataloader:

            # this is a whole batch
            user, item, rating, seq, feature = move_content_to_device(content, device)

            seq = seq.t() # ????? LOL IT WORKS PQ EN UN LLOC TRANSPOSAT I L'ALTRE NO??????

            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size) # Maybe no funca?

            # Obvi que amb la feature donarà millors resultats, si l'estàs utilizant com
            # a cosa a partir de la qual començar a generar el text del output de l'explicació
            if use_feature:
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
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
    predicted_rating = [(r, p) for (r, p) in zip(dataloader.rating.tolist(), rating_predict)] # he canviat data per dataloader
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    peter_logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    peter_logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in dataloader.seq.tolist()] # he canviat data per dataloader
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    peter_logger.info(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    peter_logger.info(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    peter_logger.info(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    peter_logger.info(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    peter_logger.info(now_time() + 'FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in dataloader.feature.squeeze(1).tolist()]  # ids to words, he canviat data per dataloader
    FMR = feature_matching_ratio(feature_batch, feature_test)
    peter_logger.info(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        peter_logger.info(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    return text_out


# primer cop que utilitzo voluntàriament pq el necessito el name main xD
if __name__ == "__main__":

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


    corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
    tgt_len = args.words + 1  # added <bos> or <eos>
    ntokens = len(corpus.word_dict)
    word2idx = corpus.word_dict.word2idx
    pad_idx = word2idx['<pad>']
    test_dataloader = Batchify(corpus.test, word2idx, args.words, args.batch_size)

    idx2word = corpus.word_dict.idx2word
    feature_set = corpus.feature_set


    mytext_criterion = nn.NLLLoss(ignore_index=pad_idx)  # És això duplicació de codi?
    myrating_criterion = nn.MSELoss()


    peter_loss = lambda pred, content: peter_loss_good(pred, content, args.context_reg, args.text_reg, args.rating_reg,
                                                       mytext_criterion, myrating_criterion, ntokens, tgt_len)


    # Run on test data.
    test_losses = test(test_dataloader, mymodel, peter_loss, mydevice, args.use_feature)
    c_loss, t_loss, r_loss, loss = test_losses
    peter_logger.info('=' * 89)
    peter_logger.info(f"{now_time()}{peter_content(c_loss, t_loss, r_loss)} on test")

    prediction_path = os.path.join(path, args.outf)

    peter_logger.info(now_time() + 'Generating text')
    text_o = generate(test_dataloader, mymodel, mydevice, args.use_feature, args.words)
    with open(prediction_path, 'w', encoding='utf-8') as f:
        f.write(text_o)
    peter_logger.info(now_time() + 'Generated text saved to ({})'.format(prediction_path))
