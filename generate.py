
# # Aquí és on descodifica el context
# # Es donen només els ids dels topk tokens més probables per cada batch
# # S'haurien de tornar com a tokens instead ja ?
# def predict(log_context_dis, topk):
#     word_prob = log_context_dis.exp()  # (batch_size, ntoken)
#     if topk == 1:
#         context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
#     else:
#         context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
#     # print('predicted context')
#     # print(context.shape)
#     # print(context)
#     # assert(False)
#     return context  # (batch_size, topk)


# Ara estic tornant a fer servir variables globals, que és molt lleig i porta to unexpected behaviour
# Encara no he fet pràcticament res en aquesta funció
# Ara mateix ja hauria d'aprendre com funciona aquesta funció, pq sinó no tinc ni idea de com es generen les paraules...
# És com si ho fes del tot manual. Si ho fes tot d'una tirada potser li sortirien 16 paraules que no estan connectades,
# perquè les genera totes en paral·lel



# aviat la podré borrar sencera aquesta funció lletja aquí

# Aquesta funció crec que no toca tenir-la aquí, sinó en el model, són les estratègies de decoding
# def generate(data : MyDataset, dataloader, model, device):

#     tokenize = data.tokenize
#     untokenize = data.untokenize
#     max_rating = data.max_rating
#     min_rating = data.min_rating
#     ratings = data.ratings
#     sequences = data.texts
#     text_fixed_tokens = data.text_fixed_tokens

#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     idss_predict = [] # les explicacions en id's, generades amb greedy
#     context_predict = [] # les paraules més probables de les explicacions
#     rating_predict = [] # les prediccions de rating

#     # should call the model.generate, not do it manually here, it's ugly and usual models
#     # already have the generate method inside them, doing the strategi only outside is kinda ugly

#     with torch.no_grad():

#         for real in tqdm.tqdm(dataloader):

#             real = move_to_device(real, device, transpose_seq=False) # aquí específicament NO ho transposaven
#             user, item, rating, seq = real

#             bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size) # Maybe no funca?

#             text = bos  # (src_len - 1, batch_size)

#             start_idx = text.size(0)
#             for idx in range(text_fixed_tokens):
                
#                 if idx == 0:
#                     # En la línia de sota peta per algun motiu, però el model crec que no l'he canviat?
#                     # És simplement la primera vegada que es crida el forward amb el seq_predictions=False,
#                     # tots els anteriors no ho havien fet mai

#                     log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    
#                     # el rating s'usa tal qual
#                     rating_predict.extend(rating_p.tolist())

#                     # predicts the topk most probable tokens for the review. Sense descodificar-lo, només amb el id encara
#                     # probably si si he de fer feed del context al model és millor que estigui en ids,
#                     # però si he d'inferir resultats amb els tokens és més senzill d'interpretar les coses

#                     context = predict(log_context_dis, topk=text_fixed_tokens)  # (batch_size, words)
#                     context_predict.extend(context.tolist()) # crec que aquí afora té més sentit com a tokens no ID's
#                     # aquí simplement es construeix

#                 else:
#                     log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)

#                 # L'única cosa que es modifica és el text, cada cop se li va afegint una nova paraula
#                 # És el greedy decoding, que en cada step posa la paraula més probable.
#                 # Per això en el PETER probably genera kind of coses genèriques, perquè s'utilitza
#                 # una estratègia de decoding de text molt senzilla. Crec que seria fàcil i interessant
#                 # estudiar altres, totes, les estratègies bàsiques de decoding

#                 word_prob = log_word_prob.exp()  # (batch_size, ntoken)
#                 word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
#                 text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
#             ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
#             idss_predict.extend(ids)
    
#     return idss_predict, context_predict, rating_predict

#     print(idss_predict)
#     print(context_predict)
#     print(rating_predict)

# Això eren les mètriques del PETER. Tot i que no crec gaire en les mètriques i el que m'importa és més fer algo útil,
# una mica de les mètriques sí que les hauria d'usar i calcular
# Efectivament aquestes mètriques són del generate i no del test, en el test simplement s'usa la mateixa mètrica que el train

    # predicted_rating = [(r, p) for (r, p) in zip(ratings, rating_predict)] # he canviat data per dataloader
    # RMSE = root_mean_square_error(predicted_rating, max_rating, min_rating)
    # peter_logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    # MAE = mean_absolute_error(predicted_rating, max_rating, min_rating)
    # peter_logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))
    
    # text_test = [untokenize(ids) for ids in sequences] #
    # text_predict = [untokenize(ids, wrong=True) for ids in idss_predict] # he posat que si
    # # és untrained no salti l'error si es genera una sèrie de tokens sense el <bos> ni <eos>


    # # Ho necessita en formats de tokens per calcular aquesta mena de coses? Quite weird ngl
    # BLEU1 = bleu_score(sequences, idss_predict, n_gram=1, smooth=False)
    # peter_logger.info(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    # BLEU4 = bleu_score(sequences, idss_predict, n_gram=4, smooth=False)
    # peter_logger.info(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    # USR, USN = unique_sentence_percent(idss_predict)
    # peter_logger.info(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    
    # tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    # ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    # for (k, v) in ROUGE.items():
    #     peter_logger.info(now_time() + '{} {:7.4f}'.format(k, v))
    # text_out = ''
    # for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
    #     text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    # return text_out



import argparse
import os
import json
import sys
from torch.utils.data import DataLoader, Subset
import torch
import tqdm

from utils.peter import now_time
from data import MyDataset, MySplitDataset, setup_logger
# , now_time és un import que es fa a través de utils.peter i funciona??


# Join cmd arguments and the arguments saved previously from the training
def parse_arguments():
    cmd_parser = argparse.ArgumentParser(description='Generate')
    cmd_parser.add_argument('id', type=str, help='model id')
    cmd_parser.add_argument('strategy', type=str, choices=['greedy'], help='decoding strategy')
    cmd_parser.add_argument('result_id', type=str, help='result id')
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA')
    cmd_args = cmd_parser.parse_args()

    path = f"out/{cmd_args.id}"
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')
    
    base_path = f"out/{cmd_args.id}/results"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    result_path = f"{base_path}/{cmd_args.result_id}.json"
    if os.path.isfile(result_path):
        raise ValueError('This result id already exists!')

    with open(f'{path}/train.json', 'r') as f:
        train_args = json.load(f)
    
    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


from data import move_to_device
from peter_model import PETER


def generate(data : MyDataset, dataloader, model : PETER, device, strategy):
    results = []

    for batch in tqdm.tqdm(dataloader):

        user, item, rating, text = move_to_device(batch, device)

        # sembla que tarda uns 2 min a generar text amb context_window = 15 amb greedy pel test

        # it's predicted using: user & item only (not using the real rating, the real text or the real context)
        # Ara mateix és només per una batch
        assert strategy == 'greedy', 'Only greedy strategy is implemented'
        predicted = model.generate(data.context_window, num_beams=1, do_sample=False, user=user, item=item, device=device)
        predicted_rating, predicted_context, predicted_text = predicted

        # Coses reals
        decoded_user = [data.user_decode(u) for u in user]
        decoded_item = [data.item_decode(i) for i in item]
        decoded_text = [data.text_unvectorize(list(t)) for t in text]
        # estava mirant com es creava el context per comparar-ho amb el real

        decoded_predicted_context = [data.text_unvectorize(c, raw=True) for c in predicted_context]

        # sembla q tmb cal raw pq sinó peta? he entrenat ara 3 èpoques
        # Els <bos> sí que els posa, però el <eos> sembla que de moment no el posa sovint. De fet el <bos> potser el posa el model en sí
        # Potser predir l'últim token és estúpid, pq és either <eos> o <pad>, igual que el primer que és sempre <bos>
        decoded_predicted_text = [data.text_unvectorize(list(t), raw=True) for t in predicted_text] # needs a list to call the .index

        batch_results = [{'user': decoded_user[i],
             'item': decoded_item[i],
             'predicted_rating': predicted_rating[i].item(),
             'real_rating': rating[i].item(), # cal l'item pq si no és tensor i no és serialitzable
             'predicted_context': decoded_predicted_context[i],
             'predicted_text': decoded_predicted_text[i],
             'real_text': decoded_text[i]} for i in range(len(decoded_user))]
        
        results.extend(batch_results)
    
    return results



if __name__ == "__main__":
    args = parse_arguments()
    
    path = os.path.join('out', args.id)

    mylogs = os.path.join(path, 'logs')
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log')
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')

    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    if torch.cuda.is_available():
        if args.cpu:
            peter_logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.context_window)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, True)

    test_data = Subset(mydata, mysplitdata.test)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    results = generate(mydata, test_dataloader, mymodel, mydevice, args.strategy)

    result_path = f"out/{args.id}/results/{args.result_id}.json"

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)


    # it's predicted using: user & item only (not using the real rating, the real text or the real context)

    # La predicció de context simplement intenta predir les paraules més probables del text sense importar l'ordre,
    # de manera que l'objectiu és que sigui una tasca on no importa l'ordre i amb una única step ja predigui
    # per on anirà possiblement el text

    # encara he de fer les mètriques, i si pot ser que siguin més útils que les que tenien en PETER
