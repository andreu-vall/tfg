import argparse
import os
import json
import sys
from torch.utils.data import DataLoader, Subset
import torch
import tqdm

from utils.peter import now_time, unique_sentence_percent, bleu_score, rouge_score
from peter_model import PETER
from data import MyDataset, MySplitDataset, setup_logger


# Auxiliar, ho crida per cada batch i junta els resultats
def generate(data : MyDataset, dataloader, model : PETER, device, strategy):

    results_json, results_metrics = [], []

    for batch in tqdm.tqdm(dataloader):

        batch = [elem.to(device) for elem in batch] # moure a cuda

        user, item, rating, text = batch # en el generate no es transposa el text

        # sembla que tarda uns 2 min a generar text amb context_window = 15 amb greedy pel test

        # it's predicted using: user & item only (not using the real rating, the real text or the real context)
        assert strategy == 'greedy', 'Only greedy strategy is implemented'
        predicted = model.generate(data.context_window, num_beams=1, do_sample=False, user=user, item=item, device=device)
        predicted_rating, predicted_context, predicted_text = predicted
  
        decoded_user = [data.user_decode(u) for u in user]
        decoded_item = [data.item_decode(i) for i in item]

        # havent entrenat una època sembla que ja ha après a generar <bos> i <eos>
        decoded_text = [data.text_decode(list(t)) for t in text] # needs a list to call the .index
        untokenized_text = [data.untokenize(t) for t in decoded_text]

        decoded_predicted_context = [data.text_decode(list(c), raw=True) for c in predicted_context]
        untokenized_predicted_context = [data.untokenize(c) for c in decoded_predicted_context]

        # sembla q tmb cal raw pq sinó peta? he entrenat ara 3 èpoques
        # Els <bos> sí que els posa, però el <eos> sembla que de moment no el posa sovint. De fet el <bos> potser el posa el model en sí
        # Potser predir l'últim token és estúpid, pq és either <eos> o <pad>, igual que el primer que és sempre <bos>
        # Entrenat 5 èpoques en el summary sí que ja ha après a posar el <bos> i <eos>
        decoded_predicted_text = [data.text_decode(list(t)) for t in predicted_text] # , raw=True
        untokenized_predicted_text = [data.untokenize(t) for t in decoded_predicted_text]

        for i in range(len(decoded_user)):
            results_json.append({
                'user': decoded_user[i],
                'item': decoded_item[i],
                'predicted_rating': predicted_rating[i].item(), # cal l'item pq si no és tensor i no és serialitzable
                'real_rating': rating[i].item(),
                'predicted_context': untokenized_predicted_context[i], # en realitat n'hi ha més, simplement mostro els més alts
                # real_context no té sentit pq simplement son les paraules més freqüents del text
                'predicted_text': untokenized_predicted_text[i],
                'real_text': untokenized_text[i]
            })
            results_metrics.append({
                'tokens_predicted_text': decoded_predicted_text[i],
                'tokens_real_text': decoded_text[i],
                'predicted_text': untokenized_predicted_text[i],
                'real_text': untokenized_text[i]
            })

    return results_json, results_metrics



def compute_text_quality(results_metrics):

    tokens_text_predicted = [result['tokens_predicted_text'] for result in results_metrics]
    tokens_text_real = [result['tokens_real_text'] for result in results_metrics]
    
    # Pel BLEU necessito els tokens en la forma de llista de tokens com a string
    BLEU1 = bleu_score(tokens_text_real, tokens_text_predicted, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_text_real, tokens_text_predicted, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    
    USR, USN = unique_sentence_percent(tokens_text_predicted)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))

    predicted_text = [result['predicted_text'] for result in results_metrics]
    real_text = [result['real_text'] for result in results_metrics]

    # En canvi pel ROUGE necessito els texts passats a string real tot junt ja. Possiblement té més valor doncs
    ROUGE = rouge_score(real_text, predicted_text)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))




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


if __name__ == "__main__":
    args = parse_arguments()
    
    path = os.path.join('out', args.id)

    mylogs = os.path.join(path, 'logs')
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log')
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')

    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.context_window)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, True)

    test_data = Subset(mydata, mysplitdata.test)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    results_json, results_metrics = generate(mydata, test_dataloader, mymodel, mydevice, args.strategy)

    with open(f"out/{args.id}/results/{args.result_id}.json", 'w') as f:
        json.dump(results_json, f, indent=4)

    
    compute_text_quality(results_metrics)


    # Demà seguiré per aquí + el test, i aviat he de començar a escriure la memòria ja...

    # it's predicted using: user & item only (not using the real rating, the real text or the real context)

    # La predicció de context simplement intenta predir les paraules més probables del text sense importar l'ordre,
    # de manera que l'objectiu és que sigui una tasca on no importa l'ordre i amb una única step ja predigui
    # per on anirà possiblement el text
