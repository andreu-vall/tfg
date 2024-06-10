import argparse
import os
import json
from torch.utils.data import DataLoader, Subset
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F

from peter_model import PETER
from data import MyDataset, MySplitDataset, record_execution, decode_batch_results, get_RMSE_MAE
from utils.peter import bleu_score, rouge_score



def get_topk_tokens(log_token_dis, topk):
    token_prob = log_token_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(token_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(token_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)



# Es podria generar text amb max_length <= self.context_window, i a partir d'aquí si el vulguessis fer més llarg
# ja no li podries passar tot el text prèviament generat pel propi model, i.e. no és capaç de agafar un input
# tant llarg. Encara hauria de simplificar els arguments del PETER per realment només els que crec que són útils
# i provar a jugar amb ells com varien les coses
def generate_batch(model:nn.Module, bos_idx, max_length, top_k_sample, num_beams, user, item, device):

    assert num_beams==1, "beam search not yet implemented"
    # Estaria bé també tenir el beam search però realment tampoc aporta gaire més, simplement tardarà més
    # en generar perquè consdierarà més possibilitats a l'hora. L'única cosa que m'aportaria és que jo sabria
    # quina és la probabilitat combinada d'una següència sencera

    batch_size = user.size(0)
    # Comencem amb tot <bos> i anirem afegint paraules iterativament
    text = torch.full((1, batch_size), bos_idx).to(device)
    # PETER: text = torch.tensor(self.bos_idx).repeat(batch_size, 1).t()  # (src_len - 1, batch_size)
    
    user = user.to(device)
    item = item.to(device)

    for step in range(max_length):

        #falta implementar el generate amb el nou forward. Cal decidir exactament com ho vull fer i fer-ho

        # l'hauré de cridar ara amb el mode='sequential'
        
        # probably no funcioni encara

        # posat el rating com a None? Potser l'hauria de calcular la primera vegada i després passar-li
        # el propi rating que he calculat en la primera iteració?
        #
        # Ah clar, em falta passar-li

        if step == 0:
            # yikes amb un mode ara peta aquí
            log_word_prob, log_context_dis, rating, _ = model(user, item, None, text, 'sequential')
            context = get_topk_tokens(log_context_dis, topk=max_length)
        else:
            # li he posat el rating ara
            log_word_prob, _, _, _ = model(user, item, rating, text, 'sequential')



        # step 0: es calcula el rating, el context i el 1r token (a partir de <bos>)
        # if step == 0:
        #     log_word_prob, log_context_dis, rating, _ = model(user, item, text, False)
        #     context = get_topk_tokens(log_context_dis, topk=max_length)
        
        # # step > 0: se li introdueix el seu text que havia generat fins ara i genera 1 token més
        # else:
        #     log_word_prob, _, _, _ = model(user, item, text, False, False, False)

        # squeeze: elimina dimensions de mida 1
        # unsqueeze: afegeix dimensions de mida 1¡
        if top_k_sample == 1: # greedy
            _, next_token = torch.max(log_word_prob, dim=-1)
            #print('here next_token shape was:', next_token.shape) # [128]
        
        else: # sampling, ojo que maybe el copilot la pot haver liat aquí. això fa que no sigui determinístic la generació
            topk_prob, topk_indices = torch.topk(F.softmax(log_word_prob, dim=-1), top_k_sample)
            next_token = topk_indices.view(-1)[torch.multinomial(topk_prob, 1).squeeze()]
            #print('here next_token shape is:', next_token.shape) # [128]

        text = torch.cat([text, next_token.unsqueeze(0)], 0)

    return rating, context, text.T


# Auxiliar, ho crida per cada batch i junta els resultats
def generate(data:MyDataset, dataloader, model:PETER, device, top_k_sample, num_beams, target_length):

    if target_length is None: # potser hauria d'adclarir en execució script que si no s'pesecifica s'usa la shape del model
        target_length = data.max_tokens
    else:
        assert target_length <= data.max_tokens, "target_length should be less or equal to the context_window"
        # technically it could be less, but then the model would "forget" what it has said itself, because it doesn't
        # have the capacity to process such a long input in it's architecture

    results, metrics = [], []

    for batch in tqdm.tqdm(dataloader, mininterval=1):

        batch = [elem.to(device) for elem in batch] # moure a cuda

        user, item, rating, text = batch # en el generate NO es transposa el text

        # falta posar el paràmetre de la longitud a generar, que ha de ser probablement més petita que la context_window.
        # De fet podria ser més llarga però llavors s'oblidaria del que ha dit ell mateix i seria possiblement kinda stupid?
        # No del tot necessàriament? Potser és interessant provar-ho si és fàcil de fer-ho

        # it's predicted using: user & item only (not using the rating nor the text)
        predicted = generate_batch(model, data.token_dict.bos, target_length, top_k_sample, num_beams, user, item, device)
        
        predicted_rating, predicted_context, predicted_text = predicted

        batch_results, batch_metrics = decode_batch_results(
            user, item, rating, text, predicted_rating, predicted_context, predicted_text, data)

        results.extend(batch_results)
        metrics.extend(batch_metrics)

    parameters = {
        "top_k_sample": top_k_sample,
        "num_beams": num_beams,
        "target_length": target_length # també és molt important
    }
    return parameters, metrics, results



def compute_text_quality(results_metrics):

    tokens_text_predicted = [result['tokens_predicted_text'] for result in results_metrics]
    tokens_text_real = [result['tokens_real_text'] for result in results_metrics]
    
    # Pel BLEU necessito els tokens en la forma de llista de tokens com a string
    BLEU1 = bleu_score(tokens_text_real, tokens_text_predicted, n_gram=1, smooth=False) # són en %
    BLEU4 = bleu_score(tokens_text_real, tokens_text_predicted, n_gram=4, smooth=False)
    
    # Es feia amb O(N^2) quan ja tinc tmb les frases com a string...
    # USR, USN = unique_sentence_percent(tokens_text_predicted)
    predicted_text = [result['predicted_text'] for result in results_metrics]
    unique_sentences = set(predicted_text)
    USN = len(unique_sentences)
    USR = USN / len(predicted_text)

    real_text = [result['real_text'] for result in results_metrics]

    # En canvi pel ROUGE necessito els texts passats a string real tot junt ja. Possiblement té més valor doncs?
    ROUGE = rouge_score(real_text, predicted_text)  # a dictionary

    return {
        'BLEU-1': BLEU1,
        'BLEU-4': BLEU4,
        'USR': USR,
        'USN': USN,
        'ROUGE': ROUGE
    }



# Join cmd arguments and the arguments saved previously from the training
def parse_arguments():
    cmd_parser = argparse.ArgumentParser(description='Generate')

    cmd_parser.add_argument('train_id', type=str, help='model id')
    cmd_parser.add_argument('result_id', type=str, help='result id')

    cmd_parser.add_argument('--top_k_sample', type=int, help='top k sampling', default=1) # 1=greedy, >1=sampling
    cmd_parser.add_argument('--num_beams', type=int, help='number of beams', default=1)
    cmd_parser.add_argument('--seed', type=int, help='random seed', default=42)

    cmd_parser.add_argument('--target_length', type=int, help='target length') # should be optative and default the context_window?
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA')
    cmd_args = cmd_parser.parse_args()

    path = f"out/{cmd_args.train_id}"
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')
    
    base_path = f"out/{cmd_args.train_id}/results"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    result_path = f"{base_path}/{cmd_args.result_id}.json"
    if os.path.isfile(result_path):
        raise ValueError('This result id already exists!')

    with open(f'{path}/train_args.json', 'r') as f:
        train_args = json.load(f)
    
    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    path = os.path.join('out', args.train_id)
    record_execution(path)

    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.max_tokens, args.vocab_size)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, True)

    test_data = Subset(mydata, mysplitdata.test)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # crec que sobretot en el generate és important de posar una seed!!
    # Sembla que funciona, amb 2 iteracions diferents ha generat exactament el mateix text, nicee
    torch.manual_seed(args.seed)

    parameters, metrics, results = generate(mydata, test_dataloader, mymodel, mydevice,
                                            args.top_k_sample, args.num_beams, args.target_length)

    RMSE, MAE = get_RMSE_MAE(results, mydata.max_rating, mydata.min_rating)

    # crec que tarda lo seu en calcular la qualitat del text, potser estaria bé posar una barra de progrés
    text_quality = compute_text_quality(metrics)

    metrics_json = {
        #"losses": losses_dic, # les losses crec que seria complicat calulcar-les i no tenen molt sentit en el generate
        "RMSE": RMSE,
        "MAE": MAE,
        "text_quality": text_quality
    }
    parameters['seed'] = args.seed # ara mateix només la tinc al main però és molt important posar-la en el json
    generate_json = {
        "parameters": parameters,
        "metrics": metrics_json,
        "results": results
    }
    with open(f"out/{args.train_id}/results/{args.result_id}.json", 'w') as f:
        json.dump(generate_json, f, indent=4)
    
    # Demà dilluns ja no afegiré cap funcionalitat més. He d'escriure ja la memòria, que si no no tindré
    # algo amb cara i ulls pel dijous, i aniré en una espiral de posar-me nerviós fins el dia 10 de juny

    # Demà seguiré per aquí + el test, i aviat he de començar a escriure la memòria ja...
    # LOL la memòria. M'he de passar un dia límit a mi mateix de posar'm-hi ja en serio,
    # o si no no entregaré el TFG aquest semestre...

    # it's predicted using: user & item only (not using the real rating, the real text or the real context)

    # La predicció de context simplement intenta predir les paraules més probables del text sense importar l'ordre,
    # de manera que l'objectiu és que sigui una tasca on no importa l'ordre i amb una única step ja predigui
    # per on anirà possiblement el text
