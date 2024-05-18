import argparse
import os
import json
from torch.utils.data import DataLoader, Subset
import torch
import tqdm
import torch.nn as nn

from peter_model import PETER
from data import MyDataset, MySplitDataset, record_execution, generate_batch_results, compute_text_quality, get_RMSE_MAE


# Crec que hauria de moure aquests 2 mètodes fora

def get_topk_tokens(log_token_dis, topk):
    token_prob = log_token_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(token_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(token_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


# tot i que sembla que funciona, encara falten implementar coses i acabar-la de mirar
# abans de mirar-me el generate he d'entendre també com funciona en el train i test
# tot i que potser mirarar el generate tmb ajuda

# Es podria generar text amb max_length <= self.context_window, i a partir d'aquí si el vulguessis fer més llarg
# ja no li podries passar tot el text prèviament generat pel propi model, i.e. no és capaç de agafar un input
# tant llarg. Encara hauria de simplificar els arguments del PETER per realment només els que crec que són útils
# i provar a jugar amb ells com varien les coses
def generate_batch(model:nn.Module, bos_idx, max_length, num_beams, do_sample, user, item, device):

    assert num_beams==1, "only greedy generation for now" # per donar una pista amb l'assert
    assert do_sample==False, "only greedy generation for now" # és una mica estrany el format ngl

    batch_size = user.size(0)

    # Comencem amb tot <bos> i anirem afegint paraules iterativament
    text = torch.full((1, batch_size), bos_idx).to(device)
    
    user = user.to(device)
    item = item.to(device)

    # # sembla que en el PETER, el primer text és torch.Size([1, 128]) amb tot start_idx
    # # el següents es va afegint una dimensió més amb les paraules greedy descodificades per cadascú

    # # PETER: size (src_len - 1, batch_size)
    # # should it already be of all the size and only edit parts of it?
    # text = torch.tensor(self.bos_idx).repeat(batch_size, 1).t()  # (src_len - 1, batch_size)
    #print('text shape', text.shape)

    #text = torch.tensor(self.bos_idx) # ara mateix no tinc el beggining of sequence aquí
    # li hauré de canviar la shape?

    # ojo que això és bastant del copilot encara

    # A la step 0 es calcula la predicció de context i de rating
    # Greedy: a tots els steps, inclòs el 0, es calcula el següent token més probable del text. S'havia començat amb el <bos>

    # tindria sentit anar ajustant la predicció dels ratings i context en cada step? Crec que no, perquè la cosa és que les
    # generes això en primer lloc i després vas generant el text a poc a poc amb la idea de en base de això hauria de ser

    # És necessari executar el model max_length cops, perquè l'únic que sap fer el model és predir exactament 1 token més
    # a partir de tot el que ja sap, i si no sap res doncs necessitarà max_length cops per generar el text de la longitud
    # que vulguis

    # aquesta funció encara no he tocat gaires coses, és la traducció més llegible del codi de PETER,
    # encara no he canviat la estratègia de res

    for step in range(max_length):
        if step == 0: # step 0: es calcula el rating, el context i el 1r token
            log_word_prob, log_context_dis, rating, _ = model(user, item, text, False) # el forward del model
            context = get_topk_tokens(log_context_dis, topk=max_length)
        else: # step > 0: se li introdueix el seu text que havia generat fins ara i es NOMÉS el següent token
            # tècnicament el model també torna a calcular una altra predicció de context i rating però s'ignora,
            # perquè el que importa aquí és generar un text llarg de manera autoregressiva
            log_word_prob, _, _, _ = model(user, item, text, False, False, False) # el forward del model
        # print('step', step)
        # print('here, log_word_prob shape is', log_word_prob.shape)
        _, next_token = torch.max(log_word_prob, dim=-1)
        # print('and the next token shape is', next_token.shape)
        # if step == 1:
        #     assert(False)
        text = torch.cat([text, next_token.unsqueeze(0)], 0) # aquí és on es concatena

    return rating, context, text.T # millor transposar aquí ja? Crec que sí


# Auxiliar, ho crida per cada batch i junta els resultats
def generate(data:MyDataset, dataloader, model:PETER, device, num_beams, do_sample):

    results, metrics = [], []

    for batch in tqdm.tqdm(dataloader, mininterval=1):

        batch = [elem.to(device) for elem in batch] # moure a cuda

        user, item, rating, text = batch # en el generate no es transposa el text
        batch_size = user.size(0)

        # falta posar el paràmetre de la longitud a generar, que ha de ser probablement més petita que la context_window.
        # De fet podria ser més llarga però llavors s'oblidaria del que ha dit ell mateix i seria possiblement kinda stupid?
        # No del tot necessàriament? Potser és interessant provar-ho si és fàcil de fer-ho

        # it's predicted using: user & item only (not using the real rating, the real text or the real context)
        predicted = generate_batch(model, data.token_dict.bos, data.context_window, num_beams, do_sample, user, item, device)
        
        predicted_rating, predicted_context, predicted_text = predicted

        # la diferència és que aquí les coses ja venen en forma correcta del generate_batch
            # (return rating, context, text.T # millor transposar aquí ja? Crec que sí)
        # input d'aquí: user, item, text; predicted_context, predicted_text (aquests 2 ja estan bé), predicted_rating
        # output: results

        


        batch_results, batch_metrics = generate_batch_results(
            user, item, rating, text, predicted_rating, predicted_context, predicted_text, data)

        results.extend(batch_results)
        metrics.extend(batch_metrics)
        

        # for i in range(batch_size):
        #     results.append({
        #         'user': decoded_user[i],
        #         'item': decoded_item[i],
        #         'predicted_rating': predicted_rating[i].item(), # cal l'item pq si no és tensor i no és serialitzable
        #         'real_rating': rating[i].item(),
        #         'predicted_context': untokenized_predicted_context[i], # en realitat n'hi ha més, simplement mostro els més alts
        #         # real_context no té sentit pq simplement son les paraules més freqüents del text
        #         'predicted_text': untokenized_predicted_text[i],
        #         'real_text': untokenized_text[i]
        #     })
        #     # el metrics és les coses que s'utilitzaran després per calcular més mètriques. Crec que seria millor fet tot junt
        #     # ngl i després ja calcularé les mètriques usant els results. i.e. no val la pena separar-ho
        #     # Well crec que ho havia separat pq no volia escriure en el json: tokens_predicted_text, tokens_real_text
        #     # i necessitava aquests 2 per calcular les mètriques, per això ara mateix ho tinc separat
        #     metrics.append({
        #         'tokens_predicted_text': decoded_predicted_text[i],
        #         'tokens_real_text': decoded_text[i],
        #         'predicted_text': untokenized_predicted_text[i],
        #         'real_text': untokenized_text[i]
        #     })

    parameters = {
        "num_beams": num_beams,
        "do_sample": do_sample
    }
    return parameters, metrics, results






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
        train_args = json.load(f)['parameters']
    
    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    path = os.path.join('out', args.id)

    record_execution(path)

    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.context_window)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, True)

    test_data = Subset(mydata, mysplitdata.test)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    assert args.strategy == 'greedy', 'Only greedy strategy is implemented'
    num_beams = 1
    do_sample = False

    parameters, metrics, results = generate(mydata, test_dataloader, mymodel, mydevice, num_beams, do_sample)

    RMSE, MAE = get_RMSE_MAE(results, mydata.max_rating, mydata.min_rating)

    text_quality = compute_text_quality(metrics)

    metrics_json = {
        #"losses": losses_dic, # les losses crec que seria complicat calulcar-les i no tenen molt sentit en el generate
        "RMSE": RMSE,
        "MAE": MAE,
        "text_quality": text_quality
    }
    generate_json = {
        "parameters": parameters,
        "metrics": metrics_json,
        "results": results
    }
    with open(f"out/{args.id}/results/{args.result_id}.json", 'w') as f:
        json.dump(generate_json, f, indent=4)



    # Demà seguiré per aquí + el test, i aviat he de començar a escriure la memòria ja...
    # LOL la memòria. M'he de passar un dia límit a mi mateix de posar'm-hi ja en serio,
    # o si no no entregaré el TFG aquest semestre...

    # it's predicted using: user & item only (not using the real rating, the real text or the real context)

    # La predicció de context simplement intenta predir les paraules més probables del text sense importar l'ordre,
    # de manera que l'objectiu és que sigui una tasca on no importa l'ordre i amb una única step ja predigui
    # per on anirà possiblement el text
