import argparse
import os
import torch
import torch.nn as nn
import json
import tqdm
from torch.utils.data import DataLoader, Subset

from utils.peter import now_time, root_mean_square_error, mean_absolute_error
from data import MyDataset, MySplitDataset, record_execution
from losses import peter_loss
from generate import get_topk_tokens


# en el case de save_results necessito el data per fer desencriptacions
def test(dataloader:DataLoader, model, loss_fn, device, save_results=False, data:MyDataset=None):
    
    model.eval()

    if save_results:
        assert data is not None
        results = [] # crec que voldré guardar més coses?

    total_losses = torch.zeros(4)
    
    with torch.no_grad():

        iterable = dataloader
        if save_results: # Only put the progress bar when saving results
            iterable = tqdm.tqdm(dataloader, desc='Test', mininterval=1)

        for batch in iterable:

            batch = [elem.to(device) for elem in batch] # moure a cuda

            user, item, rating, text = batch
            batch_size = user.size(0)

            # YAY ara ja ho he entès pq es fa així i ho he deixat tot molt més net, good job me

            transposed_text = text.t()

            # target_text is shifted right by 1 (as per each query (token) I try to predict the next one)
            input_text = transposed_text[:-1]
            target_text = transposed_text[1:]

            # example:
            # the model will receive ["I", "am", "trying", "to", "predict"] as input
            # and will try to predict ["am", "trying", "to", "predict", "this"] (text shifted right)
            # this is the meaning of the output shifted right

            predicted = model(user, item, input_text)

            log_word_prob, log_context_dis, predicted_rating, _ = predicted

            loss_input = [log_word_prob, log_context_dis, predicted_rating]
            loss_output = [target_text, rating]

            losses = loss_fn(loss_input, loss_output)

            total_losses += torch.tensor(losses) * batch_size

            if not save_results:
                continue
            

            

            # if seq_prediction (és el cas casi sempre) -> func.log_softmax(self.hidden2token(hidden[self.src_len:]), dim=-1) (TOT)
            # else                                      -> func.log_softmax(self.hidden2token(hidden[-1]), dim=-1) (1 cosa...)

            # what is the point of the 2nd one? crec que el primer que sempre usa tot és millor

            # log_context_dis -> distribució de probabilitats predida dels tokens pel text en base només a user, item
            # rating -> predicció de nota, basant NOMÉS en el user, item (NO s'usa per res el text)
            # attns -> atencions que utilitza en transformer_encoder

            # En aquest cas si es fa servir per algo la predicció de l'últim token és una mica estúpid pq només està intenant
            # predir l'última paraula del text. Crec que tindria més sentit intentar-los predir tots alhora però per separat,
            # i després quan vulguis generar text amb sentit sí que té sentit generar-los de forma seqüencial
            
            
            # Hi ha moltes coses que són exactament igual que al generate, possiblement hauria de no duplicar codi si és possible

            decoded_user = [data.user_decode(u) for u in user]
            decoded_item = [data.item_decode(i) for i in item]

            max_length = data.context_window
            predicted_context = get_topk_tokens(log_context_dis, topk=max_length)

            decoded_predicted_context = [data.text_decode(list(c), mode='literal') for c in predicted_context]
            untokenized_predicted_context = [data.untokenize(c) for c in decoded_predicted_context]


            # en el generate: predicted_rating, predicted_context, predicted_text = predicted,
            # predicted = generate_batch(...)
            # en canvi aquí: log_word_prob, log_context_dis, predicted_rating, attns = predicted

            # el predicted_text venia de allí: predicted_rating, predicted_context, predicted_text = predicted 
            log_word_prob, log_context_dis, predicted_rating, _ = predicted # això és el que retorna el model
            # predicted = model(user, item, text) JO estic avaluant el meu forward directament.
            # En el altre lloc venia de més alt nivell, 

            # 1. text = torch.full((1, batch_size), bos_idx).to(device)
            # for step in range(max_length):
            #   text = torch.cat([text, next_token.unsqueeze(0)], 0) # aquí és on es concatena
            # text.T

            # Aviam aquí doncs com construeixo jo el predicted_text a partir del raw model(user, item, text)

            # model(user, item, text) -> log_word_prob, log_context_dis, rating, attns
            # He de descodificar amb greedy el log_word_prob crec en cada posició?

            # això és el que feia el generate: _, next_token = torch.max(log_word_prob, dim=-1)

            # Aviam estava aquí amb el predicted_text
            # per veure algo de resultats té sentit fer un greedy decode de cada token individualment,
            # la frase no tindrà probably sentit però per veure per on està tirant el meu model assisidament

            # log_word_prob -> predicted_text


            # LOL al final el que s'optimitza del text només és un classificador de les probabilitats de les paraules

            # Aquí he descofificat de manera greedy jo manualment les paraules del model:
            word_prob = log_word_prob.exp() # [6, 128, 9994]
            word_idx = torch.argmax(word_prob, dim=2) # [6, 128]

            # falta posar-li el <bos>, que ja s'ha suposat que el tindria i s'ha predit les següents posicions
            bos_value = data.token_dict.bos
            bos_tensor = torch.full((1, batch_size), bos_value)
            predicted_text = torch.cat([bos_tensor, word_idx.cpu()], 0).T

            decoded_predicted_text = [data.text_decode(list(t), mode='sceptic') for t in predicted_text] # , raw=True
            untokenized_predicted_text = [data.untokenize(t) for t in decoded_predicted_text]

            decoded_text = [data.text_decode(list(t), mode='correct') for t in text]
            untokenized_text = [data.untokenize(t) for t in decoded_text]

            # real_text: text.T -> decoded_text -> untokenized_text
            # predicted_text: predicted_text -> decoded_predicted_text -> untokenized_predicted_text (seems correct?)

            # tècnicament pel test es podria usar sampling enlloc de tot greedy? o llavors no tindria tant sentit?

            if save_results:
                for i in range(batch_size):
                    results.append({
                        'user': decoded_user[i],
                        'item': decoded_item[i],
                        'predicted_rating': predicted_rating[i].item(),
                        'real_rating': rating[i].item(),
                        'predicted_context': untokenized_predicted_context[i],
                        'predicted_text': untokenized_predicted_text[i],
                        'real_text': untokenized_text[i]
                    })

            # Quan ho vaig fer directe vaig tenir un munt de problemes amb memòria GPU i memòria RAM:
            # # LOL aquesta 1 línia m'ha portat un munt de problemes
            # for i in range(4): # era un extend enlloc de append
            #     predictions[i].extend(predicted[i].cpu()) # important passar-lo a cpu,
                # si no no ho allibera de la GPU en cada batch i per tant acaba petant per memòria

    losses = total_losses / len(dataloader.dataset)

    losses_dic = {
        'loss': losses[0].item(),
        'context_loss': losses[1].item(),
        'text_loss': losses[2].item(),
        'rating_loss': losses[3].item()
    }

    if not save_results:
        return losses_dic # seria millor tornar un results buit? no crec
    
    return losses_dic, results


# Combinar els arguments amb els de train, pq hi ha moltes coses que es necessitaven de train
def parse_arguments():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('train_id', type=str)
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA') # hauria de comprovar si es pot canviar
    # només en inferència, estic bastant segur que sí que es pot canviar

    cmd_args = cmd_parser.parse_args()

    path = os.path.join('out', cmd_args.train_id)
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')

    with open(f'out/{cmd_args.train_id}/train.json', 'r') as f:
        train_args = json.load(f)['parameters']

    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join('out', args.train_id)

    record_execution(path)

    if torch.cuda.is_available():
        if args.cpu:
            print(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.context_window)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, load_split=True)

    test_data = Subset(mydata, mysplitdata.test)
    mytest_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    tgt_len = args.context_window + 1 # exactly same as train now
    ntokens = len(mydata.token_dict)

    text_criterion = nn.NLLLoss(ignore_index=mydata.token_dict.pad)
    rating_criterion = nn.MSELoss()

    myloss_fn = lambda loss_input, loss_output: peter_loss(
        loss_input, loss_output, text_criterion, rating_criterion,
        args.context_reg, args.text_reg, args.rating_reg, ntokens, tgt_len
    )
    # Aquí en el test sí que passo el data ara mateix per fer descodificacions. en canvi en el train no el passava
    # Per caluclar el MAE i el RMSE necessito els valors predits a part de les losses
    losses_dic, results = test(mytest_dataloader, mymodel, myloss_fn, mydevice, save_results=True, data=mydata)

    rating_predictions = [result['predicted_rating'] for result in results]
    real_ratings = [result['real_rating'] for result in results]
    real_predicted_rating = [(r, p) for (r, p) in zip(real_ratings, rating_predictions)]

    RMSE = root_mean_square_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)
    MAE = mean_absolute_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)

    metrics = {
        "losses": losses_dic,
        "RMSE": RMSE,
        "MAE": MAE
    }
    results_json = {
        "metrics": metrics,
        "results": results
    }
    with open(f"out/{args.train_id}/test.json", 'w') as f:
        json.dump(results_json, f, indent=4)
