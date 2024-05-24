import argparse
import os
import torch
import torch.nn as nn
import json
import tqdm
from torch.utils.data import DataLoader, Subset

from utils.peter import now_time
from data import MyDataset, MySplitDataset, record_execution, decode_batch_results, get_RMSE_MAE
from losses import peter_loss
from generate import get_topk_tokens, compute_text_quality


def test(dataloader:DataLoader, model, loss_fn, device, save_results=False, data:MyDataset=None):
    
    model.eval()

    if save_results:
        assert data is not None
        results, metrics = [], []

    total_losses = torch.zeros(4)
    
    with torch.no_grad():

        iterable = dataloader
        if save_results: # Only put the progress bar when saving results
            iterable = tqdm.tqdm(dataloader, desc='Test', mininterval=1)

        for batch in iterable:

            batch = [elem.to(device) for elem in batch] # moure a cuda
            user, item, rating, text = batch
            batch_size = user.size(0)

            # target_text is transposed and shifted right by 1 (as per each query (token) I try to predict the next one)
            # example:
            # the model will receive ["I", "am", "trying", "to", "predict"] as input
            # and will try to predict ["am", "trying", "to", "predict", "this"] (text shifted right)
            # this is the meaning of the output shifted right
            transposed_text = text.t()
            input_text = transposed_text[:-1]
            target_text = transposed_text[1:]

            predicted = model(user, item, rating, input_text, mode='parallel') #1234
            log_word_prob, log_context_dis, predicted_rating, _ = predicted

            loss_input = [log_word_prob, log_context_dis, predicted_rating]
            loss_output = [target_text, rating]
            losses = loss_fn(loss_input, loss_output)

            total_losses += torch.tensor(losses) * batch_size

            if not save_results:
                continue
            

            # Construeixo "manualment" predicted_context, predicted_text per veure el que fa realment el model quan entrena,
            # tot i que com a text generat automàtic és bastant fals pq li estic donant pistes que un cas de generació no té

            # LOL al final el que s'optimitza del text només és un classificador de les probabilitats de les paraules
            # tècnicament pel test es podria usar sampling enlloc de tot greedy? o llavors no tindria tant sentit?

            predicted_context = get_topk_tokens(log_context_dis, topk=data.max_tokens)

            word_prob = log_word_prob.exp() # [6, 128, 9994]
            word_idx = torch.argmax(word_prob, dim=2) # [6, 128] (greedy decode)

            # falta T i posar-li el <bos>, que ja s'ha suposat que el tindria i s'ha predit les següents posicions
            bos_tensor = torch.full((1, batch_size), data.token_dict.bos)
            predicted_text = torch.cat([bos_tensor, word_idx.cpu()], 0).T

            batch_results, batch_metrics = decode_batch_results(
                user, item, rating, text, predicted_rating, predicted_context, predicted_text, data)

            results.extend(batch_results)
            metrics.extend(batch_metrics)


    losses = total_losses / len(dataloader.dataset)

    losses_dic = {
        'loss': losses[0].item(),
        'context_loss': losses[1].item(),
        'text_loss': losses[2].item(),
        'rating_loss': losses[3].item()
    }
    if not save_results:
        return losses_dic # seria millor tornar un results buit? no crec
    
    return losses_dic, results, metrics


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

    with open(f'out/{cmd_args.train_id}/train_args.json', 'r') as f:
        train_args = json.load(f)

    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join('out', args.train_id)
    record_execution(path)

    if torch.cuda.is_available() and args.cpu:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.max_tokens)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, load_split=True)

    test_data = Subset(mydata, mysplitdata.test)
    mytest_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Amb freqüències, amb poques èpoques va pitjor pq no aprèn ni a dir coses populars
    # frequencies = torch.tensor(list(mydata.token_dict.entity_count.values()), dtype=torch.float, device=mydevice)
    # weights = 1 / frequencies
    # weights /= weights.sum()
    # text_criterion = nn.NLLLoss(weight=weights, ignore_index=mydata.token_dict.pad)

    text_criterion = nn.NLLLoss(ignore_index=mydata.token_dict.pad)
    # text_criterion = nn.CrossEntropyLoss(ignore_index=mydata.token_dict.pad)  # ignore the padding when computing loss

    rating_criterion = nn.MSELoss()

    myloss_fn = lambda loss_input, loss_output: peter_loss(
        loss_input, loss_output, text_criterion, rating_criterion,
        args.context_reg, args.text_reg, args.rating_reg
    )
    losses_dic, results, metrics = test(mytest_dataloader, mymodel, myloss_fn, mydevice, save_results=True, data=mydata)
    # test fent el save_results, en tokenizer-bert-base-uncased window_size=5: 42s (només cal tenir
    # en compte per no fer-ho quan no cal); el mateix sense fer el save_results: 5s
    
    RMSE, MAE = get_RMSE_MAE(results, mydata.max_rating, mydata.min_rating)

    # ara que he optimitzat el compute_text_quality el podria caclular sense cap problema, tot i que és una mica de cheating
    # pq li passo per generar cada token individualment els tokens anteriors reals, no els que ha predit el meu model
    text_quality = compute_text_quality(metrics)

    metrics_json = {
        "losses": losses_dic,
        "RMSE": RMSE,
        "MAE": MAE,
        "text_quality": text_quality
    }
    results_json = {
        "metrics": metrics_json,
        "results": results
    }
    with open(f"out/{args.train_id}/test.json", 'w') as f:
        json.dump(results_json, f, indent=4)
