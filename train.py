import os
import json
import tqdm
import torch
import argparse
from torch import nn
from torch.utils.data import Subset, DataLoader

from peter_model import PETER
from utils.peter import now_time
from data import MyDataset, MySplitDataset, record_execution
from test import test
from losses import peter_loss


# i put one hint only, in general it's better to hint all the types
def train_epoch(dataloader: DataLoader, model, loss_fn, optimizer, device, clip, epoch):
    
    model.train()

    total_losses = torch.zeros(4)

    # mininterval de 1 segon pq no cal que es vagi actualitzant cada cop que fa un batch
    for batch in tqdm.tqdm(dataloader, position=1, mininterval=1, desc=f"Epoch {epoch} progress"):

        batch = [elem.to(device) for elem in batch] # moure a cuda

        user, item, rating, text = batch
        batch_size = user.size(0)

        transposed_text = text.t() # Generarem text a nivell de token per tots els usuaris
        input_text = transposed_text[:-1]
        target_text = transposed_text[1:] # output text is shifted right

        predicted = model(user, item, input_text)
        log_word_prob, log_context_dis, predicted_rating, attns = predicted # per si vull les atencions

        loss_input = [log_word_prob, log_context_dis, predicted_rating]
        loss_output = [target_text, rating]
        batch_losses = loss_fn(loss_input, loss_output)

        batch_loss = batch_losses[0]
        batch_loss.backward() # El típic gradient descent que torch ja ho fa sol

        # Això ho han posat els de PETER, cal comprovar si realment és necessari
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() # Update els paràmetres del model segons la loss i l'optimitzador
        optimizer.zero_grad() # cal posar a 0 manualment els gradients de torch, pq si no els guarda per la següent iteració
        
        total_losses += torch.tensor(batch_losses) * batch_size
        

    losses = total_losses / len(dataloader.dataset)

    losses_dic = {
        'loss': losses[0].item(),
        'context_loss': losses[1].item(),
        'text_loss': losses[2].item(),
        'rating_loss': losses[3].item()
    }
    return losses_dic


# wait ara mateix a l'època 2 ja hi ha hagut un overfit? kinda strange que entreni tant ràpid no?
# hauria de guardar totes les dades importants de l'entrenament en un json crec
def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, epochs, endure_times, \
          device, model_path):
    
    metrics = [] # metrics serà en cada posició un diccionari

    val_losses = test(val_dataloader, model, loss_fn, device)
    metrics.append({
        "epoch": 0,
        "valid": val_losses
    })

    with open(model_path, 'wb') as f:
        torch.save(model, f)
    
    best_val_loss = val_losses['loss']
    endure_count = 0

    for epoch in tqdm.tqdm(range(1, epochs + 1), total=epochs, position=0, desc="Training progress"):

        train_losses = train_epoch(train_dataloader, model, loss_fn, optimizer, device, args.clip, epoch)
        val_losses = test(val_dataloader, model, loss_fn, device)

        metrics.append({
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0], # Torna mútliples grups de params
            "train": train_losses,
            "valid": val_losses
        })

        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss'] 
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            if endure_count == endure_times:
                print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            
            scheduler.step()
    
    return metrics
    


# Hauria de treure els arguments no rellevants per mi, pq el train.py -h és massa llarg
def parse_arguments():

    # Potser hauria de separar més entre hiperparàmetres del model i hiperparàmetres de l'entrenament
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str, help='path for loading the pickle data')
    parser.add_argument('tokenizer', choices=['tokenizer-bert-base-uncased'], help='tokenizer to use')
    parser.add_argument('context_window', type=int) # és molt important
    parser.add_argument('split_id', type=str, help='load indexes')
    parser.add_argument('train_id', type=str, help='model id')

    # why would I ever want to limit the vocabulary of the text, users or items? just use a simpler tokenizer then
    parser.add_argument('--text_vocab_size', type=int, help='number of tokens to keep in the text vocabulary')

    # he de treure arguments i orgnaitzar millor els arguments, que la hint és massa llarga...

    # Paràmetres del model
    parser.add_argument('--emsize', type=int, default=512, help='size of embeddings')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the transformer')
    parser.add_argument('--nhid', type=int, default=2048, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')

    # El clip és de l'entrenament o del model?

    # Paràmetres d'entrenament
    parser.add_argument('--lr', type=float, default=1.0, help='initial learning rate') # Si learning_rate >= 1, matemàticament
    # no necessàriament hauria de convergir (suma inf teòrica), tot i que per jugar potser ho puc provar
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit') # estava a 50, de moment poso 1 pq sempre crido amb 1
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA') # He invertit l'argument pq és més normal usar GPU
    #parser.add_argument('--log_interval', type=int, default=200, help='report interval')

    parser.add_argument('--endure_times', type=int, default=5, help='the maximum endure times of loss increasing on validation')

    # Serveixen per definir l'objectiu a minimitzar
    parser.add_argument('--rating_reg', type=float, default=0.1, help='regularization on recommendation task')
    parser.add_argument('--context_reg', type=float, default=1.0, help='regularization on context prediction task')
    parser.add_argument('--text_reg', type=float, default=1.0, help='regularization on text generation task')


    # Crec que la PETER mask sempre era millor que la left-to-right mask! Seria millor usar-la per defecte
    # tmb canvio per default la peter_mask
    #parser.add_argument('--peter_mask', action='store_true', help='True to use peter mask; Otherwise left-to-right mask')
    parser.add_argument('--left_to_right_mask', action='store_true', help='True to use left-to-right mask; Otherwise peter mask')

    return parser.parse_args()



# YAY ja no uso variables globals enlloc
if __name__ == "__main__":
    
    args = parse_arguments()

    mypath = os.path.join('out', args.train_id)
    if os.path.exists(mypath):
        raise ValueError('This id already exists!')
    os.makedirs(mypath)

    record_execution(mypath)


    
    torch.manual_seed(args.seed) # Set the random seed manually for reproducibility.
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    mymodel_path = os.path.join(mypath, 'model.pt')


    ###############################################################################
    # Load data
    ###############################################################################

    # puc posar my als que es passen com variables per assegurar que no siguin globals
    # la resta no cal. i un cop vegi que no es passen globals ja no caldria tampoc

    data = MyDataset(args.data_path, args.tokenizer, args.context_window) #, args.text_vocab_size)

    split_data = MySplitDataset(args.data_path, len(data), args.split_id, True)

    train_data = Subset(data, split_data.train)
    mytrain_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = Subset(data, split_data.valid)
    myval_dataloader = DataLoader(val_data, batch_size=args.batch_size)


    ###############################################################################
    # Build the model
    ###############################################################################

    # Crec que carregar directament un altre model no té del tot sentit, ja que depèn dels users, items i word tokens
    # Fer transfer learning de les word tokens (dels embeddings apresos) seria fàcil i es podria fer
    # En canvi els users i els items ja és molt més complicat i no val la pena intentar-ho
    
    src_len = 2  # [u, i]

    tgt_len = args.context_window + 1  # added <bos> or <eos>
    ntokens = len(data.token_dict)
    nuser = len(data.user_dict)
    nitem = len(data.item_dict)

    # here i use by default the PETER mask so that i don't have to call it with --peter_mask always
    mymodel = PETER(not args.left_to_right_mask, src_len, tgt_len, nuser, nitem, ntokens, args.emsize,
                    args.nhead, args.nhid, args.nlayers, args.dropout, data.token_dict.bos,
                    data.token_dict.eos, data.token_dict.pad, data.token_dict.unk).to(mydevice)

    ###############################################################################
    # Training code
    ###############################################################################

    text_criterion = nn.NLLLoss(ignore_index=data.token_dict.pad)  # ignore the padding when computing loss
    rating_criterion = nn.MSELoss()

    myloss_fn = lambda loss_input, loss_output: peter_loss(
        loss_input, loss_output, text_criterion, rating_criterion,
        args.context_reg, args.text_reg, args.rating_reg, ntokens, tgt_len
    )

    # Aquests són els paràmetres que usava PETER, no els he provat a canviar yet
    myoptimizer = torch.optim.SGD(mymodel.parameters(), lr=args.lr) #, momentum=0.9) # de moment no li poso el momentum
    # scheduler adjusts the learning rate according to a schedule that you define
    #   optimizer: SGD or Adam, the optimizer object that you are using to train your model
    #   step_size: the number of epochs after which you want to decrease your learning rate

    myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)

    train_metrics = train(mymodel, myloss_fn, myoptimizer, myscheduler, mytrain_dataloader, myval_dataloader,
          args.epochs, args.endure_times, mydevice, mymodel_path)
    
    train_json = {
        'parameters': args.__dict__,
        'metrics': train_metrics
    }
    with open(f'out/{args.train_id}/train.json', 'w') as f:
        json.dump(train_json, f, indent=4)
