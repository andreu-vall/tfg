import os
import json
import tqdm
import torch
import argparse
from torch import nn
from torch.utils.data import Subset, DataLoader
import time

from peter_model import PETER
from data import MyDataset, MySplitDataset, record_execution
from test import test
from losses import peter_loss


# i put one hint only, in general it's better to hint all the types
def train_epoch(dataloader: DataLoader, model, loss_fn, optimizer, device, clip, epoch):
    
    model.train()

    total_losses = torch.zeros(4)

    for batch in tqdm.tqdm(dataloader, position=1, mininterval=1, desc=f"Epoch {epoch} progress"):

        batch = [elem.to(device) for elem in batch] # moure a cuda

        user, item, rating, text = batch
        batch_size = user.size(0)

        transposed_text = text.t() # Generarem text a nivell de token per tots els usuaris
        input_text = transposed_text[:-1]
        target_text = transposed_text[1:] # output text is shifted right

        predicted = model(user, item, rating, input_text, mode='parallel') #1234
        log_word_prob, log_context_dis, predicted_rating, attns = predicted

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


# Ara mateix patience=0, seria interessant mai canviar-ho i posar-ho com un paràmetre?
def train(model, loss_fn, optimizer, train_dataloader, val_dataloader, max_epochs,
          device, path, lr_decay_factor, lr_improv_threshold, min_lr):
    
    model_path = f"{path}/model.pt"
    metrics_path = f"{path}/train_metrics.json"
    
    metrics = [] # metrics serà en cada posició un diccionari

    val_losses = test(val_dataloader, model, loss_fn, device)
    best_val_loss = val_losses['loss']
    metrics.append({
        "epoch": 0,
        "valid": val_losses
    })
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(model_path, 'wb') as f:
        torch.save(model, f)

    for epoch in tqdm.tqdm(range(1, max_epochs + 1), total=max_epochs, position=0, desc="Training progress"):

        start_time = time.time()

        train_losses = train_epoch(train_dataloader, model, loss_fn, optimizer, device, args.clip, epoch)
        val_losses = test(val_dataloader, model, loss_fn, device)

        end_time = time.time()
        epoch_time = end_time - start_time
        
        metrics.append({
            "epoch": epoch,
            "time": epoch_time, # en segons
            "learning_rate": optimizer.param_groups[0]['lr'], # Torna mútliples grups de params
            "train": train_losses,
            "valid": val_losses
        })
        with open(metrics_path, 'w') as f: # Ho guardo cada vegada per si l'entrenament s'atura malament que quedi tot escrit, és poc
            json.dump(metrics, f, indent=4)
        #train_loss = train_losses['loss'] # de moment no la utilitzo per les decisions, crec que sempre s'anirà reduint aquest

        previous_val_loss = metrics[-2]['valid']['loss']
        new_val_loss = metrics[-1]['valid']['loss']

        # si no millora més que el threshold, en particular si empitjora:
        improvement = (previous_val_loss - new_val_loss) / previous_val_loss # positiu si es redueix, negatiu si empitjora
        if improvement < lr_improv_threshold:
            optimizer.param_groups[0]['lr'] *= lr_decay_factor
            if optimizer.param_groups[0]['lr'] < min_lr:
                break
        
        # només guardo el millor model
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss
            with open(model_path, 'wb') as f:
                torch.save(model, f)
    

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str, help='path for loading the pickle data')
    parser.add_argument('tokenizer', choices=['tokenizer-bert-base-uncased'], help='tokenizer to use')
    parser.add_argument('max_tokens', type=int, help='real tokens without counting <bos> and <eos>')
    #parser.add_argument('context_window', type=int) # és molt important
    parser.add_argument('split_id', type=str, help='load indexes')
    parser.add_argument('train_id', type=str, help='model id')

    parser.add_argument('recommender_type', choices=['PETER', 'andreu'])

    parser.add_argument('--bert_embeddings', action='store_true', help='use bert embeddings')

    parser.add_argument('--max_epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--initial_lr', type=float, default=1.0, help='initial learning rate') # Si learning_rate >= 1, matemàticament
    # no necessàriament hauria de convergir (suma inf teòrica), tot i que per jugar potser ho puc provar
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='factor by which to decrease the learning rate') # peter tenia 0.25
    
    # AVIAM PROVO DE REDUIR-HO MOLT MÉS PQ ES COMENCI ABANS A REDUIR EL LEARN RATE
    # CREC QUE ENCARA EL PUC AUGMENTAR BASTANT MÉS, O ALTERNATIVAMENT COMPARAR TANT EL IMPROVEMENT EN TRAIN COM VALID
    # PER VEURE SI ESTÀ OVERFITTING
    parser.add_argument('--lr_improv_threshold', type=float, default=0.05, help='minimum improvement to keep the learning rate') # 0.01?
    # si no es redueix la loss en un 2.5% redueixo a la meitat el learning_rate. Crec que estaria bé tmb veure
    # el del tranining, però sembla que no és gaire comú
    # en aquest anterior crec que serà interessant provar diferents valors (més grans)
    parser.add_argument('--min_lr', type=float, default=1/2**8, help='minimum learning rate after which the training stops')

    # Crec que és millor aturar amb el min_lr no amb un número de cops de reduir el learning_rate
    # parser.add_argument('--endure_times', type=int, default=5, help='the maximum endure times of loss increasing on validation')
    # podria definir una paciència, però crec que en general prefereixo sempre la paciència de 0
    # també podria provar amb el momentum, que el exemple oficial de torch l'utilitzaven i potser sigui millor així

    # Aquestes tres crec que també són molt importants per definir a què li dona importància el teu model quan aprèn
    parser.add_argument('--rating_reg', type=float, default=0.5, help='regularization on recommendation task')
    parser.add_argument('--context_reg', type=float, default=0.2, help='regularization on context prediction task')
    parser.add_argument('--text_reg', type=float, default=1, help='regularization on text generation task')
    # PETER: 0.1, 1, 1
    # Andreu 1a suggerència: 0.5, 0.2, 1

    # Aquest deien ells que era important, en algun moment o altre he de provar si realment sense això fa més overfitting
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')

    # Paràmetres del model, de moment no els he tocat
    parser.add_argument('--emsize', type=int, default=512, help='size of embeddings')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the transformer')
    parser.add_argument('--nhid', type=int, default=2048, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')

    # Per reproductibilitat només
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # He invertit l'argument pq és més normal usar GPU
    parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA')

    # Si vulgués limitar el vocabulari de text, users o items, tinc els mètodes implementats en MyDataset.
    # Tot i així no crec que sigui una bona manera d'aconseguir res, ja que si tens massa d'una cosa pots
    # reduir-te el dataset o agafar un millor tokenitzador
    # parser.add_argument('--text_vocab_size', type=int, help='number of tokens to keep in the text vocabulary')

    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_arguments()

    mypath = os.path.join('out', args.train_id)
    if os.path.exists(mypath):
        raise ValueError('This id already exists!')
    os.makedirs(mypath)

    record_execution(mypath)

    with open(f"{mypath}/train_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    torch.manual_seed(args.seed) # Set the random seed manually for reproducibility.

    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    ###############################################################################
    # Load data
    ###############################################################################

    # puc posar my als que es passen com variables per assegurar que no siguin globals
    # la resta no cal. i un cop vegi que no es passen globals ja no caldria tampoc els my

    data = MyDataset(args.data_path, args.tokenizer, args.max_tokens)

    split_data = MySplitDataset(args.data_path, len(data), args.split_id, True)

    train_data = Subset(data, split_data.train)
    mytrain_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # is it using the torch seed?

    val_data = Subset(data, split_data.valid)
    myval_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    # Crec que carregar directament un altre model no té del tot sentit, ja que depèn dels users, items i word tokens
    # Fer transfer learning de les word tokens (dels embeddings apresos) seria fàcil i es podria fer
    # En canvi els users i els items ja és molt més complicat i no val la pena intentar-ho
    
    ntokens = len(data.token_dict)
    nuser = len(data.user_dict)
    nitem = len(data.item_dict)

    mymodel = PETER(args.max_tokens, nuser, nitem, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout,
                    data.token_dict.pad, args.recommender_type, args.bert_embeddings, data.token_dict.idx_to_entity).to(mydevice)

    ###############################################################################
    # Training code
    ###############################################################################

    # Crec que aquest text_criterion és simplement una funció que s'aprèn per classificar en N classes,
    # és bastant possible que estigui esbiaixada cap als tokens més comuns, potser somehow es podria
    # intentar que no ho estigués tant? Li puc passar un weight paràmetre

    # Espera un input que sigui log-probabilities of each class, que segons la documentació oficial de
    # pytorch pots conseguir de manera fàcil amb el log_softmax a la last layer de la network.
    # Crec que el intended use és més aviat per un problema de classificació de N classes (on N és petit),
    # i inclús en aquest cas ja m'avisa que possiblement aprendrà més a fer les més populars.
    # Important ingorar el index de padding, perquè és simplement per fer que les frases tinguin la mateixa
    # length, i en general no has d'aprendre mai a posar padding, simplement es posa si en algun cas hi ha un <eos>.

    # La loss function no té cap paràmetre a entrenar, simplement és una mesura overall de si estàs encertant o no
    # les paraules que exactament volies predir. simplement agafen un input, output i retornen un valor
    # puc provar amb lo dels weights aviam què

    # Yikes a primer cop d'ull amb poques èpoques sembla que empitjora, pq ara no sap ni predir les coses
    # populars, que era el que estava aconseguint el PETER

    # Si fos una classificació en poques classes (com el rating per exemple si no ho fes coninu),
    # aleshores segurament lo dels pesos sí que seria útil. Però en el meu cas no serveix aplicar-li
    # weights així, perquè dona molta importància a tokens que només apareixen 1 cop, i llavors doncs
    # les frases generades no tenen cap mena de sentit

    # Per tant millor que estigui biased cap a les reviews populars i ja està...
    
    # frequencies = torch.tensor(list(data.token_dict.entity_count.values()), dtype=torch.float, device=mydevice)
    # weights = 1 / frequencies
    # weights /= weights.sum()
    # text_criterion = nn.NLLLoss(weight=weights, ignore_index=data.token_dict.pad)

    # El de PETER, sense weights. Estarà esbiaixat cap als tokens més comuns, però almenys dona coses amb sentit
    text_criterion = nn.NLLLoss(ignore_index=data.token_dict.pad)

    # Cross Entropy, llavors cal eliminar el log_softmax de l'última capa de la xarxa
    #text_criterion = nn.CrossEntropyLoss(ignore_index=data.token_dict.pad)


    rating_criterion = nn.MSELoss()

    myloss_fn = lambda loss_input, loss_output: peter_loss(
        loss_input, loss_output, text_criterion, rating_criterion,
        args.context_reg, args.text_reg, args.rating_reg
    )

    myoptimizer = torch.optim.SGD(mymodel.parameters(), lr=args.initial_lr) #, momentum=0.9) # de moment no li poso el momentum
    # optimizer: SGD or Adam, the optimizer object that you are using to train your model
    # encara no he provat altres optimitzadors, pot valdre la pena

    # torno a usar el OnPleateau però ara amb un threshold, 
    #myscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(myoptimizer, 'min', patience=5, factor=0.1, threshold=0.01)
    #myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)

    train(mymodel, myloss_fn, myoptimizer, mytrain_dataloader, myval_dataloader, args.max_epochs,
          mydevice, mypath, args.lr_decay_factor, args.lr_improv_threshold, args.min_lr)
