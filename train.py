import sys
import os
import logging
import json
import tqdm

import torch
import argparse
from torch import nn
from torch.utils.data import Subset, DataLoader

from peter_model import PETER
from utils.peter import now_time, content, loss, peter_validation_msg
from data import MyDataset, MySplitDataset, setup_logger, move_to_device
from test import test


# Abans d'afegir funcionalitat extra, la funcionalitat bàsica l'hauria de tenir més clara i
# entendre finalment com funcionen els transformers, que encara no ho tinc clar del tot en codi.

# is there any point to repeating peter_logger?


# i put one hint only, in general it's better to hint all the types
def train_epoch(dataloader: DataLoader, model, loss_fn, optimizer, device, log_interval, clip, epoch):

    peter_logger = logging.getLogger("peter_logger")
    andreu_logger = logging.getLogger("andreu_logger") # Falta acabar de definir les coses que vull fer log jo
    # pq les de peter són kinda arbitràries i no donen l'entrenament real del model
    
    model.train() # pq el torch ho sàpiga i prepari per backpropagation

    total_losses, interval_losses = torch.zeros(4), torch.zeros(4)
    interval_sample = 0

    num_batches = len(dataloader)
    
    for batch, real in enumerate(tqdm.tqdm(dataloader, position=1, mininterval=1, desc=f"Epoch {epoch} progress")):
        
        real = move_to_device(real, device) # li dono el nom real pq l'hauré de passar a un mètode (el loss_fn)
        user, item, rating, text = real
        batch_size = user.size(0)

        # M'hauria de centrar en l'important i aprendre com funciona el codi rellevant

        # print('before treure la última posicio, text shape is', text.shape)

        text = text.t() # Aquí el PETER el transposava
 
        text = text[:-1]  # (src_len + tgt_len - 2, batch_size)

        # Es treu l'últim token de cada text de cada usuari

        # pq es treu la última posició de text???? Només s'intenta predir l'últim token en el train?
        # És com si li passessin tot el text sencer, es treu NOMÉS l'últim token i s'intenta predir
        # l'últim token sense donar-li òbviament al model i tenint tots els anteriors ja donats

        # crec que és com si treguessis l'últim token i llavors l'intentes predir?
        # print('here the text thing is', text.shape)
        # print(text)

        predicted = model(user, item, text) # no s'usa el rating pel train
        batch_losses = loss_fn(predicted, real) # però sí quan es calcula la pèrdua

        c_loss, r_loss, t_loss, loss = batch_losses

        loss.backward()

        # Això ho han posat els de PETER, comprovar si realment és necessari
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() # Update els paràmetres del model segons la loss i l'optimitzador

        # Canviat al final com a l'exemple bo, aviam si així s'arregla. YAY s'ha solucionat
        optimizer.zero_grad()
        
        total_losses += torch.tensor(batch_losses) * batch_size

        # És millor la progress bar de tqdm que anar imprint coses cada cert interval. Potser puc borrar ja
        interval_losses += torch.tensor(batch_losses) * batch_size
        interval_sample += batch_size
        if (batch + 1) % log_interval == 0 or batch == num_batches - 1: # ara començo batch a 0 enlloc de 1
            interval_average_losses = interval_losses / interval_sample
            context_loss, text_loss, rating_loss, real_loss = interval_average_losses
            peter_logger.info(f"{now_time()}{content(context_loss, text_loss, rating_loss)} | {(batch+1):5d}/{num_batches:5d} batches")
            interval_losses = torch.zeros(4)
            interval_sample = 0
        
    return (total_losses / len(dataloader.dataset)).tolist()


def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, epochs, endure_times, log_interval, \
          device, model_path, rating_reg):

    peter_logger = logging.getLogger("peter_logger")
    andreu_logger = logging.getLogger("andreu_logger")

    #andreu_logger.info(now_time() + 'epoch 0')
    # print("it's computing the loss for the validation set")
    # print("li estic passant la loss_fn del model")
    val_losses = test(val_dataloader, model, loss_fn, device)
    # print("it finished computing the loss for the validation set")
    real_loss = val_losses[3]  # real_loss for the Gradient Descent
    #andreu_logger.info(f"{now_time()}real_loss on validation: {real_loss}") # real_loss:4.4

    if epochs == 0:
        andreu_logger.info(now_time() + 'No epochs to train')
    
    else:
        best_val_loss = real_loss
        endure_count = 0
        #andreu_logger.info(now_time() + 'Start training') # té problemes amb el tqdm tot el que posi al mateix temps per pantalla

    # mininterval so it's a bit less distracting when updating
    for epoch in tqdm.tqdm(range(1, epochs + 1), total=epochs, position=0, desc="Training progress"):

        peter_logger.info(f"{now_time()}epoch {epoch}")

        train_losses = train_epoch(train_dataloader, model, loss_fn, optimizer, device, log_interval, args.clip, epoch)
        #andreu_logger.info(f"{now_time()}real_loss on training: {train_losses[3]}") # real_loss:4.4f

        val_losses = test(val_dataloader, model, loss_fn, device)
        real_loss = val_losses[3]  # real_loss for the Gradient Descent

        #andreu_logger.info(f"{now_time()}real_loss on validation: {real_loss}") # real_loss:4.4f
        peter_logger.info(peter_validation_msg(val_losses, rating_reg))

        if real_loss < best_val_loss:
            best_val_loss = real_loss
            andreu_logger.info(now_time() + 'better valiation loss, saving the model')
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            peter_logger.info(f"{now_time()}Endured {endure_count}/{endure_times} time(s)")
            if endure_count == endure_times:
                peter_logger.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            
            scheduler.step()

            andreu_logger.info(f"{now_time()}Learning rate set to {scheduler.get_last_lr()[0]}") # Torna mútliples grups de params
    


def parse_arguments():

    # Potser hauria de separar més entre hiperparàmetres del model i hiperparàmetres de l'entrenament
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str, help='path for loading the pickle data')
    parser.add_argument('tokenizer', choices=['bert-base-uncased'], help='tokenizer to use')
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
    parser.add_argument('--log_interval', type=int, default=200, help='report interval')

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

    mylogs = os.path.join(mypath, 'logs')
    os.makedirs(mylogs)

    # si em va imprimint coses per pantalla kinda trenca el tqdm (en el sentit que n'ha de fer un altre)
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log') # potser per pantalla de moment puc posar els 2
    andreu_logger = setup_logger('andreu_logger', f'{mylogs}/andreu.log', True)

    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')
    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    # Si en un altre lloc ja faig el logs dels arguments i aquest el vull més aviat per carregar-los en el test,
    # potser no tots els arguments són necessaris pel test (n'hi ha que són específics del train)
    with open(f'out/{args.train_id}/train.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Això no ho vull veure per pantalla cada cop que executo jo... Només ho posaré en el fitxer i ja
    peter_logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        peter_logger.info('{:40} {}'.format(arg, getattr(args, arg)))
    peter_logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if args.cpu:
            peter_logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    mymodel_path = os.path.join(mypath, 'model.pt')


    ###############################################################################
    # Load data
    ###############################################################################

    peter_logger.info(now_time() + 'Loading data')

    data = MyDataset(args.data_path, args.tokenizer, args.context_window) #, args.text_vocab_size)

    mysplitdata = MySplitDataset(args.data_path, len(data), args.split_id, True)

    train_data = Subset(data, mysplitdata.train)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = Subset(data, mysplitdata.valid)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    # hauria de simplificar les coses poc importants i donar més importància les realment important


    ###############################################################################
    # Build the model
    ###############################################################################

    # Crec que carregar directament un altre model no té del tot sentit, ja que depèn dels users, items i word tokens
    # Fer transfer learning de les word tokens (dels embeddings apresos) seria fàcil i es podria fer
    # En canvi els users i els items ja és molt més complicat i no val la pena intentar-ho

    andreu_logger.info(now_time() + 'Building model')
    
    src_len = 2  # [u, i]

    tgt_len = args.context_window + 1  # added <bos> or <eos>
    ntokens = len(data.token_dict)
    nuser = len(data.user_dict)
    nitem = len(data.item_dict)

    # here i use by default the PETER mask so that i don't have to call it with --peter_mask always
    mymodel = PETER(not args.left_to_right_mask, src_len, tgt_len, nuser, nitem, ntokens, args.emsize,
                    args.nhead, args.nhid, args.nlayers, args.dropout, data.token_dict.bos,
                    data.token_dict.eos, data.token_dict.pad, data.token_dict.unk).to(mydevice)

    mytext_criterion = nn.NLLLoss(ignore_index=data.token_dict.pad)  # ignore the padding when computing loss
    myrating_criterion = nn.MSELoss()


    ###############################################################################
    # Training code
    ###############################################################################

    # variables de args: context_reg, text_reg, rating_red
    # peter_loss = lambda predicted, real: loss(predicted, real, args.context_reg, args.text_reg, args.rating_reg,
    #                                           mytext_criterion, myrating_criterion, ntokens, tgt_len)
    
    # # Per si necessito fer prints a mitges only. Només posa els arguments de args
    def peter_loss(predicted, real):
        return loss(predicted, real, args.context_reg, args.text_reg, args.rating_reg, 
                    mytext_criterion, myrating_criterion, ntokens, tgt_len)
    

    # Aquests són els paràmetres que usava PETER, no els he provat a canviar yet
    myoptimizer = torch.optim.SGD(mymodel.parameters(), lr=args.lr) #, momentum=0.9) # de moment no li poso el momentum
    # scheduler adjusts the learning rate according to a schedule that you define
    #   optimizer: SGD or Adam, the optimizer object that you are using to train your model
    #   step_size: the number of epochs after which you want to decrease your learning rate

    myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)
    
    train(mymodel, peter_loss, myoptimizer, myscheduler, train_dataloader, val_dataloader, args.epochs, args.endure_times,
          args.log_interval, mydevice, mymodel_path, args.rating_reg)
