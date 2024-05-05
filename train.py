import sys
import os
import datetime
import logging
import json

import torch
import argparse
from torch import nn

from models.peter import PETER
from utils.peter import DataLoader, Batchify, now_time
from utils.andreu import move_content_to_device, peter_content, peter_loss_good, setup_logger
from test import test


def train_epoch(dataloader, model, loss_fn, optimizer, device, log_interval, use_feature, clip):

    peter_logger = logging.getLogger("peter_logger")
    andreu_logger = logging.getLogger("andreu_logger") # Falta acabar de definir les coses que vull fer log jo
    
    model.train()

    total_losses, interval_losses = torch.zeros(4), torch.zeros(4)
    interval_sample = 0

    num_batches = len(dataloader)
    
    for batch, content in enumerate(dataloader):

        content = move_content_to_device(content, device)

        user, item, rating, seq, feature = content # (batch_size, seq_len), batch += 1 (comentari ràndom PETER)
        batch_size = user.size(0)
 
        if use_feature:
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

        pred = model(user, item, text)

        batch_losses = loss_fn(pred, content)
        c_loss, r_loss, t_loss, loss = batch_losses

        loss.backward()

        # Això ho han posat els de PETER, comprovar si realment és necessari
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() # Update els paràmetres del model segons la loss i l'optimitzador

        # Canviat al final com a l'exemple bo, aviam si així s'arregla. YAY s'ha solucionat
        optimizer.zero_grad()

        interval_losses += torch.tensor(batch_losses) * batch_size
        total_losses += torch.tensor(batch_losses) * batch_size
        interval_sample += batch_size

        if (batch + 1) % log_interval == 0 or batch == num_batches - 1: # ara començo batch a 0 enlloc de 1
            interval_average_losses = interval_losses / interval_sample

            context_loss, text_loss, rating_loss, real_loss = interval_average_losses

            peter_logger.info(f"{now_time()}{peter_content(context_loss, text_loss, rating_loss)} | \
                              {(batch+1):5d}/{num_batches:5d} batches")
            
            interval_losses = torch.zeros(4)
            interval_sample = 0
        
    return (total_losses / dataloader.total_elements).tolist()





# Això és el que feien en el PETER. Per la validació s'ignora el loss de context (i maybe tmb el de rating)
# He de borrar probably la variable rating_reg
def peter_validation_msg(val_losses, rating_reg):
    #peter_logger = logging.getLogger("peter_logger")
    c_loss, t_loss, r_loss, real_loss = val_losses
    printed_loss = t_loss
    if rating_reg != 0: # what even is rating_reg?
        printed_loss += r_loss
    return f"{now_time()}{peter_content(c_loss, t_loss, r_loss)} | valid loss {printed_loss:4.4f} on validation"



# Varibales de args per la cara: rating_reg, use_feature
def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, epochs, endure_times, log_interval, \
          device, model_path, use_feature, rating_reg):

    peter_logger = logging.getLogger("peter_logger")
    andreu_logger = logging.getLogger("andreu_logger")

    andreu_logger.info(now_time() + 'Epoch 0 validation')
    val_losses = test(val_dataloader, model, loss_fn, device, use_feature)
    real_loss = val_losses[3]  # real_loss for the Gradient Descent

    andreu_logger.info(f"{now_time()}real_loss on validation: {real_loss}") # real_loss:4.4f

    with open(model_path, 'wb') as f:
        torch.save(model, f)

    if epochs == 0:
        andreu_logger.info(now_time() + 'No epochs to train')
        return

    andreu_logger.info(now_time() + 'Start training')

    best_val_loss = real_loss
    endure_count = 0

    for epoch in range(1, epochs + 1):

        peter_logger.info(f"{now_time()}epoch {epoch}")

        train_losses = train_epoch(train_dataloader, model, loss_fn, optimizer, device, log_interval, args.use_feature, args.clip)
        andreu_logger.info(f"{now_time()}real_loss on training: {train_losses[3]}") # real_loss:4.4f

        val_losses = test(val_dataloader, model, loss_fn, device, use_feature)
        real_loss = val_losses[3]  # real_loss for the Gradient Descent

        andreu_logger.info(f"{now_time()}real_loss on validation: {real_loss}") # real_loss:4.4f
        peter_logger.info(peter_validation_msg(val_losses, rating_reg))

        # Save the model if the validation loss is the best we've seen so far.
        if real_loss < best_val_loss:
            best_val_loss = real_loss
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            peter_logger.info(f"{now_time()}Endured {endure_count}/{endure_times} time(s)")
            if endure_count == endure_times:
                peter_logger.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            
            scheduler.step()

            # Potser cal alliberar manualment memòria de la GPU? Probably not

            # Carrego el millor model, perquè si no és el model que ja ha fet overfit. Crec que així és molt millor
            # pel meu cas, tot i que sembla que no és massa estàndard fer-ho així en general. Crec que precisament
            # no fer-ho així contribueix en gran part en la diferència entre diferentes seeds. És a dir, estan perdent
            # el temps per una tonteria molt gran i simple i m'intenten justificar a mi que no es torna enrere,
            # quan tornar enrere és molt fàcil i crec que donarà molts millors resultats
            andreu_logger.info(f"a{now_time()}Loading the best model")
            with open(model_path, 'rb') as f:
                model = torch.load(f).to(device)

            andreu_logger.info(f"{now_time()}Learning rate set to {scheduler.get_last_lr()[0]}") # Torna mútliples grups de params

        # Si vulgués aturar el bucle d'entrenament a mitges, i continuar exactament a partir d'on estava,
        # hauria de guardar: model, optimizer, scheduler, epoch, endure_count, best_val_loss



# YAY ja no uso variables globals enlloc
if __name__ == "__main__":

    start = datetime.datetime.now()
    
    # Potser hauria de separar més entre hiperparàmetres del model i hiperparàmetres de l'entrenament

    parser = argparse.ArgumentParser()

    # Paràmetres obligatoris (sense el -- ja es posa sol required=True)
    parser.add_argument('id', type=str, help='model id')
    parser.add_argument('data_path', type=str, help='path for loading the pickle data')
    parser.add_argument('index_dir', type=str, help='load indexes')

    # També hi ha l'atribut útil choices per restringir els valors
    # Seria batant ideal organitzar millor els paràmetres

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
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA') # He invertit l'argument pq és més normal usar GPU
    parser.add_argument('--log_interval', type=int, default=200, help='report interval')

    # Whether I want a directory to load the model instead of building a new one
    # Also whether I want to resume the training at a particular step (in case of crash). But I'll do that later
    parser.add_argument('--source_checkpoint', type=str, default=None, help='directory to load the model')

    parser.add_argument('--vocab_size', type=int, default=20000, help='keep the most frequent words in the dict')
    parser.add_argument('--endure_times', type=int, default=5, help='the maximum endure times of loss increasing on validation')

    # Serveixen per definir l'objectiu a minimitzar
    parser.add_argument('--rating_reg', type=float, default=0.1, help='regularization on recommendation task')
    parser.add_argument('--context_reg', type=float, default=1.0, help='regularization on context prediction task')
    parser.add_argument('--text_reg', type=float, default=1.0, help='regularization on text generation task')


    # Crec que la PETER mask sempre era millor que la left-to-right mask! Seria millor usar-la per defecte
    parser.add_argument('--peter_mask', action='store_true', help='True to use peter mask; Otherwise left-to-right mask')


    parser.add_argument('--use_feature', action='store_true', help='False: no feature; True: use the feature')
    parser.add_argument('--words', type=int, default=15, help='number of words to generate for each sample')

    args = parser.parse_args()

    mypath = os.path.join('out', args.id)
    if os.path.exists(mypath):
        raise ValueError('This id already exists!')
    os.makedirs(mypath)

    mylogs = os.path.join(mypath, 'logs')
    os.makedirs(mylogs)
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log', True) # potser per pantalla de moment puc posar els 2
    andreu_logger = setup_logger('andreu_logger', f'{mylogs}/andreu.log', True)
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')

    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    # Si en un altre lloc ja faig el logs dels arguments i aquest el vull més aviat per carregar-los en el test,
    # potser no tots els arguments són necessaris pel test (n'hi ha que són específics del train)
    with open(f'out/{args.id}/train.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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
    corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
    word2idx = corpus.word_dict.word2idx
    pad_idx = word2idx['<pad>']
    train_dataloader = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
    val_dataloader = Batchify(corpus.valid, word2idx, args.words, args.batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.source_checkpoint is None:
        andreu_logger.info(now_time() + 'Building model')
        if args.use_feature:
            src_len = 2 + train_dataloader.feature.size(1)  # [u, i, f]
        else:
            src_len = 2  # [u, i]
        tgt_len = args.words + 1  # added <bos> or <eos>
        ntokens = len(corpus.word_dict)
        nuser = len(corpus.user_dict)
        nitem = len(corpus.item_dict)
        mymodel = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens,
                    args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(mydevice)

    else:
        andreu_logger.info(now_time() + 'Loading model')
        with open(args.source_checkpoint, 'rb') as f:
            mymodel = torch.load(f).to(mydevice)

    mytext_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
    myrating_criterion = nn.MSELoss()


    ###############################################################################
    # Training code
    ###############################################################################

    # variables de args: context_reg, text_reg, rating_red
    peter_loss = lambda pred, content: peter_loss_good(pred, content, args.context_reg, args.text_reg, args.rating_reg,
                                                    mytext_criterion, myrating_criterion, ntokens, tgt_len)
    
    
    
    # Tinc la teoria que el no descartar el model generat amb un learning_rate massa gran fa que es perdi
    # ja molt pel que fa al resultat final del model, fent un overfitting de la òstia

    # El Colab em va timar criticant el seu scheduler i que usés el ReduceLROnPlateau,
    # però ja estava prou bé i és més fàcil de llegir el codi i dona millors resultats

    # Una de les avantatges del logger és que podria usar el tqdm per exemple crec, tot i que not sure
    # del tot pq si redirigeixo el logging també a stdout llavors no podré fer servir tampoc el tqdm.

    myoptimizer = torch.optim.SGD(mymodel.parameters(), lr=args.lr) #, momentum=0.9) # de moment no li poso el momentum
    myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)
    # scheduler adjusts the learning rate according to a schedule that you define
    #   optimizer: SGD or Adam, the optimizer object that you are using to train your model
    #   step_size: the number of epochs after which you want to decrease your learning rate
    #   gamma: the factor by which you want to decrease the learning rate (0.25)

    train(mymodel, peter_loss, myoptimizer, myscheduler, train_dataloader, val_dataloader, \
          args.epochs, args.endure_times, args.log_interval, mydevice, mymodel_path, args.use_feature, args.rating_reg)
