import sys
import os
import datetime
import logging

import torch
import argparse
import torch.nn as nn

from models.peter import PETER
from utils.peter import DataLoader, Batchify, now_time
from utils.andreu import move_content_to_device, peter_print_long, peter_content, peter_loss_good
from test import test

start = datetime.datetime.now()


# Potser hauria de separar entre hiperparàmetres del model i hiperparàmetres de l'entrenament

parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')

# Paràmetres obligatoris
parser.add_argument('id', type=str, # si poso el arguments sense el -- (id enlloc de --id) ja es posa auto required=True
                    help='model id')
parser.add_argument('data_path', type=str,
                    help='path for loading the pickle data')
parser.add_argument('index_dir', type=str,
                    help='load indexes')

# També hi ha l'atribut útil choices per restringir els valors
# Seria batant ideal organitzar millor els paràmetres

# Paràmetres del model
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')


# El clip és de l'entrenament o del model?

# Paràmetres d'entrenament
parser.add_argument('--lr', type=float, default=1.0,    # No té sentit matemàticament començar amb un learning rate major que 1,
                    help='initial learning rate')       # perquè podria explotar la suma infinita, tot i que realment no és inf.
parser.add_argument('--clip', type=float, default=1.0,  # Tot i així estaria bé provar-ho algun cop
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, # 100
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cpu', action='store_true',   # He canviat el default behavious a GPU, ja que sempre
                    help='don\'t use CUDA')         # ho faré així i crec que és més estàndard en torch
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')

# Whether I want a directory to load the model instead of building a new one
# Also whether I want to resume the training at a particular step (in case of crash). But I'll do that later
parser.add_argument('--source_checkpoint', type=str, default=None,
                    help='directory to load the model')

parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')

# Serveixen per definir l'objectiu a minimitzar
parser.add_argument('--rating_reg', type=float, default=0.1,
                    help='regularization on recommendation task')
parser.add_argument('--context_reg', type=float, default=1.0,
                    help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')


parser.add_argument('--peter_mask', action='store_true',
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--use_feature', action='store_true',
                    help='False: no feature; True: use the feature')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
args = parser.parse_args()


path = os.path.join('out', args.id)
if os.path.exists(path):
    raise ValueError('This id already exists!')
os.makedirs(path)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", #f"{now_time()}%(message)s", # [%(threadName)-12.12s] [%(levelname)-5.5s] 
    handlers=[
        logging.FileHandler(f"{args.id}/train.log"),
        logging.StreamHandler()
    ]
)

# Crec que imprimir també això és força útil i important
# el -u i la redirecció al fitxer aquí no es mostra
logging.info(f"{os.path.basename(sys.executable)} {' '.join(sys.argv)}")
logging.info(start.strftime("%Y-%m-%d %H:%M:%S"))


# Si en un altre lloc ja faig el logs dels arguments i aquest el vull més aviat per carregar-los en el test,
# potser no tots els arguments són necessaris pel test (n'hi ha que són específics del train)
import json
with open(f'out/{args.id}/train-args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)



logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    logging.info('{:40} {}'.format(arg, getattr(args, arg)))
logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cpu:
        logging.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
mydevice = torch.device('cuda' if not args.cpu else 'cpu')

model_path = os.path.join(path, 'model.pt')
prediction_path = os.path.join(path, args.outf)

###############################################################################
# Load data
###############################################################################

logging.info(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
word2idx = corpus.word_dict.word2idx
pad_idx = word2idx['<pad>']
train_dataloader = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True) # potser només caldria carregar si entreno
val_dataloader = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_dataloader = Batchify(corpus.test, word2idx, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################

if args.source_checkpoint is None:
    logging.info(now_time() + 'Building model')
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
    logging.info(now_time() + 'Loading model')
    with open(args.source_checkpoint, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()


###############################################################################
# Training code
###############################################################################



# variables de args per la cara (well aquí és global no mètode): context_reg, text_reg, rating_red
peter_loss = lambda pred, content: peter_loss_good(pred, content, args.context_reg, args.text_reg, args.rating_reg,
                                                   text_criterion, rating_criterion, ntokens, tgt_len)




# variables de args per la cara: use_feature, clip
def train_epoch(dataloader, model, loss_fn, optimizer, device, log_interval):
    
    model.train()

    total_losses, interval_losses = torch.zeros(4), torch.zeros(4)
    interval_sample = 0

    num_batches = len(dataloader)
    
    for batch, content in enumerate(dataloader):

        content = move_content_to_device(content, device)

        user, item, rating, seq, feature = content # (batch_size, seq_len), batch += 1 (comentari ràndom PETER)
        batch_size = user.size(0)
 
        if args.use_feature:
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

        pred = model(user, item, text)

        batch_losses = loss_fn(pred, content)
        c_loss, r_loss, t_loss, loss = batch_losses

        loss.backward()

        # Això ho han posat els de PETER, comprovar si realment és necessari
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step() # Update els paràmetres del model segons la loss i l'optimitzador

        # Canviat al final com a l'exemple bo, aviam si així s'arregla. YAY s'ha solucionat
        optimizer.zero_grad()

        interval_losses += torch.tensor(batch_losses) * batch_size
        total_losses += torch.tensor(batch_losses) * batch_size
        interval_sample += batch_size

        if (batch + 1) % log_interval == 0 or batch == num_batches - 1: # ara començo batch a 0 enlloc de 1
            interval_average_losses = interval_losses / interval_sample

            cur_c_loss, cur_t_loss, cur_r_loss, cur_loss = interval_average_losses

            peter_print_train(cur_c_loss, cur_t_loss, cur_r_loss, batch, num_batches)
            
            interval_losses = torch.zeros(4)
            interval_sample = 0
        
    return (total_losses / dataloader.total_elements).tolist()




def peter_print_train(context_loss, text_loss, rating_loss, batch, num_batches):
    logging.info(f"{now_time()}{peter_content(context_loss, text_loss, rating_loss)} | \
          {(batch+1):5d}/{num_batches:5d} batches")




# Cal canviar l'entrenament lleugerament per no fer-lo tant basat en el nº d'èpoques,
# que per si sol no m'importa per res, i que l'optimizer que s'encarrega de modificar
# el learn_rate tingui més en compte les mètriques de validació.
# Com està fet ara que doni una misèria pitjor 1 cop en la validació i ja es redueix
# el learn_rate a la meitat, i aleshores tarda possiblement el doble a entrenar.
# De fet havia vist alguns cops on havia empitjorat uns quants cops al principi
# i després ha seguit amb un munt d'èpoques més i per tant ha tardat molt més.

# Si millora molt seria interessant tornar a agumentar el learn_rate?



#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

# scheduler adjusts the learning rate according to a schedule that you define
#   optimizer: SGD or Adam, the optimizer object that you are using to train your model
#   step_size: the number of epochs after which you want to decrease your learning rate
#   gamma: the factor by which you want to decrease the learning rate (0.25)

# Segons el Colab és qüestionable at best aquesta manera que estan fent servir ells el shceduler
# Crec que el que em va dir era fals i em va fer perdre el temps amb el ReduceLROnPlateau

# El learning_rate ho diu les mètriques en el dataset de validació, no jo sense ni idea de per què


# PROBLEMA DELS PRINTS: PROBABLY NO PODRÉ USAR TQDM I PRINTS AL MATEIX TEMPS, PQ EL TQDM SOBREESCRIUR LA ÚLTIMA
# DEL OUTPUT MOLTES VEGADES, EN CANVI ELS PRINTS ES VAN AFEGINT LÍNIES I LÍNIES. POTSER EL APPROACH DE LOGGING
# ÉS SUPERIOR DONCS. DE MOMENT NO POSO DONCS LA PROGRESS BAR DE TQDM

# He vist que en general esperar vàries èpoques un cop ja has vist que millora en el train i empitjora en el validation
# amb el mateix learning_rate, només fa que facis un overfitting de la òstia al training dataset. Per tant el que feien
# els de PETER de reduir després de cada òstia probably sigui lo millor

myoptimizer = torch.optim.SGD(mymodel.parameters(), lr=args.lr) #, momentum=0.9) # de moment no li poso el momentum
myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)

# Crec que és interessant fer la validation del model abans de fer cap pas d'entrenament. Per saber
# d'on venies un cop s'instancia el model. Podries venir de coses aleatòries i donar molt alt, o fer
# transfer learning entre datasets i aleshores tindria molt sentit veure inicialment com de bo és

# És una mica lleig tant identat tot. Potser al final acabarà seguint el mateix camí que l'Alejandro en moltes coses.
# Ja entenc per què no es va guardar els models del SEQUER, perquè en general tarden bastant poc en entrenar i així
# va poder fer moltes proves. Només va guardar els resultats de les mètriques que possiblement necessitava pel paper

# Varibales de args per la cara: rating_reg
def train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, endure_times, log_interval):

    # Hauria de guardar també el model amb 0 èpoques? It's a bit silly but I will

    # Not sure ni el now_time l'hauria de seguir cridant manualemnt a tot arreu

    logging.info(now_time() + 'Epoch 0 validation') # Crec que la validation de l'època 0 sempre és interessant
    val_losses = test(val_dataloader, mymodel, peter_loss, mydevice, args.use_feature)
    val_loss = val_losses[3] # real_loss for the Gradient Descent
    peter_print_long(val_losses, args.rating_reg)

    with open(model_path, 'wb') as f:
        torch.save(model, f)

    if epochs == 0:
        logging.info(now_time() + 'No epochs to train')
        return []


    # De moment uso el now_time() per evitar parsejar diferent, però és kinda ugly
    logging.info(now_time() + 'Start training')

    best_val_loss = val_loss
    endure_count = 0

    for epoch in range(1, epochs + 1):

        logging.info(f"{now_time()}epoch {epoch} train loop")
        train_losses = train_epoch(train_dataloader, model, peter_loss, optimizer, mydevice, log_interval)
        peter_print_long(train_losses, args.rating_reg, 'training')

        logging.info(f"{now_time()}epoch {epoch} validation")
        val_losses = test(val_dataloader, model, peter_loss, mydevice, args.use_feature)
        peter_print_long(val_losses, args.rating_reg)
        val_loss = val_losses[3] # real_loss for the Gradient Descent

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            logging.info(f"{now_time()}Endured {endure_count}/{endure_times} time(s)")
            if endure_count == endure_times:
                logging.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            
            scheduler.step()

            # Potser cal alliberar manualment memòria de la GPU?

            # Carrego el millor model, perquè si no és el model que ja ha fet overfit. Crec que així és molt millor
            # pel meu cas, tot i que sembla que no és massa estàndard fer-ho així en general. Crec que precisament
            # no fer-ho així contribueix en gran part en la diferència entre diferentes seeds. És a dir, estan perdent
            # el temps per una tonteria molt gran i simple i m'intenten justificar a mi que no es torna enrere,
            # quan tornar enrere és molt fàcil i crec que donarà molts millors resultats
            with open(model_path, 'rb') as f:
                model = torch.load(f).to(mydevice)

            logging.info(f"{now_time()}Learning rate set to {scheduler.get_last_lr()[0]}") # Torna mútliples grups de params

        # Si vulgués aturar el bucle d'entrenament a mitges, i continuar exactament a partir d'on estava,
        # hauria de guardar: model, optimizer, scheduler, epoch, endure_count, best_val_loss



# Encara em queden algunes variables global: model_path per exemple

# Això serà necessari sobretot pel test per poder posar en codi de testing en el test.py i per exemple fer-lo
# córrer des del train.py sense executar TOTES les coses del test, només agafar el mètode aquell que calia
# i que té més sentit que estigui en el test. Tot i així en el train costarà una mica posar tot lo necessari dintre
# només del main. Realment des de test o altres llocs voldré improtar mètodes del train.py?

# RN és una mica inútil pq encara hi ha moltes coses que encara estan fora de les funcions, but I will fix it later
if __name__ == "__main__":

    train(mymodel, myoptimizer, myscheduler, train_dataloader, val_dataloader, args.epochs, args.endure_times, args.log_interval)
