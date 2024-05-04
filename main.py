import sys
import os
import datetime

import math
import torch
import argparse
import torch.nn as nn
from module import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


# Crec que imprimir també això és força útil i important
# el -u i la redirecció al fitxer aquí no es mostra
print(f"{os.path.basename(sys.executable)} {' '.join(sys.argv)}")

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))


# Potser hauria de separar entre hiperparàmetres del model i hiperparàmetres de l'entrenament

parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')

# Paràmetres obligatoris
parser.add_argument('--id', type=str, required=True,
                    help='model id')
parser.add_argument('--data_path', type=str, required=True,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, required=True,
                    help='load indexes')

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
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')

# Whether I want a directory to load the model instead of building a new one
# Also whether I want to resume the training at a particular step (in case of crash). But I'll do that later
parser.add_argument('--source_checkpoint', type=str, default=None,
                    help='directory to load the model')
parser.add_argument('--test_only', action='store_true',
                    help='Skip the training and only run the test')

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

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
mydevice = torch.device('cuda' if args.cuda else 'cpu') # I si es vol fer el train i el test en diferents devices?

path = os.path.join('out', args.id)
if os.path.exists(path):
    raise ValueError('The model id already exists') # Això serà un problema
os.makedirs(path)

model_path = os.path.join(path, 'model.pt')
prediction_path = os.path.join(path, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
word2idx = corpus.word_dict.word2idx
pad_idx = word2idx['<pad>']
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
train_dataloader = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True) # potser només caldria carregar si entreno
val_dataloader = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_dataloader = Batchify(corpus.test, word2idx, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################

if args.source_checkpoint is None:
    print(now_time() + 'Building model')
    if args.use_feature:
        src_len = 2 + train_dataloader.feature.size(1)  # [u, i, f]
    else:
        src_len = 2  # [u, i]
    tgt_len = args.words + 1  # added <bos> or <eos>
    ntokens = len(corpus.word_dict)
    nuser = len(corpus.user_dict)
    nitem = len(corpus.item_dict)
    model = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens,
                args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(mydevice)

else:
    print(now_time() + 'Loading model')
    with open(args.source_checkpoint, 'rb') as f:
        model = torch.load(f).to(mydevice)

text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()


###############################################################################
# Training code
###############################################################################


# Aquí és on descodifica el context
def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


# context_reg, text_reg, rating_reg: la importància relativa de les 3 tasques a optimitzar
def peter_loss_good(pred, content, context_reg, text_reg, rating_reg):
    user, item, rating, seq, feature = content

    log_word_prob, log_context_dis, rating_p, _ = pred  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

    c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,))) # This is a bit ugly
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
    r_loss = rating_criterion(rating_p, rating)

    loss = c_loss * context_reg + t_loss * text_reg + r_loss * rating_reg # ordre més normal ara

    return c_loss, t_loss, r_loss, loss


# variables de args per la cara (well aquí és global no mètode): context_reg, text_reg, rating_red
peter_loss = lambda pred, content: peter_loss_good(pred, content, args.context_reg, args.text_reg, args.rating_reg)


# Moure a GPU i transposar seq i feature. Segons Copilot és més estàndard moure les coses a GPU
# si cal en el train enlloc del dataloader, perquè així compta en el temps del train
def move_content_to_device(content, device):
    user, item, rating, seq, feature = content
    # batch_size = user.size(0)

    user = user.to(device)  # (batch_size,)
    item = item.to(device)
    rating = rating.to(device)
    seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
    feature = feature.t().to(device)  # (1, batch_size)
    return user, item, rating, seq, feature


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

# variables de args per la cara: use_feature
def test(dataloader, model, loss_fn, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = torch.zeros(4)
    
    with torch.no_grad():
        for content in dataloader:

            content = move_content_to_device(content, device)

            user, item, rating, seq, feature = content
            batch_size = user.size(0)

            if args.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

            pred = model(user, item, text)

            losses = loss_fn(pred, content) # [c_loss, r_loss, t_loss, loss] # al revés?

            total_losses += torch.tensor(losses) * batch_size

    return (total_losses / dataloader.total_elements).tolist() # Crec q amb el tolist s'hauria de solucionar les referències


# Encara no he fet pràcticament res en aquesta funció
# variables de args per la cara: use_feature, words
def generate(dataloader, model, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        for content in dataloader:

            # this is a whole batch
            user, item, rating, seq, feature = move_content_to_device(content, device)

            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size) # Maybe no funca?

            # Obvi que amb la feature donarà millors resultats, si l'estàs utilizant com
            # a cosa a partir de la qual començar a generar el text del output de l'explicació
            if args.use_feature:
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for idx in range(args.words):
                # produce a word at each step

                # Aquí és el pas clau on genera múltiples coses executant el model iterativament a partir
                # de les coses generades. Encara no ho acabo d'entendre però em sembla una manera més bona
                # de generar recomanacions que no tant el P5. M'agraden més les bones arquitecutres que no
                # simplement els bons prompts, perquè fer bons prompts és més un art i fer bones arquitectures
                # és un procés científic

                # COm es genera aquí un context tant llarg, si només es crida una vegada pel context?

                if idx == 0:
                    log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.extend(rating_p.tolist())
                    context = predict(log_context_dis, topk=args.words)  # (batch_size, words)
                    context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)

                # L'única cosa que es modifica és el text, cada cop se li va afegint una nova paraula
                # És el greedy decoding, que en cada step posa la paraula més probable.
                # Per això en el PETER probably genera kind of coses genèriques, perquè s'utilitza
                # una estratègia de decoding de text molt senzilla. Crec que seria fàcil i interessant
                # estudiar altres, totes, les estratègies bàsiques de decoding

                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(dataloader.rating.tolist(), rating_predict)] # he canviat data per dataloader
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in dataloader.seq.tolist()] # he canviat data per dataloader
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    print(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    print(now_time() + 'FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in dataloader.feature.squeeze(1).tolist()]  # ids to words, he canviat data per dataloader
    FMR = feature_matching_ratio(feature_batch, feature_test)
    print(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    return text_out


# De moment segueixo amb els prints de PETER per comparar fàcilment. Però més endavant potser seria millor
# canviar a logging com l'Alejandro. Així per exemple podria fer servir el tqdm que és més pro i útil

# Si els números són molt grans no dona tot de la mateixa mida tampoc...
def peter_content(context_loss, text_loss, rating_loss):
    exp_context_loss = math.exp(context_loss)
    exp_text_loss = math.exp(text_loss)
    return f"context ppl {exp_context_loss:4.4f} | text ppl {exp_text_loss:4.4f} | rating loss {rating_loss:4.4f}"

def peter_print_train(context_loss, text_loss, rating_loss, batch, num_batches):
    print(f"{now_time()}{peter_content(context_loss, text_loss, rating_loss)} | \
          {(batch+1):5d}/{num_batches:5d} batches")

# Això és el que feien en el PETER. Per la validació s'ignora el loss de context (i maybe tmb el de rating)
# variables de args per la cara: rating_reg
def peter_print_long(val_losses, name='validation'):
    c_loss, t_loss, r_loss, real_loss = val_losses
    printed_loss = t_loss
    if args.rating_reg != 0: # what even is rating_reg?
        printed_loss += r_loss
    print(f"{now_time()}{peter_content(c_loss, t_loss, r_loss)} | valid loss {printed_loss:4.4f} on {name}. Real: {real_loss:4.4f}")





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

myoptimizer = torch.optim.SGD(model.parameters(), lr=args.lr) #, momentum=0.9) # de moment no li poso el momentum
myscheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, 1, gamma=0.25)

# Crec que és interessant fer la validation del model abans de fer cap pas d'entrenament. Per saber
# d'on venies un cop s'instancia el model. Podries venir de coses aleatòries i donar molt alt, o fer
# transfer learning entre datasets i aleshores tindria molt sentit veure inicialment com de bo és

# És una mica lleig tant identat tot. Potser al final acabarà seguint el mateix camí que l'Alejandro en moltes coses.
# Ja entenc per què no es va guardar els models del SEQUER, perquè en general tarden bastant poc en entrenar i així
# va poder fer moltes proves. Només va guardar els resultats de les mètriques que possiblement necessitava pel paper

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, endure_times, log_interval):

    # De moment uso el now_time() per evitar parsejar diferent, però és kinda ugly
    print(now_time() + 'Start training')

    best_val_loss = float('inf')
    endure_count = 0

    for epoch in range(1, epochs + 1):

        print(f"{now_time()}epoch {epoch} train loop")
        train_losses = train_epoch(train_dataloader, model, peter_loss, optimizer, mydevice, log_interval)
        peter_print_long(train_losses, 'training')

        print(f"{now_time()}epoch {epoch} validation")
        val_losses = test(val_dataloader, model, peter_loss, mydevice)
        peter_print_long(val_losses)
        val_loss = val_losses[3] # real_loss for the Gradient Descent

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(model_path, 'wb') as f:
                torch.save(model, f)
        else:
            endure_count += 1
            print(f"{now_time()}Endured {endure_count}/{endure_times} time(s)")
            if endure_count == endure_times:
                print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
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

            print(f"{now_time()}Learning rate set to {scheduler.get_last_lr()[0]}") # Torna mútliples grups de params

        # Si vulgués aturar el bucle d'entrenament a mitges, i continuar exactament a partir d'on estava,
        # hauria de guardar: model, optimizer, scheduler, epoch, endure_count, best_val_loss


print(now_time() + 'Epoch 0 validation')
val_losses = test(val_dataloader, model, peter_loss, mydevice)
peter_print_long(val_losses)

if not args.test_only:
    train(model, myoptimizer, myscheduler, train_dataloader, val_dataloader, args.epochs, args.endure_times, args.log_interval)


# THE TEST

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(mydevice)

# Run on test data.
test_losses = test(test_dataloader, model, peter_loss, mydevice)
print('=' * 89)
peter_print_long(test_losses, 'test')

print(now_time() + 'Generating text')
text_o = generate(test_dataloader)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
