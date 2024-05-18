import argparse
import os
import torch
import torch.nn as nn
import json
import sys
import tqdm
from torch.utils.data import DataLoader, Subset

from utils.peter import now_time, content, root_mean_square_error, mean_absolute_error
from data import MyDataset, MySplitDataset, setup_logger
from losses import peter_loss
from generate import get_topk_tokens


# Crec que el següent que puc fer és entendre del tot el test
# Però aviat hauria de sopar i anar a dormir, i demà serà un altre dia



# Hauria de veure ara ja pq cullons borra un token del text en el test, quin sentit té? Crec que cap

# yikes, en el case de save_results necessito el data per fer desencriptacions
def test(dataloader:DataLoader, model, loss_fn, device, save_results=False, data:MyDataset=None):

    # quan estic en mode test real, hi ha dues coses diferents:
    # - vull la progress bar
    # - vull guardar els resultats per posar-los en un json
    
    model.eval() # Turn on evaluation mode which disables dropout

    if save_results:
        assert data is not None
        results = [] # crec que voldré guardar més coses

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

            predicted = model(user, item, input_text)

            log_word_prob, log_context_dis, predicted_rating, _ = predicted

            loss_input = [log_word_prob, log_context_dis, predicted_rating]
            loss_output = [target_text, rating]

            losses = loss_fn(loss_input, loss_output)

            total_losses += torch.tensor(losses) * batch_size

            if not save_results:
                continue
            
            

            # 2. borren un token del text (probably eos o pad)...
            # 3. predicted = model(user, item, text editat així)



            # Tècnicament es podria construir un text sencer, però cada single token seria generat a partir de
            # el text real que vols predir + només s'afegiria un token a la predicció. I la resta de tokens es
            # seguirien fent usant fins al step N - 1 el text real, pq el predit els generen tots en paral·lel
            # i per tant encara no els has generat

            # Ja no borro 1 element del text MAI i no peta sembla
            # text = text.t() # en el test es transposa el text
            # batch[3] = text # pq tmb cal tranposat usat com a batch
            
            # inicialment tenia mida [128, 10], que és [batch_size, fixed_tokens=8(argument del train) +2]
            # print('text will be transposed, from', real[3].shape)

            # pq es borra en el test??? En el test només s'intenta predir l'últim token o què?
            # És com dir que en el test només prediràs exactament 1 token, que és l'últim dels texts que els passis

            # print('removing fsr the last token of the text')
            
            # AVIAM PROVO DE COMENTAR AIXÒ AVIAM QUÈ PASSA. DONCS PASSA QUE L'ESPERAVA EL MODEL DE MIDA 1 MENYS
            # PER TANT POTSER HE DE CANVIAR DONCS ELS MODEL I TORNAR A ENTRENAR DES DE 0?
            #1234
            # # és per fer el shift right del output
            # text = text[:-1]  # (src_len + tgt_len - 2, batch_size)

            # print('text shape is', text.shape)
            # print('text is', text)

            # en el step de test simplement es crida un model(user, item, text normal)
            # predicted = model(user, item, text)

            # log_word_prob, log_context_dis, predicted_rating, attns = predicted # això és el que retorna el model

            # print('log_word_prob shape is', log_word_prob.shape) # [7, 128, 9994], abbans [6, 128, 9994] i [11, 128, 11_042]
            # # 7 -> aprox mida del context_window (5) + 2 (bos, eos)
            # # 128 -> batch_size
            # # 9994 -> ntokens
            # assert False

            # pel train_id_9 sembla que no hi havia leakage??
            # Puc provar de desfer el canvi del text[:-1] i veure què passa

            # No estic segur sí el leakage que em passa ara és per haver canviar la mida de la attentiont mask per la cara

            # log_word_prob -> 

            # if seq_prediction (és el cas casi sempre) -> func.log_softmax(self.hidden2token(hidden[self.src_len:]), dim=-1) (TOT)
            # else                                      -> func.log_softmax(self.hidden2token(hidden[-1]), dim=-1) (1 cosa...)

            # what is the point of the 2nd one? crec que el primer que sempre usa tot és millor

            # log_context_dis -> distribució de probabilitats predida dels tokens pel text en base només a user, item
            # rating -> predicció de nota, basant NOMÉS en el user, item (NO s'usa per res el text)
            # attns -> atencions que utilitza en transformer_encoder

            # En aquest cas si es fa servir per algo la predicció de l'últim token és una mica estúpid pq només està intenant
            # predir l'última paraula del text. Crec que tindria més sentit intentar-los predir tots alhora però per separat,
            # i després quan vulguis generar text amb sentit sí que té sentit generar-los de forma seqüencial
            
            # es calcula la loss_fn EXACTAMENT igual que en el train
            # ara doncs tmb ho he de canviar aquí obiously
            
            
            # ara encara em dona problemes de RAM així!!!

            # # LOL aquesta 1 línia m'ha portat un munt de problemes
            # for i in range(4): # era un extend enlloc de append
            #     predictions[i].extend(predicted[i].cpu()) # important passar-lo a cpu,
                # si no no ho allibera de la GPU en cada batch i per tant acaba petant per memòria

            # log_word_prob, log_context_dis, rating, attns

            # si vull posar aquestes coses, les hauria de descodificar...


            # Fer ara tantes coses en el test fa que tardi molt més en executar-lo
            # No hi ha cap problema though, en el train igualment s'hauran de fer moltíssimes èpoques,
            # i només em cal 1 test guardant resultats



            decoded_user = [data.user_decode(u) for u in user]
            decoded_item = [data.item_decode(i) for i in item]

            # tinc log_word_prob, log_context_dis, predicted_rating, attns

            max_length = data.context_window
            predicted_context = get_topk_tokens(log_context_dis, topk=max_length)

            # log_context_dis -> predicted_context

            # Pel context no hi haurà ni <bos> ni <eos>/<cut>
            decoded_predicted_context = [data.text_decode(list(c), mode='literal') for c in predicted_context]
            untokenized_predicted_context = [data.untokenize(c) for c in decoded_predicted_context]

            # i need predicted_text

            # en el generate: predicted_rating, predicted_context, predicted_text = predicted,
            # predicted = generate_batch(...)
            # en canvi aquí: log_word_prob, log_context_dis, predicted_rating, attns = predicted

            # el predicted_text venia de allí: predicted_rating, predicted_context, predicted_text = predicted 
            log_word_prob, log_context_dis, predicted_rating, attns = predicted # això és el que retorna el model
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

            # log_word_prob.view(-1, ntokens) directament i el hidden2token?

            # LOL al final el que s'optimitza del text només és un classificador de les probabilitats de les paraules

            # He de descodificar amb greedy individualment cada token i juntar-ho, per tenir una representació visual
            # de què està intentant predir el meu model

            # necessito predicted_text, que són les ids dels tokens més probables en cada step generats en paral·lel
            # tinc log_word_prob

            # print('log_word_prob shape is', log_word_prob.shape) # [6, 128, 9994] # abans [11, 128, 11_042]
            # assert False
            # PETER ho feia així:
            word_prob = log_word_prob.exp()
            # print('word_prob shape is', word_prob.shape) # [6, 128, 9994]
            word_idx = torch.argmax(word_prob, dim=2) # he canviat de la dim=1 a la dim=2
            # print('word_idx shape is', word_idx.shape) # [6, 128] Ara sí! Això ja és lo bo crec

            # where TF does the leakage come from? If it should not be fed the next token?

            # Claus: al cridar-lo amb tot el hidden, ha calgut fer el argmaz en la dim=2 no la dim 1,
            # i cal transposar el text pq estava treballant a nivell de token, però després ho vull a nivell de review

            # falta posar-li el <bos>, que ja s'ha suposat que el tindria i s'ha predit les següents posicions

            word_idx_cpu = word_idx.cpu() # ja ho puc passar a CPU pq ara l'únic que faré serà descodificar-ho
            # i escriure-ho en un json, tenir en CPU ara ja és millor

            bos_value = data.token_dict.bos
            # segons copilot (efectivament si el posava directament en el cat no funcava)
            # potser haur
            bos_tensor = torch.full((1, batch_size), bos_value) # què faig ho moc o no?
            # si no estan en el mateix no ho puc concatenar

            # print('bos tensor shape is', bos_tensor.shape) # [6, 1]. Però hauria de ser [1, 128]
            # print('word_idx shape is', word_idx.shape)     # [6, 128] LOL copilot l'ha encertat aquí

            # a partir d'aquí no necessiot ja més cuda per res


            predicted_text = torch.cat([bos_tensor, word_idx_cpu], 0).T


            # predicted_text = word_idx.T # will it work? Nope. Aviam he dit que aquest tenia length [6, 128]. L'he de transposar?

            # unsqueezed = word_idx.unsqueeze(0)
            # print('unsqueezed shape is', unsqueezed.shape) # [1, 6, 128]? Ara sí! Why is the unsqueeze? Crec que jo no el necessiot

            # log_word_prob shape is torch.Size([128, 20004])
            # word_prob shape is torch.Size([128, 20004])
            # word_idx shape is torch.Size([128])
            # unsqueezed shape is torch.Size([1, 128])


            # text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)

            # pel test de coses en paral·lel sembla que torna a caldre el raw


            # amb 1 època no necessàriament haurà après a posar el <eos> o <cut> al final de la frase
            decoded_predicted_text = [data.text_decode(list(t), mode='sceptic') for t in predicted_text] # , raw=True
            untokenized_predicted_text = [data.untokenize(t) for t in decoded_predicted_text]

            #1234
            # the model inputs should not include the last token
            # the target should be shifted one step to the right

            # example:
            # the model will receive ["I", "am", "trying", "to", "predict"] as input
            # and will try to predict ["am", "trying", "to", "predict", "this"] (text shifted right)
            # this is the meaning of the output shifted right

            # print('batch_size is', batch_size) # 128
            # print('len of untokenized_predicted_text is', len(untokenized_predicted_text)) # 6
            # print('len of elem 0 is', len(untokenized_predicted_text[0])) # 548

            # LOL sembla que el text s'ha passat al model sense <eos>!!!!
            # de moment ho passo, no estic segur si hauria de posar el <eos> si en realitat encara no acaba
            # why is raw necessary aquí???? si en teoria s'estava fixant tot no? És per culpa dels de PETER,
            # que per la cara eliminen l'últim token!!!!!
            # això canvia quan canvio el text
            #1234 posat el raw=True pq calli ara només
            # ara ja no cal de transposar pq ho he tranposat en una altra variable?
            # què decideixo jo, de posar sempre el <eos> si tallo una cosa per la cara o no? És una mica important
            # prendre una decisió i ser consistent a tot arreu amb el que trïi
            decoded_text = [data.text_decode(list(t), mode='correct') for t in text]
            untokenized_text = [data.untokenize(t) for t in decoded_text]

            # print('len of decoded_text is', len(decoded_text)) # 6 Clar pq s'ha transposat!
            # print('len of untokenized_text is', len(untokenized_text)) # 6??

            # real_text: text.T -> decoded_text -> untokenized_text
            # predicted_text: predicted_text -> decoded_predicted_text -> untokenized_predicted_text (seems correct?)


            # sembla que cridar el .item() ja fa que els valors no estiguin a la GPU.
            # El problema és quan estava guardant coses que no sabia exactament que eren
            # i per tant estava guardant moltes més coses de les que creia i en no fer una
            # conversió de format es quedaven a la GPU i m'explotava la memòria

            if save_results:
                for i in range(batch_size):
                    results.append({
                        'user': decoded_user[i],
                        'item': decoded_item[i],
                        'predicted_rating': predicted_rating[i].item(),
                        'real_rating': rating[i].item(),
                        'predicted_context': untokenized_predicted_context[i],
                        'predicted_text': untokenized_predicted_text[i], # weird que sigui 1 més que window_size but whatever
                        'real_text': untokenized_text[i]
                        # podria posar el predicted_context aquí
                        # i també el real_text
                        # i de fet també el predicted_text, tot i que possiblement no tindria gaire sentit,
                        # pq es genera en paral·lel i per tant en cada step suposo que lo que hi ha generat
                        # fins ara era lo bo, no lo que ha generat realment el propi model
                        # 'text': text[:, i].tolist(), # probably sí que cal
                        # 'attns': attns[i].tolist()
                    })


                # rating_predictions.extend(predicted[2].cpu().tolist())

            # el 2 són els ratings normals predits
            # puc agafar això de moment només

            # print('prectitions[1] len is', len(predictions[1])) # 128 (batch_size)
            # print('prectitions[2] len is', len(predictions[2])) # 128
            # print('prectitions[3] len is', len(predictions[3])) # 2

            # print('prediction[0][0] shape és', predictions[0][0].shape) # [128, 11_042] (probabilitats de tots els tokens per review)
            # print('prediction[1][0] shape és', predictions[1][0].shape) # [11_042] (probabilitats dels tokens once?)
            # print('prediction[2][0] shape és', predictions[2][0].shape) # []???
            # print('prediction[3][0] shape és', predictions[3][0].shape) # [128, 13, 13]

            # # There's 11_042 tokens
            # # vaig usar context_window=10, + 3 extra surten les attentions

            # assert(False)

            # assert(False)
    
    # ara al final em sortia [1716, 19850, 19850, 312]

    losses = (total_losses / len(dataloader.dataset)).tolist()

    if not save_results:
        return losses # seria millor tornar un results buit? no crec
    
    return losses, results


# Combinar els arguments amb els de train, pq hi ha moltes coses que es necessitaven de train
def parse_arguments():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('train_id', type=str)
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA') # hauria de comprovar si es pot canviar
    # només en inferència, estic bastant segut que sí

    cmd_args = cmd_parser.parse_args()

    path = os.path.join('out', cmd_args.train_id)
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')

    with open(f'out/{cmd_args.train_id}/train_parameters.json', 'r') as f:
        train_args = json.load(f)

    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join('out', args.train_id)

    mylogs = os.path.join(path, 'logs')
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')
    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

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

    print('tgt_len', tgt_len)
    print('ntokens', ntokens)

    text_criterion = nn.NLLLoss(ignore_index=mydata.token_dict.pad)
    rating_criterion = nn.MSELoss()

    myloss_fn = lambda loss_input, loss_output: peter_loss(
        loss_input, loss_output, text_criterion, rating_criterion,
        args.context_reg, args.text_reg, args.rating_reg, ntokens, tgt_len
    )
    # Aquí en el test sí que passo el data ara mateix per fer descodificacions. en canvi en el train no el passava
    # Per caluclar el MAE i el RMSE necessito els valors predits a part de les losses
    test_losses, test_results = test(mytest_dataloader, mymodel, myloss_fn, mydevice, save_results=True, data=mydata)
    c_loss, t_loss, r_loss, loss = test_losses
    print(f"{now_time()}{content(c_loss, t_loss, r_loss)} on test") # will delete it?
    # Això encara són mètriques weird del PETER que he de canviar per les losses reals del model,
    # ja que no representen realment la loss que avalua el model

    # De fet falten calcular les mètriques abans de posar en el json
    with open(f"out/{args.train_id}/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=4)
    
    rating_predictions = [result['predicted_rating'] for result in test_results]
    real_ratings = [result['real_rating'] for result in test_results]

    real_predicted_rating = [(r, p) for (r, p) in zip(real_ratings, rating_predictions)]

    RMSE = root_mean_square_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(MAE))

    # ara mateix només ho escriu en el peter.log
    # possiblement hauria de escriure-ho per pantalla tmb, i veure q vull exactament posar en el text
    # ara en el test tampoc no es veu la barra del progrés pq la he tret en general de quan es feia test,
    # ja que quan es fa el test amb la validation no aportava realemnt gaire

    # No entenc que he canviat pq ara peti
    # He de tornar a entrenar?
