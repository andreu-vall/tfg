import argparse
import os
import torch
import torch.nn as nn
import json
import sys
from torch.utils.data import DataLoader, Subset

from utils.peter import now_time, content, loss, root_mean_square_error, mean_absolute_error
from data import MyDataset, MySplitDataset, setup_logger


def test(dataloader: DataLoader, model, loss_fn, device):
    
    model.eval() # Turn on evaluation mode which disables dropout

    # el problema ha sigut afegir-li les prediccions aquí en el test
    predictions = [[], [], [], []] # concat all predictions: log_word_prob, log_context_dis, rating, attns
    total_losses = torch.zeros(4)
    
    with torch.no_grad():

        for batch in dataloader: # Millor sense barra de progrés, pq només tarda uns 10 segons i ocupa espai de la pantalla per res

            assert len(batch) == 4

            batch = [elem.to(device) for elem in batch] # moure a cuda

            user, item, rating, text = batch

            text = text.t() # en el test es transposa el text

            batch[3] = text # pq tmb cal tranposat usat com a batch
            
            # inicialment tenia mida [128, 10], que és [batch_size, fixed_tokens=8(argument del train) +2]
            # print('text will be transposed, from', real[3].shape)
            
            batch_size = user.size(0)

            # ja ho entenc, ara uso el real a un lloc i no s'ha tranposat allí el text!

            # print('transposed text is', text.shape)

            # pq es borra en el test??? En el test només s'intenta predir l'últim token o què?
            # És com dir que en el test només prediràs exactament 1 token, que és l'últim dels texts que els passis

            # print('removing fsr the last token of the text')
            
            text = text[:-1]  # (src_len + tgt_len - 2, batch_size)

            # print('text shape is', text.shape)
            # print('text is', text)

            predicted = model(user, item, text)

            # En aquest cas si es fa servir per algo la predicció de l'últim token és una mica estúpid pq només està intenant
            # predir l'última paraula del text. Crec que tindria més sentit intentar-los predir tots alhora però per separat,
            # i després quan vulguis generar text amb sentit sí que té sentit generar-los de forma seqüencial
            
            losses = loss_fn(predicted, batch) # [c_loss, r_loss, t_loss, loss]
            
            # LOL aquesta 1 línia m'ha portat un munt de problemes
            for i in range(4): # era un extend enlloc de append
                predictions[i].extend(predicted[i].cpu()) # important passar-lo a cpu,
                # si no no ho allibera de la GPU en cada batch i per tant acaba petant per memòria

            total_losses += torch.tensor(losses) * batch_size
    
    print('ha acabat el test')
    print('les lengths de cada cosa són:', [len(elem) for elem in predictions])
    # ara al final em surt [1716, 19850, 19850, 312] (len 2 coses del mig estan bé però les altres no?)

    return (total_losses / len(dataloader.dataset)).tolist(), predictions


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

    with open(f'out/{cmd_args.train_id}/train.json', 'r') as f:
        train_args = json.load(f)

    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

    return args


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join('out', args.train_id)

    mylogs = os.path.join(path, 'logs')
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log', True) # si no ara mateix ni imprimeix res
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')
    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    if torch.cuda.is_available():
        if args.cpu:
            peter_logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run without --cpu')
    mydevice = torch.device('cuda' if not args.cpu else 'cpu')

    model_path = os.path.join(path, 'model.pt')
    with open(model_path, 'rb') as f:
        mymodel = torch.load(f).to(mydevice)

    mydata = MyDataset(args.data_path, args.tokenizer, args.context_window)
    mysplitdata = MySplitDataset(args.data_path, len(mydata), args.split_id, load_split=True)

    test_data = Subset(mydata, mysplitdata.test)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    tgt_len = args.context_window + 1  # added <bos> or <eos>

    mytext_criterion = nn.NLLLoss(ignore_index=mydata.token_dict.pad)
    myrating_criterion = nn.MSELoss()

    peter_loss = lambda predicted, real: loss(predicted, real, args.context_reg, args.text_reg, args.rating_reg,
                                              mytext_criterion, myrating_criterion, len(mydata.token_dict), tgt_len)

    # Per caluclar el MAE i el RMSE necessito els valors predits a part de les losses
    test_losses, predictions = test(test_dataloader, mymodel, peter_loss, mydevice)
    c_loss, t_loss, r_loss, loss = test_losses
    peter_logger.info('=' * 89)
    peter_logger.info(f"{now_time()}{content(c_loss, t_loss, r_loss)} on test") # will delete it?

    
    log_word_prob, log_context_dis, predicted_rating, attns = predictions

    real_ratings = []
    for batch in test_dataloader:
        real_ratings.extend(batch[2].tolist())

    real_predicted_rating = [(r, p) for (r, p) in zip(real_ratings, predicted_rating)]


    RMSE = root_mean_square_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)
    peter_logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(real_predicted_rating, mydata.max_rating, mydata.min_rating)
    peter_logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))



    # ara mateix només ho escriu en el peter.log
    # possiblement hauria de escriure-ho per pantalla tmb, i veure q vull exactament posar en el text
    # ara en el test tampoc no es veu la barra del progrés pq la he tret en general de quan es feia test,
    # ja que quan es fa el test amb la validation no aportava realemnt gaire

    # No entenc que he canviat pq ara peti
    # He de tornar a entrenar?
