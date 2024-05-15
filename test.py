import argparse
import os
import torch
import torch.nn as nn
import json
import sys
import tqdm

from torch.utils.data import DataLoader, Subset

from utils.peter import now_time, content, loss

from data import MyDataset, MySplitDataset, setup_logger, move_to_device


def test(dataloader: DataLoader, model, loss_fn, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = torch.zeros(4)
    
    with torch.no_grad():
        

        for real in dataloader: # Millor sense barra de progrés, pq només tarda uns 10 segons i ocupa espai de la pantalla per res

            # print('in the test')

            assert len(real) == 4

            # inicialment tenia mida [128, 10], que és [batch_size, fixed_tokens=8(argument del train) +2]
            # print('text will be transposed, from', real[3].shape)
            
            real = move_to_device(real, device, transpose_text=True) # en el test tmb ho transposaven
            user, item, rating, text = real
            batch_size = user.size(0)

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
            
            losses = loss_fn(predicted, real) # [c_loss, r_loss, t_loss, loss]

            total_losses += torch.tensor(losses) * batch_size

    return (total_losses / len(dataloader.dataset)).tolist()




# Encara he de netejar una mica això del main que ho tinc realment molt lleig
# primer cop que utilitzo voluntàriament pq el necessito el name main xD
if __name__ == "__main__":

    # This could be cleaned up a bit
    # El meu modus operandis és agafar un codi que funciona i executar-lo jo i anar-lo editant per entendre'l

    # Parse command line arguments. Primer hem de fet aquest pq és el que dona la id
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('train_id', type=str)
    cmd_parser.add_argument('--cpu', action='store_true', help='don\'t use CUDA') # hauria de comprovar si es pot canviar
    # només en inferència, estic bastant segut que sí

    cmd_args = cmd_parser.parse_args()

    path = os.path.join('out', cmd_args.train_id)
    if not os.path.exists(path):
        raise ValueError('This id doesn\'t exist!')
    

    mylogs = os.path.join(path, 'logs')
    peter_logger = setup_logger('peter_logger', f'{mylogs}/peter.log')
    history_logger = setup_logger('history_logger', f'{mylogs}/history.log')

    history_logger.info(f"{now_time()}python {' '.join(sys.argv)}")

    with open(f'out/{cmd_args.train_id}/train.json', 'r') as f:
        train_args = json.load(f)

    merged_args = {**train_args, **vars(cmd_args)} # el segon diccionari sobreescriu el primer segons Copilot
    args = argparse.Namespace(**merged_args)

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
    # té sentit fer el shuffle en el test???
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    pad_idx = mydata.token_dict.pad


    tgt_len = args.context_window + 1  # added <bos> or <eos>
    ntokens = len(mydata.token_dict)

    mytext_criterion = nn.NLLLoss(ignore_index=pad_idx)  # És això duplicació de codi?
    myrating_criterion = nn.MSELoss()

    peter_loss = lambda predicted, real: loss(predicted, real, args.context_reg, args.text_reg, args.rating_reg,
                                              mytext_criterion, myrating_criterion, ntokens, tgt_len)

    # Run on test data.
    test_losses = test(test_dataloader, mymodel, peter_loss, mydevice)
    c_loss, t_loss, r_loss, loss = test_losses
    peter_logger.info('=' * 89)
    peter_logger.info(f"{now_time()}{content(c_loss, t_loss, r_loss)} on test")

    # ara mateix només ho escriu en el peter.log
    # possiblement hauria de escriure-ho per pantalla tmb, i veure q vull exactament posar en el text
    # ara en el test tampoc no es veu la barra del progrés pq la he tret en general de quan es feia test,
    # ja que quan es fa el test amb la validation no aportava realemnt gaire

