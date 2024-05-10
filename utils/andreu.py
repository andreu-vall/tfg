import math
import logging


# Aquest fitxer possiblement no cal? O on l'hauria de posar que tingui més sentit


def setup_logger(name, log_file, stdout=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))    # Log to file
    if stdout:
        logger.addHandler(logging.StreamHandler())      # Log to stdout
    return logger


# Moure a GPU i transposar seq i feature. Segons Copilot és més estàndard moure les coses a GPU
# si cal en el train enlloc del dataloader, perquè així compta en el temps del train
# Hauria ara de definit on vull fer les transposicions jo i per què
def move_content_to_device(content, device):
    user, item, rating, seq = content
    # batch_size = user.size(0)

    user = user.to(device)  # (batch_size,)
    item = item.to(device)
    rating = rating.to(device)

    # Amb això de la transposició he de decidir si la faig i a on i per què
    seq = seq.t().to(device)    # (tgt_len + 1, batch_size)
    
    return user, item, rating, seq


# context_reg, text_reg, rating_reg: la importància relativa de les 3 tasques a optimitzar
def peter_loss_good(pred, content, context_reg, text_reg, rating_reg, text_criterion, rating_criterion, ntokens, tgt_len):

    # Això és la clau de l'entrenament. Depenen del que si posis loss aprendrà a fer una
    # cosa o altra el model

    user, item, rating, seq = content
    
    # content és el real, la cosa del batch sencer que em dona algun DataLoader
    # pred és el predicted amb model(user, item, text)
    # Hi ha un pas intermig on es calcula un hidden amb el transformer_encoder
    # i després es descodifiquen coses derivades d'aquest:
    # concretament és [log_word_prob, log_context_dis, rating, attns]

    # rating usa un MLP i hidden[0]
    # log_context_dis simplement literalment descodifica hidden[1] com a words and calls it el context
    # log_word_prob simplement literalment descodifica hidden[2:]
        # Hi ha el cas especial que no acabo d'entendre del tot que només descodifica una única paraula
        # crec que és per quan fa el decoding, que en realitat ho fa bastant estrany. No sé si és gaire
        # estàndard fer-ho així manualment, o si s'usen Decoders més sofisticats

    # Per descodificar aquestes dues coses s'utilitza un nn.Linear(emsize, ntoken),
    # de manera que crec que dona totes les probabilitats per tots els tokens del model



    log_word_prob, log_context_dis, rating_p, _ = pred  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

    # La predicció del context és simplement predir quines paraules és possible que surint en l'explicació que 
    # estan intentant predir, sense importar l'ordre, de manera que aprengui les paraules probables a utilitzar
    # L'argument dels de PETER és que si no feien això, que ho van introduir ells, el transformer no personalitzava
    # les explicacions i aprenia a generar la mateixa explicació per tothom, el que passava amb el POD després
    # de gastar un munt d'hores i recursos entrenant amb GPU...

    # Crec que simplement li treu el beggining i ending token i es fa directe la comparació sencera
    c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,))) # This is a bit ugly
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
    r_loss = rating_criterion(rating_p, rating)

    loss = c_loss * context_reg + t_loss * text_reg + r_loss * rating_reg # ordre més normal ara

    return c_loss, t_loss, r_loss, loss


def peter_content(context_loss, text_loss, rating_loss):
    exp_context_loss = math.exp(context_loss)
    exp_text_loss = math.exp(text_loss)
    return f"context ppl {exp_context_loss:4.4f} | text ppl {exp_text_loss:4.4f} | rating loss {rating_loss:4.4f}"
