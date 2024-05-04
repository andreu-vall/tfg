import math
from utils.peter import now_time

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



# Això és el que feien en el PETER. Per la validació s'ignora el loss de context (i maybe tmb el de rating)
# variables de args per la cara: rating_reg (mig solucionat però no provat yet)
# He de borrar probably la variable rating_reg
def peter_print_long(val_losses, rating_reg, name='validation'):
    c_loss, t_loss, r_loss, real_loss = val_losses
    printed_loss = t_loss
    if rating_reg != 0: # what even is rating_reg?
        printed_loss += r_loss
    print(f"{now_time()}{peter_content(c_loss, t_loss, r_loss)} | valid loss {printed_loss:4.4f} on {name}. Real: {real_loss:4.4f}")



# context_reg, text_reg, rating_reg: la importància relativa de les 3 tasques a optimitzar
def peter_loss_good(pred, content, context_reg, text_reg, rating_reg, text_criterion, rating_criterion, ntokens, tgt_len):
    user, item, rating, seq, feature = content

    log_word_prob, log_context_dis, rating_p, _ = pred  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

    c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,))) # This is a bit ugly
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
    r_loss = rating_criterion(rating_p, rating)

    loss = c_loss * context_reg + t_loss * text_reg + r_loss * rating_reg # ordre més normal ara

    return c_loss, t_loss, r_loss, loss


# De moment segueixo amb els prints de PETER per comparar fàcilment. Però més endavant potser seria millor
# canviar a logging com l'Alejandro. Així per exemple podria fer servir el tqdm que és més pro i útil.
# Però aquest canvi el faré més endavant, crec que de moment deixaré els prints del PETER exactament igual
# i a més afegiré els meus amb un format potser similar i amb algo que permeti identificar que és el meu
# per quan hagi de parsejar uns fitxer i altres ho pugui distingir fàcilment si els del PETER tmb ho tenen
# o és una introducció meva

# Si els números són molt grans no dona tot de la mateixa mida tampoc...
def peter_content(context_loss, text_loss, rating_loss):
    exp_context_loss = math.exp(context_loss)
    exp_text_loss = math.exp(text_loss)
    return f"context ppl {exp_context_loss:4.4f} | text ppl {exp_text_loss:4.4f} | rating loss {rating_loss:4.4f}"

