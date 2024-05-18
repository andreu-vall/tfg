   



# em falta: tgt_len, ntokens. Ho poso paràmetres de la funció?

def peter_loss(loss_input, loss_output, text_criterion, rating_criterion,
               context_reg, text_reg, rating_reg, ntokens, tgt_len):
        
        log_word_prob, log_context_dis, predicted_rating = loss_input
        target_text, real_rating = loss_output

        # en algun lloc de de posar les dependències de variables globals encara

        context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))
        context_target_text = target_text[:-1] # els de PETER treien l'últim token fsr
        # what ara segueix petant? tot i haver-ho canviat abans?
        # s'han perdut tokens per algun lloc? peta lo altre, el context_dis.view(-1, ntokens),
        # com és que ha canvait això??

        context_loss = text_criterion(context_dis.view(-1, ntokens), context_target_text.reshape((-1,)))

        text_loss = text_criterion(log_word_prob.view(-1, ntokens), target_text.reshape((-1,)))
        rating_loss = rating_criterion(predicted_rating, real_rating)

        total_loss = context_loss * context_reg + text_loss * text_reg + rating_loss * rating_reg

        return total_loss, context_loss, text_loss, rating_loss


