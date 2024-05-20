   

def peter_loss(loss_input, loss_output, text_criterion, rating_criterion, context_reg, text_reg, rating_reg, ntokens, context_window):
        
    log_word_prob, log_context_dis, predicted_rating = loss_input
    target_text, real_rating = loss_output

    if log_context_dis is None:
        context_loss = 0
    else:
        context_dis = log_context_dis.unsqueeze(0).repeat((context_window - 1, 1, 1))
        context_loss = text_criterion(context_dis.view(-1, ntokens), target_text.reshape((-1,)))

    text_loss = text_criterion(log_word_prob.view(-1, ntokens), target_text.reshape((-1,)))

    rating_loss = rating_criterion(predicted_rating, real_rating)

    # els valors de PETER de context_reg, text_reg, rating_reg són una mica estranys: 1, 1, 0.1
    # em sembla estrany que li donin TANTA importància al context com al text sencer, i el rating a penes importa?
    total_loss = context_loss * context_reg + text_loss * text_reg + rating_loss * rating_reg

    return total_loss, context_loss, text_loss, rating_loss

