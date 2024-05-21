   

def peter_loss(loss_input, loss_output, text_criterion, rating_criterion, context_reg, text_reg, rating_reg, ntokens):
    
    log_word_prob, log_context_dis, predicted_rating = loss_input
    target_text, real_rating = loss_output

    # Ara mateix estic ignorant la tasca de context amb això
    if log_context_dis is None:
        assert False
        context_loss = 0
    else:
        # el context tendeix a servir només per predir les més common stop words: i this the a it is and .
        # Bastant estrany la veritat això del context
        
        # print('log_context_dis:', log_context_dis.shape) # [128, 13606]
        # print('target_text:', target_text.shape) # [13, 128]
        # LOL per fer lo del context es repeteix el mateix molts cops??? que lleig ngl
        # context_dis = log_context_dis.unsqueeze(0).repeat((target_text.shape[0], 1, 1))
        # # print('context_dis:', context_dis.shape) # [12, 128, 13606]
        # context_loss = text_criterion(context_dis.view(-1, ntokens), target_text.reshape((-1,)))
        # Ara mateix no esitc fent res de context_loss i igualment prediu coses?
        context_loss = 0

    text_loss = text_criterion(log_word_prob.view(-1, ntokens), target_text.reshape((-1,)))

    rating_loss = rating_criterion(predicted_rating, real_rating)

    # els valors de PETER de context_reg, text_reg, rating_reg són una mica estranys: 1, 1, 0.1
    # em sembla estrany que li donin TANTA importància al context com al text sencer, i el rating a penes importa?
    total_loss = context_loss * context_reg + text_loss * text_reg + rating_loss * rating_reg

    return total_loss, context_loss, text_loss, rating_loss

