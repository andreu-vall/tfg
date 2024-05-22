   

def peter_loss(loss_input, loss_output, text_criterion, rating_criterion, context_reg, text_reg, rating_reg, ntokens):
    
    log_word_prob, log_context_dis, predicted_rating = loss_input
    target_text, real_rating = loss_output

    # Ara mateix estic ignorant la tasca de context amb això
    if log_context_dis is None:
        assert False
        # context_loss = 0
    else:
        # el context tendeix a servir només per predir les més common stop words: i this the a it is and .
        # Bastant estrany la veritat això del context
        
        # print('log_context_dis:', log_context_dis.shape) # [128, 13606]
        # print('target_text:', target_text.shape) # [13, 128]
        # LOL per fer lo del context es repeteix el mateix molts cops??? que lleig ngl
        
        # print('before unsqueeze:', log_context_dis.shape) # [128, 12867]
        # print('after unsqueeze:', log_context_dis.unsqueeze(0).shape) # [1, 128, 12867]
        # print('after repeat:', log_context_dis.unsqueeze(0).repeat((target_text.shape[0], 1, 1)).shape) # [11, 128, 12867]

        # D'aquesta manera en el context només es premia predir les paraules més comunes, en qualsevol ordre
        # A més el context crec que només serveix per regularitzar els embeddings a partir dels quals es calcula,
        # que són user, item i rating. Perquè pel que fa a la predicció final no s'utilitza absolutament per res.
        # Igual que la predicció que feien de rating, també només s'utilitzava per modificar els embeddings de user, item.

        # Potser la tasca del context és podria fer que realment sigués més útil si ignorés les paraules més comuns i
        # que realment servís per predir les paraules clau de la review en qualsevol ordre. De tota manera estem en la
        # limitació que el que serveix el context és només per modificar lleugerament els embeddings

        # Perquè la predicció de text depèn molt tot de la posició de cada paraula, en canvi almenys en el context no
        # importa l'ordre de les paraules, sinó la freqüència i si apareixen o no i quants cops

        # assert False, 'stop'
        context_dis = log_context_dis.unsqueeze(0).repeat((target_text.shape[0], 1, 1))
        # # print('context_dis:', context_dis.shape) # [12, 128, 13606]
        context_loss = text_criterion(context_dis.view(-1, ntokens), target_text.reshape((-1,)))
        # Ara mateix no esitc fent res de context_loss i igualment prediu coses?
        # context_loss = 0
    
    # A MIRAR QUAN TORNI DE RENTAR-ME LES DENTS AQUÍ

    # Es compara directament tot lo predict amb el target_text

    # input_text: text[:-1], output_text: text[1:]

    # print('log_word_prob shape:', log_word_prob.shape) # [11, 128, 11043]
    # És el que dona les probabilitats per cada hidden que ha usat:
    # extra_input: user, item, rating; real_text till some token.
    # tries to predict the very next token, and gives a value to every single token in the vocabulary

    # Això és simplement directament el output_text, que vindria a ser el que agradaria que generés
    # print('target_text shape:', target_text.shape) # [11, 128]


    # NOMÉS ES PREMIA QUE ENCERTI ELS TOKENS. PER TANT TENDIRÀ A GENERAR ELS TEXTOS MES COMUNS
    
    text_loss = text_criterion(log_word_prob.view(-1, ntokens), target_text.reshape((-1,)))

    rating_loss = rating_criterion(predicted_rating, real_rating)

    # els valors de PETER de context_reg, text_reg, rating_reg són una mica estranys: 1, 1, 0.1
    # em sembla estrany que li donin TANTA importància al context com al text sencer, i el rating a penes importa?
    total_loss = context_loss * context_reg + text_loss * text_reg + rating_loss * rating_reg

    return total_loss, context_loss, text_loss, rating_loss

