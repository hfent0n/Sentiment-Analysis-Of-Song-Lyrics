import torch

def accuracy(preds, y):
    try: # Checking for 2d
        y.shape[1] ## for 2d the outputs need to be normalized in the range 0-1 through the sigmoid function.
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() 
        acc = correct.sum() / correct.nelement()   
    except IndexError:
        top_pred = preds.argmax(1, keepdim = True) ## Take the maximum of the 4 outputs for category prediction
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, criterion, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        
        # Make the model aware of padding for the RNN
        if model.id == 'lstm':
            lyrics, lyrics_lengths = batch.lyrics
            predictions = model(lyrics, lyrics_lengths)
        else:
            predictions = model(batch.lyrics)
        
        # Concatenate the arousal and valence of the input and compute BCE loss on both
        if model.output_dim == 2:
            predictions = predictions.squeeze(1)
            label = torch.cat((batch.arousal.unsqueeze(1), batch.valence.unsqueeze(1)), dim=1)
            loss = criterion(predictions, label)
            acc = accuracy(predictions, label)
        # Compute Cross entropy loss for categorical predictions
        else:
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
       
            # Make the model aware of padding for the RNN
            if model.id == 'lstm':
                lyrics, lyrics_lengths = batch.lyrics
                predictions = model(lyrics, lyrics_lengths)
            else:
                predictions = model(batch.lyrics)
            
            # Concatenate the arousal and valence of the input and compute BCE loss on both
            if model.output_dim == 2:
                predictions = predictions.squeeze(1)
                label = torch.cat((batch.arousal.unsqueeze(1), batch.valence.unsqueeze(1)), dim=1)
                loss = criterion(predictions, label)
                acc = accuracy(predictions, label)
            # Compute Cross entropy loss for categorical predictions
            else:
                loss = criterion(predictions, batch.label)
                acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
import time