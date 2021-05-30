
import argparse
import time
import torch
import torch.nn as nn

import torch.optim as optim
from Models import CNN, RNN, CLSTM
from prepare_data import *
from train_eval import train, evaluate

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # Initial set up of PyTorch
    MAX_VOCAB_SIZE  = 25_000
    SEED            = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # User options 
    parser = argparse.ArgumentParser()
    parser.add_argument('emotion_model', type=str.lower, default="4cat", choices = ['2d', '4cat'], help='2d/4cat: use 2d (2d) or 4 categories (4cat) models')
    parser.add_argument('dl_model', type=str.lower, default="cnn", choices = ['cnn', 'lstm', 'clstm'], help='deep learning model to use: CNN, LSTM, CLSTM')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--n', type=int, default=20, help='epochs')
    args = parser.parse_args()
    emotion_model   = args.emotion_model
    dl_model        = args.dl_model
    BATCH_SIZE      = args.bs
    N               = args.n
    
    #Prepare lyrics dataset depending on selected emotion and dl model
    if emotion_model == '4cat':
        LYRICS, LABEL, train_iterator, valid_iterator, test_iterator = prepare_data(emotion_model, dl_model, device, BATCH_SIZE, SEED, MAX_VOCAB_SIZE)
        INPUT_DIM       = len(LYRICS.vocab)
        OUTPUT_DIM      = len(LABEL.vocab)
    elif emotion_model == '2d':
        LYRICS, AROUSAL, VALENCE, train_iterator, valid_iterator, test_iterator = prepare_data(emotion_model, dl_model, device, BATCH_SIZE, SEED, MAX_VOCAB_SIZE)
        INPUT_DIM       = len(LYRICS.vocab)
        OUTPUT_DIM      = 2
    
    PAD_IDX = LYRICS.vocab.stoi[LYRICS.pad_token]    
    
    # Parameters
    if dl_model == "cnn":
        EMBEDDING_DIM   = 100
        N_FILTERS       = 100
        FILTER_SIZES    = [3, 4, 5]
        DROPOUT         = 0.5
        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    
    elif dl_model == "lstm":
        EMBEDDING_DIM   = 100
        HIDDEN_DIM      = 100
        N_LAYERS        = 2
        BIDIRECTIONAL   = True
        DROPOUT         = 0.5
        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
        
    elif dl_model == "clstm":
        EMBEDDING_DIM   = 100
        N_FILTERS       = 100
        FILTER_SIZES    = [3, 4, 5]
        HIDDEN_DIM      = 100
        N_LAYERS        = 2
        BIDIRECTIONAL   = True
        DROPOUT         = 0.5
        model = CLSTM(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 
                        OUTPUT_DIM, N_LAYERS, HIDDEN_DIM, BIDIRECTIONAL, DROPOUT)


    # Load GloVe embeddings into the model
    pretrained_embeddings = LYRICS.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Zero the unknown and padding tokens
    UNK_IDX = LYRICS.vocab.stoi[LYRICS.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # Loss and optimzation
    optimizer = optim.Adam(model.parameters())

    if emotion_model == '4cat':
        criterion = nn.CrossEntropyLoss()
    
    elif emotion_model == '2d':
        criterion = nn.BCEWithLogitsLoss()
        
    model = model.to(device)
    criterion = criterion.to(device)
    
    
    
    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(N):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, criterion, optimizer)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'best-{dl_model}_{emotion_model}_model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    


