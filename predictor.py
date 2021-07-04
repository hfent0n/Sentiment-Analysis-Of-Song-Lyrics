# Framework of code adapted from the tutorials. 
# https://github.com/bentrevett/pytorch-sentiment-analysis
# This script uses pretrained models to predict the emotion behind lyrics.
import argparse
import torch
import torch.nn as nn
from models import CNN, RNN, CLSTM, LSTMCNN
from prepare_data import *
from utils import predict_sentiment

if __name__ == "__main__":
    # Initial set up of PyTorch
    MAX_VOCAB_SIZE  = 25_000
    SEED            = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # User options 
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str.lower, help='input lyrics')
    parser.add_argument('emotion_model', type=str.lower, default="4cat", choices = ['2d', '4cat'], help='2d/4cat: use 2d (2d) or 4 categories (4cat) models')
    parser.add_argument('dl_model', type=str.lower, default="cnn", choices = ['cnn', 'lstm', 'clstm', 'lstmcnn'], help='deep learning model to use: CNN, LSTM, CLSTM')
    
    args = parser.parse_args()
    input_lyrics           = args.input
    emotion_model   = args.emotion_model
    dl_model        = args.dl_model
    BATCH_SIZE      = 1
    
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
    
    elif dl_model == "lstmcnn":
        EMBEDDING_DIM   = 100
        N_FILTERS       = 100
        FILTER_SIZES    = [3, 4, 5]
        HIDDEN_DIM      = 100
        N_LAYERS        = 2
        BIDIRECTIONAL   = True
        DROPOUT         = 0.5
        model = LSTMCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 
            OUTPUT_DIM, N_LAYERS, HIDDEN_DIM, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    if emotion_model == '4cat':
        criterion = nn.CrossEntropyLoss()
    
    elif emotion_model == '2d':
        criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    model.load_state_dict(torch.load(f'./pretrained_models/best_{dl_model}_{emotion_model}_model.pt'))
    sentiment = predict_sentiment(LYRICS, device, model, input_lyrics) 
    print(sentiment)

