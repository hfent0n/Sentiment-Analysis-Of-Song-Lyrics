#Code adapted from tutorials
import random
import torch
from torchtext.legacy import data

def prepare_data(emotion_model, dl_model, device, BATCH_SIZE, SEED, MAX_VOCAB_SIZE):
    if emotion_model == '4cat':
        if dl_model in ['lstm', 'lstmcnn']:
            LYRICS = data.Field(include_lengths=True)
            LABEL = data.LabelField()
        else:
            LYRICS = data.Field(batch_first=True)
            LABEL = data.LabelField()

        fields = {'lyrics': ('lyrics', LYRICS), 'label': ('label', LABEL)}

        train_data, test_data = data.TabularDataset.splits(
                            path = './data',
                            train = 'lyrics_train.json',
                            test = 'lyrics_test.json',
                            format = 'json',
                            fields = fields
                        )

        train_data, valid_data = train_data.split(random_state = random.seed(SEED))
        LYRICS.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)
        
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.lyrics),
            sort_within_batch = True, 
            device = device,
            shuffle=True)
        
        return LYRICS, LABEL, train_iterator, valid_iterator, test_iterator

    elif emotion_model == '2d':
        if dl_model == dl_model in ['lstm', 'lstmcnn']:
            LYRICS = data.Field(include_lengths=True)
        else:
            LYRICS = data.Field(batch_first=True)
        AROUSAL = data.LabelField(dtype = torch.float)
        VALENCE = data.LabelField(dtype = torch.float)

        fields = {'lyrics': ('lyrics', LYRICS), 'arousal': ('arousal', AROUSAL), 'valence': ('valence', VALENCE)}

        train_data, test_data = data.TabularDataset.splits(
                                    path = './data',
                                    train = 'lyrics_train_2d.json',
                                    test = 'lyrics_test_2d.json',
                                    format = 'json',
                                    fields = fields
                                )
        train_data, valid_data = train_data.split(random_state = random.seed(SEED))
        LYRICS.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)
        AROUSAL.build_vocab(train_data)
        VALENCE.build_vocab(train_data)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, valid_data, test_data), 
                                                            batch_size = BATCH_SIZE,
                                                            sort_key = lambda x: len(x.lyrics),
                                                            device = device,
                                                            sort_within_batch = True, 
                                                            shuffle=True
                                                        )
                                                
        return LYRICS, AROUSAL, VALENCE, train_iterator, valid_iterator, test_iterator
        
    
    