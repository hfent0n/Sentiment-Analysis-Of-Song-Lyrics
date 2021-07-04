import re
import torch
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

def clean(lyrics):
    lyrics_no_brackets = re.sub("[\(\[].*?[\)\]]", "", lyrics)
    lyrics_lower = lyrics_no_brackets.lower()
    lyrics_single_spaced = re.sub('\n+', '\n', lyrics_lower)
    lyrics_= re.sub('^\n+', '', lyrics_single_spaced)
    tokens = tokenizer(lyrics_)
    return tokens

def predict_sentiment(LYRICS, device, model, unclean_lyrics):
    model.eval()

    tokenized = clean(unclean_lyrics)
    indexed = [LYRICS.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = list(map(lambda x: x*2 - 1, torch.sigmoid(model(tensor, length_tensor)).squeeze()))
    return f'Arousal: {prediction[0]}\nValence: {prediction[1]}'