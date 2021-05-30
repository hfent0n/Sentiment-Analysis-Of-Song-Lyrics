import json
import re
import glob
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



with open ("lyrics_train_2d_int.json", "w", encoding='utf-8') as train_file, open("lyrics_test_2d_int.json", "w", encoding='utf-8') as test_file:
    for i, song in enumerate(glob.glob('./lyrics/*_relaxed..json')):
        with open(song) as file:
            x = json.load(file)
            newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": -1, "valence": 1}
            if i > 340:
                json.dump(newObj, train_file)
                train_file.write('\n')
            else:
                json.dump(newObj, test_file)
                test_file.write('\n')
    for i, song in enumerate(glob.glob('./lyrics/*_sad..json')):
        with open(song) as file:
            x = json.load(file)
            newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": -1, "valence": -1}
            if i > 340:
                json.dump(newObj, train_file)
                train_file.write('\n')
            else:
                json.dump(newObj, test_file)
                test_file.write('\n')
    for i, song in enumerate(glob.glob('./lyrics/*_happy..json')):
        with open(song) as file:
            x = json.load(file)
            newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": 1, "valence": 1}
            if i > 340:
                json.dump(newObj, train_file)
                train_file.write('\n')
            else:
                json.dump(newObj, test_file)
                test_file.write('\n')
    for i, song in enumerate(glob.glob('./lyrics/*_angry..json')):
        with open(song) as file:
            x = json.load(file)
            newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": 1, "valence": -1}
            if i > 340:
                json.dump(newObj, train_file)
                train_file.write('\n')
            else:
                json.dump(newObj, test_file)
                test_file.write('\n')
       





