'''json_lines.py is for converting the MoodyLyrics jsons into json-lines format.'''

import argparse
import os
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


def write_json_lines(path, emotion_model, split):
    with open ("../data/lyrics_train_" + emotion_model + ".json", "w", encoding='utf-8') as train_file, \
         open("../data/lyrics_test_" + emotion_model + ".json", "w", encoding='utf-8') as test_file:
        relaxed = glob.glob(path + '/*_relaxed.json')
        relaxed_split = int(split*len(relaxed))
        for i, song in enumerate(relaxed):
            with open(song) as file:
                x = json.load(file)
                if emotion_model == '2d':
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": -1, "valence": 1}
                else:
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "label": "relaxed"}
                if i < relaxed_split:
                    json.dump(newObj, train_file)
                    train_file.write('\n')
                else:
                    json.dump(newObj, test_file)
                    test_file.write('\n')
        sad = glob.glob(path + '/*_sad.json')
        sad_split = int(split*len(sad))
        for i, song in enumerate(sad):
            with open(song) as file:
                x = json.load(file)
                if emotion_model == '2d':
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": -1, "valence": -1}
                else:
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "label": "sad"}
                if i < sad_split:
                    json.dump(newObj, train_file)
                    train_file.write('\n')
                else:
                    json.dump(newObj, test_file)
                    test_file.write('\n')
        happy = glob.glob(path + '/*_happy.json')
        happy_split = int(split*len(happy))
        for i, song in enumerate(happy):
            with open(song) as file:
                x = json.load(file)
                if emotion_model == '2d':
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": 1, "valence": 1}
                else:
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "label": "happy"} 
                if i < happy_split:
                    json.dump(newObj, train_file)
                    train_file.write('\n')
                else:
                    json.dump(newObj, test_file)
                    test_file.write('\n')
        angry = glob.glob(path + '/*_happy.json')
        angry_split = int(split*len(angry))
        for i, song in enumerate(angry):
            with open(song) as file:
                x = json.load(file)
                if emotion_model == '2d':
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "arousal": 1, "valence": -1}
                else:
                    newObj = {"lyrics": [token.text for token in clean(x["lyrics"])], "label": "angry"} 
                if i < angry_split:
                    json.dump(newObj, train_file)
                    train_file.write('\n')
                else:
                    json.dump(newObj, test_file)
                    test_file.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./ml_lyrics", help='filepath to directory containing processed MoodyLrics lyrics in json format')
    parser.add_argument('--emotion_model', type=str, choices=['2d', '4cat'], default='2d', help='2d/4cat: use 2d (2d) or 4 categories (4cat) models')
    parser.add_argument('--split', type=int, default=70, help='% of train data in train/test split')
    
    args = parser.parse_args()
    path = args.dir
    emotion_model = args.emotion_model
    split = args.split / 100
    os.mkdir('../data')
    write_json_lines(path, emotion_model, split)