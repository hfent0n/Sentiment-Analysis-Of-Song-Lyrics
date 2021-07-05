# Emotion/Sentiment Analysis of lyrics

## Usage
### MoodyLyrics
[MoodyLyrics](https://core.ac.uk/download/pdf/76535286.pdf) is a dataset containing ~2000 song lyrics labelled with "sad", "happy", "angry" or "relaxed". This helpfully follows [Russell's Circumplex model of emotion](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367156/). I've included some helpful tools in the MoodyLyrics directory to download and prepare the dataset for use in sentiment analysis.
####  ./MoodyLyrics/lyrics.py
Example of downloading MoodyLyrics songs in JSON format. 
```
python MoodyLyrics/lyrics.py ml --dir ./path/to/ml_balanced.xlsx
```
Example of downloading an individual song in JSON format. 
```
python MoodyLyrics/lyrics.py song kanye "gold digger"
```
Example of downloading an artist's discography by downloading their albums.
```
python MoodyLyrics/lyrics.py album kanye
```

####   ./MoodyLyrics/json_lines.py
The models use the lyrics in JSON lines format, which the json_lines.py script provides. Preparing the data has 3 options:

 - Directory of the MoodyLyrics processed by `lyrics.py`
 - The emotion model, which is either Russell's Circumplex model of emotion on a 2D plot or the four quadrants as separate categories (2d/4cat)
 - Train/Test split proportion
```
python MoodyLyrics/json_lines.py --dir ./MoodyLyrics/ml_lyrics --emotion_model 2d --split 70
```
### Pretrained Models
The pretrained models are the best models saved from running for 20 epochs. To use these models run the `predictor.py` script with the input, emotion model (2d/4cat), and deep learning model (cnn/lstm/clstm/lstmcnn) e.g.
```
python predictor.py "Happy Birthday to you, happy birthday to you, happy birthday dear USA happy birthday to you." 
--emotion_model 4cat --dl_model clstm 
```
### Training
To train new models use the `train.py` script with arguments
 - Emotion model (2d/4cat)
 - Deep learning model  (cnn/lstm/clstm/lstmcnn)
 - Batch size
 - Epochs

```
python train.py --emotion_model 4cat --dl_model lstmcnn --bs 128 --n 20
```
## Discussion
### Sentiment Analysis
Deep learning models were chosen over classical machine learning algorithms, because of their promise in recent years in many open-ended tasks and for their recent breakthroughs into the state-of-the-art of NLP [1]. Classic models rely on good hand-selected features, whereas deep learning models learn the features themselves [2]. To achieve the best results, we tested several deep learning models implemented in [PyTorch](https://pytorch.org/) from the range of architectures being applied to similar problems. We tested each of these models in a multi-class classification domain and in a two-dimensional domain.

### Pre-Processing and Word Embeddings

The lyrics were passed through a standard cleaning process, removing annotations e.g., “[Chorus]”, transforming to lower case, and removing white spice beyond single spacing. The lyrics were then tokenised with [spaCy](https://spacy.io/) using their model for English. To reduce on load, this is done once and serialised into JSON format and split into test and train files. All models use pretrained word-embeddings. These embeddings transfer each word to vector space where words with similar semantic meaning are close together allowing the model to learn by analogy of similar words more easily. An example of the power of these embeddings is the ability to perform analogical reasoning using vector algebra in this vector space e.g., “king” is to “queen” what “brother” is to “sister”. The embeddings also help the model not to overfit because they are trained on a very large and general dataset whereas if we trained our own, the model would only see our relatively tiny dataset. We decided to use the 100-dimension [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings trained on 27 billion tokens from 2 billion tweets. These embeddings are calculated by learning the ratios of occurrence probabilities between any two words. The [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) embeddings, calculated by using the morphological structure of words, performed better over a range of datasets and models [3], however GloVe embeddings were the most popular among the top spots in 2019 SemEval [4] text emotion detection task.


### LSTM/BiLSTM

In the 2019 SemEval [4] task to train a deep learning model to extract emotion from sentences many of the top entries unsurprisingly employed Long short-term memory (LSTM) [5] models to take advantage of the sequential nature of language. Each cell takes as input, the input at that time step, the output of the previous cell and its own hidden state. This cells learn the important temporal dynamics of the data. LSTMs mitigate the vanishing gradient problem of standard recurrent neural networks by using gates to control the flow of information the sequence so the network can learn which information to hold on to and use many time steps ahead.

### CNN

A significant minority used convoluted neural networks (CNN), which are often used effectively in tasks like image classification because of their ability to break down the problem space, extracting fundamental features and building more complex feature extractors with each layer even when trained on very different tasks [6]. This is an attractive quality for our task because of the limitations in our training data discussed earlier. They work by passing filters of various sizes over the input and pooling these together to gleam the most important information. In our 1 dimensional input of text, the filters act like n-gram feature filters, learning which short sequence of words are the most important.

[For both baseline models we followed a tutorial on Github](https://github.com/bentrevett/pytorch-sentiment-analysis).

### CLSTM and LSTM-CNN

We are not limited to choosing one of these models and indeed many current works are combinations of these models (and more) in the hopes of gaining all the advantages. For example, combining CNNs with LSTMs in different ways was studied in [7] where the author evaluates the performance of the models on determining 1 dimensional sentiment (negative, neutral, positive) of a large corpus of automatically labelled twitter posts. Our task presents different challenges with much longer sequences and multiple classes. Considering the long sequence length and the common structural composition of songs (words form lines, which form verses etc.), a CNN feeding into an LSTM makes sense. The intuition behind this is that the CNN works as a feature extractor at a lower level and the LSTM can process sequences of these features.

Perhaps more intuitively, the combination can be done in reverse. This way the LSTM processes the sequence, which the CNN can filter for local features and make a prediction on the sentiment. We implemented both a CNN and an LSTM/BiLSTM as a baseline and then combined these. We found that using BiLSTMs gave better results as they were able to extract more relevant information from the sequences and using the LSTM first gave the best results.

## Results
Using the MoodyLyrics dataset, the results of the models predictions for emotion of the lyrics on the validation set are presented:
||  LSTM | CNN | CLSTM | LSTM-CNN |
|--|--|--|--| --|
|4 categories| 61.00% | 69.79% | 47.29%| **73.39%**
|2 dimensions |80.86%|**84.424%**|68.18%| 84.20%

## References
[1] Acheampong, F.A., Wenyu, C. and Nunoo‐Mensah, H., 2020. Text‐based emotion detection: Advances, challenges, and opportunities. _Engineering Reports_, _2_(7), p.e12189.

[2] Tsaptsinos, A., 2017. Lyrics-based music genre classification using a hierarchical attention network. _arXiv preprint arXiv:1707.04678_.

[3] Polignano, M., Basile, P., de Gemmis, M. and Semeraro, G., 2019, June. A comparison of word-embeddings in emotion detection from text using bilstm, cnn and self-attention. In _Adjunct Publication of the 27th Conference on User Modeling, Adaptation and Personalization_ (pp. 63-68).

[4] Chatterjee A, Narahari KN, Joshi M, Agrawal P. SemEval-2019 task 3: EmoContext contextual emotion detection in text. InProceedings of the 13th International Workshop on Semantic Evaluation 2019 Jun (pp. 39-48).

[5] Hochreiter, S. and Schmidhuber, J., 1997. Long short-term memory. _Neural computation_, _9_(8), pp.1735-1780.

[6] Sharif Razavian, Ali, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. "CNN features off-the-shelf: an astounding baseline for recognition." In _Proceedings of the IEEE conference on computer vision and pattern recognition workshops_, pp. 806-813. 2014.

[7]  Sosa, P.M., 2017. Twitter sentiment analysis using combined LSTM-CNN models. _Eprint Arxiv_, pp.1-9.
