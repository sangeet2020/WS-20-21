# LSTM Based Parts-of-Speech Tagger
## Introduction
This work proposes the use of Long Short-Term Memory (LSTM) based models with word embeddings
that use sub-word information to perform Parts-of-Speech (POS) tagging. The proposed models include LSTM models and Bi-LSTM models and their comparison with a bigram Hidden Markov Models (HMM) based parts-of-speech tagger that uses the Viterbi algorithm.

## Description
The script:
- `main.py`: this is the main script that you should be running. This script automatically downloads the required training data and pre-trained embeddings for English language.


## Requirements
The scripts have been tested on:

- Python: `3.8.3`
- Numpy: `1.19.2`
- Keras: `2.4.3`


## Project file structure
```
├── BiLSTM
│   ├── accuray_plot_Bi-LSTM.eps
│   ├── Bi_LSTM_arch.png
│   └── loss_plot_Bi-LSTM.eps
├── BiLSTM_subword
│   ├── accuray_plot_Bi-LSTM.eps
│   ├── loss_plot_Bi-LSTM.eps
│   └── model_arch.png
├── EN-evaluation-metrics-results.text
├── LSTM
│   ├── accuray_plot_LSTM.eps
│   ├── loss_plot_LSTM.eps
│   └── LSTM_arch.png
├── LSTM_subword
│   ├── accuray_plot_LSTM.eps
│   ├── loss_plot_LSTM.eps
│   └── model_arch.png
├── main.py
├── README.md
├── wiki-news-300d-1M-subword.vec
├── wiki-news-300d-1M-subword.vec.zip
├── wiki-news-300d-1M.vec
└── wiki-news-300d-1M.vec.zip
    
```

## Usage

- **Help**: for instructions on how to run the script with appropriate arguments.
    ```        
    python main.py --help

    usage: main.py [-h] [-subword_info SUBWORD_INFO] out_dir {LSTM,BiLSTM}
    LSTM based POS tagger

    positional arguments:
    out_dir               path to save the plots and results
    {LSTM,BiLSTM}         Choice of training architecture

    optional arguments:
    -h, --help            show this help message and exit
    -subword_info SUBWORD_INFO use pre-embeddings with subword information

    ```
- 
    ```
    python main.py LSTM  LSTM > LSTM.log
    ```

    ```
    python main.py BiLSTM  BiLSTM > BiLSTM.log
    ```
    ```
    python main.py LSTM_subword  LSTM -subword_info True > LSTM_subword.log 
    ```
    ```
    python main.py BiLSTM_subword  BiLSTM -subword_info True > BiLSTM_subword.log
    ```

## Datatset
This work on POS tagging uses for English and German language uses
- English:
    - Wall Street Journal and Brown data from Penntreebank III
    - Universal Tagset
    - CoNLL 200 data
- German:
    - German POS tagger data

## Runtime

- Total runtime: 
    - LSTM- 682.752 s
    - Bi-LSTM- 956.790 s


## Results
- LSTM without subword information

```
Loss: 4.08 %,
Accuracy: 98.76 %
Total runtime: 682.752 s

    precision    recall  f1-score   support

           0       0.99      1.00      0.99   1140671
           1       0.96      0.76      0.85     76773
           2       0.78      0.97      0.86     46407
           3       1.00      0.86      0.92     38304
           4       0.80      0.99      0.88     36327
           5       0.97      0.95      0.96     33827
           6       0.91      0.68      0.78     21482
           7       0.79      0.85      0.82     13639
           8       0.89      0.93      0.91     11929
           9       0.99      0.99      0.99      9270
          10       0.73      0.08      0.15      8252
          11       0.97      0.92      0.94      5618
          12       0.90      0.30      0.45      1601

    accuracy                           0.97   1444100
   macro avg       0.90      0.79      0.81   1444100
weighted avg       0.97      0.97      0.97   1444100
```

- Bi-LSTM without subword information
```
Loss: 2.62 %,
Accuracy: 99.15 %
Total runtime: 956.790 s

             precision    recall  f1-score   support

           0       0.99      1.00      0.99   1140671
           1       0.98      0.79      0.88     76773
           2       0.86      0.95      0.90     46407
           3       1.00      0.86      0.92     38304
           4       0.84      0.94      0.89     36327
           5       0.91      0.98      0.94     33827
           6       0.82      0.89      0.85     21482
           7       0.83      0.87      0.85     13639
           8       0.92      0.93      0.92     11929
           9       0.99      0.99      0.99      9270
          10       0.72      0.20      0.32      8252
          11       0.92      0.93      0.93      5618
          12       0.83      0.16      0.27      1601

    accuracy                           0.97   1444100
   macro avg       0.89      0.81      0.82   1444100
weighted avg       0.97      0.97      0.97   1444100

```