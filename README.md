## YATO: Yet Another deep learning based Text analysis Open toolkit


## Quick Links

  - [Introduction](#Introduction)
  - [Getting Started](#getting-started)
  - [Data Format](#Data-Format)
  - [Configuration Preparation](#Configuration-Preparation)
  - [Performance](#Performance)
  - [Add Handcrafted Features](#Add-Handcrafted-Features)
- [Speed](#Speed)
- [N best Decoding](#N-best-Decoding)
- [Reproduce Paper Results and Hyperparameter Tuning](#Reproduce-Paper-Results-and-Hyperparameter-Tuning)
- [Report Issue or Problem](#Report-Issue-or-Problem)
- [Cite](#Cite)
- [Future Plan](#Future-Plan)
- [Update](#Update)

# Introduction

**YATO**, an open-source Python library for text analysis. In particular, **YATO** focuses on sequence labeling and classification tasks, including extensive fundamental NLP tasks such as part-of-speech tagging, chunking, NER, CCG super tagging, sentiment analysis, and sentence classification. **YATO** support both designing specific RNN-based and Transformer-based through user-friendly configuration and integrating the SOTA pre-trained language models such as BERT.

**YATO** is a PyTorch-based framework with flexible choices of input features and output structures.    The design of neural sequence models with **YATO** is fully configurable through a configuration file, which does not require any code work. 

A predecessor version of the framework called ****YATO**** has been accepted as a demo paper by ACL 2018. The detailed experimental report and the analysis performed using ****YATO**** were accepted as the best paper by COLING 2018.

  iooiWelcome to star this repository! 

## Getting Started

We provide an easy way to use the toolkit **YATO** from PyPI
```bash
pip install yato
```

Or directly install it from the source  code

```
git clone https://github.com/jiesutd/YATO.git
```

The code to train a Model
```python
from yato import YATO
model = YATO(configuration file)
model.train()
```
The code to decode predict file:

```
from yato import YATO
decode_model = YATO(configuration file)
result_dict = decode_model.decode()
```

return dictionary contents following value:

- speed: decode speed
- accuracy: If the decoded file contains annotation results, accuracy means verifying the accuracy
- precision:  If the decoded file contains annotation results, precision means verifying the precision
- recall:  If the decoded file contains annotation results, recall means verifying the recall
- predict_result: predict results
- nbest_predict_score: nbest decode predict scores
- label: The reflection between label and index

## Data Format

* You can refer to the data format in [sample_data](sample_data). 
* **YATO** supports both BIO and BIOES(BMES) tag schemes.  
* Notice that IOB format (***different*** from BIO) is currently not supported, because this tag scheme is old and works worse than other schemes [Reimers and Gurevych, 2017](https://arxiv.org/pdf/1707.06799.pdf). 
* The difference among these three tag schemes is explained in this [paper](https://arxiv.org/pdf/1707.06799.pdf).
* I have written a [script](utils/tagSchemeConverter.py) which converts the tag scheme among IOB/BIO/BIOES. Welcome to have a try. 

## Configuration Preparation 

You can specify the model, optimizer, and decoding through the configuration file:

### Training Configuration

#### Dataloader  
train_dir=the path of the train file    
dev_dir=the path of the validation file   
test_dir=the path of the test file    
model_dir=the path to save model weights  
dset_dir=the path of configuration encode file    

#### Model
use_crf=True/False     
use_char=True/False     
char_seq_feature=GRU/LSTM/CNN/False     
use_word_seq=True/False     
use_word_emb=True/False     
word_emb_dir=The path of word embedding file    
word_seq_feature=GRU/LSTM/CNN/FeedFowrd/False   
low_level_transformer=pretrain language model from huggingface  
low_level_transformer_finetune=True/False  
high_level_transformer=pretrain language model from huggingface  
high_level_transformer_finetune=True/False      
cnn_layer=layer number     
char_hidden_dim=dimension number      
hidden_dim=dimension number     
lstm_layer=layer number      
bilstm=True/False  

### Hyperparameters       
sentence_classification=True/False        
status=train/decode         
dropout=Dropout Rate         
optimizer=SGD/Adagrad/adadelta/rmsprop/adam/adamw    
iteration=epoch number         
batch_size=batch size           
learning_rate=learning rate         
gpu=True/False         
device=cuda:0        
scheduler=get_linear_schedule_with_warmup/get_cosine_schedule_with_warmup            
warmup_step_rate=warmup steo rate         

### Decode Configuration    
status=decode  
raw_dir=The path of decode file    
nbest=0 (NER)/1 (sentence classification)   
decode_dir=The path of decode result file  
load_model_dir=The path of model weights          
sentence_classification=True/False  

## Performance

Results on CONLL 2003 English NER task are better or comparable with SOTA results with the same structures. 

CharLSTM+WordLSTM+CRF: 91.20 vs 90.94 of [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf);

CharCNN+WordLSTM+CRF:  91.35 vs 91.21 of [Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).   

By default, `LSTM` is bidirectional LSTM.    

| ID   | Model        | Nochar | CharLSTM  | CharCNN   |
| ---- | ------------ | ------ | --------- | --------- |
| 1    | WordLSTM     | 88.57  | 90.84     | 90.73     |
| 2    | WordLSTM+CRF | 89.45  | **91.20** | **91.35** |
| 3    | WordCNN      | 88.56  | 90.46     | 90.30     |
| 4    | WordCNN+CRF  | 88.90  | 90.70     | 90.43     |

We have compared twelve neural sequence labeling models (`{charLSTM, charCNN, None} x {wordLSTM, wordCNN} x {softmax, CRF}`) on three benchmarks (POS, Chunking, NER) under statistical experiments, detail results and comparisons can be found in our COLING 2018 paper [Design Challenges and Misconceptions in Neural Sequence Labeling](https://arxiv.org/abs/1806.04470).

The results based on Pretrain Language Model were published in [YATO: Yet Another deep learning based\ Text analysis Open toolkit]()


## Add Handcrafted Features

**YATO** has integrated several SOTA neural characrter sequence feature extractors: CNN ([Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)), LSTM ([Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)) and GRU ([Yang .etc, ICLR17](https://arxiv.org/pdf/1703.06345.pdf)). In addition, handcrafted features have been proven important in sequence labeling tasks. **YATO** allows users designing their own features such as Capitalization, POS tag or any other features (grey circles in above figure). Users can configure the self-defined features through configuration file (feature embedding size, pretrained feature embeddings .etc). The sample input data format is given at [train.cappos.bmes](sample_data/train.cappos.bmes), which includes two human-defined features `[POS]` and `[Cap]`. (`[POS]` and `[Cap]` are two examples, you can give your feature any name you want, just follow the format `[xx]` and configure the feature with the same name in configuration file.)
User can configure each feature in configuration file by using 

```Python
feature=[POS] emb_size=20 emb_dir=%your_pretrained_POS_embedding
feature=[Cap] emb_size=20 emb_dir=%your_pretrained_Cap_embedding
```

Feature without pretrained embedding will be randomly initialized.


## Speed

**YATO** is implemented using fully batched calculation, making it quite effcient on both model training and decoding. With the help of GPU (Nvidia GTX 1080) and large batch size, LSTMCRF model built with **YATO** can reach 1000 sents/s and 2000sents/s on training and decoding status, respectively.

![alt text](./readme/speed.png "System speed on NER data")


## N best Decoding

Traditional CRF structure decodes only one label sequence with largest probabolities (i.e. 1-best output). While **YATO** can give a large choice, it can decode `n` label sequences with the top `n` probabilities (i.e. n-best output). The nbest decodeing has been supported by several popular **statistical** CRF framework. However to the best of our knowledge, **YATO** is the only and the first toolkit which support nbest decoding in **neural** CRF models. 

In our implementation, when the nbest=10, CharCNN+WordLSTM+CRF model built in **YATO** can give 97.47% oracle F1-value (F1 = 91.35% when nbest=1) on CoNLL 2003 NER task.

![alt text](./readme/nbest.png  "N best decoding oracle result")


## Reproduce Paper Results and Hyperparameter Tuning

To reproduce the results in our COLING 2018 paper, you only need to set the `iteration=1` as `iteration=100` in configuration file `demo.train.config` and configure your file directory in this configuration file. The default configuration file describes the `Char CNN + Word LSTM + CRF` model, you can build your own model by modifing the configuration accordingly. The parameters in this demo configuration file are the same in our paper. (Notice the `Word CNN` related models need slightly different parameters, details can be found in our COLING paper.)

If you want to use this framework in new tasks or datasets, here are some tuning [tips](readme/hyperparameter_tuning.md) by @Victor0118.


## Report Issue or Problem

If you want to report an issue or ask a problem, please attach the following materials if necessary. With these information, I can give fast and accurate discussion and suggestion. 

* `log file` 
* `config file` 
* `sample data` 


## Cite

If you use **NCRF++** in your paper, please cite our [ACL demo paper](https://arxiv.org/abs/1806.05626):

    @inproceedings{yang2018ncrf,  
     title={**YATO**: An Open-source Neural Sequence Labeling Toolkit},  
     author={Yang, Jie and Zhang, Yue},  
     booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
     Url = {http://aclweb.org/anthology/P18-4013},
     year={2018}  
    }


If you use experiments results and analysis of **NCRF++**, please cite our [COLING paper](https://arxiv.org/abs/1806.04470):

    @inproceedings{yang2018design,  
     title={Design Challenges and Misconceptions in Neural Sequence Labeling},  
     author={Yang, Jie and Liang, Shuailong and Zhang, Yue},  
     booktitle={Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
     Url = {http://aclweb.org/anthology/C18-1327},
     year={2018}  
    }

## Future Plan 

* Document classification (working)
* Support API usage
* Upload trained model on Word Segmentation/POS tagging/NER

## Updates

* 2022-May-14  YATO, init version
* 2020-Mar-06, dev version, sentence classification, framework change, model saved in one file. 
* 2018-Dec-17, **YATO** v0.2, support PyTorch 1.0
* 2018-Mar-30, **YATO** v0.1, initial version
* 2018-Jan-06, add result comparison.
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version