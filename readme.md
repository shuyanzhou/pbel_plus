# Improving Candidate Generation for Low-resource Cross-lingual Entity Linking

This is the code repo for TACL 2020 paper [Improving Candidate Generation for Low-resource Cross-lingual Entity Linking](https://arxiv.org/abs/2003.01343)

## Training
Please refer to train.sh for the arguments, all four models (charagram, charcnn, lstm-last and lstm-avg) could be launched in this bash file

## Test
Please refer to test.sh for the arguments, all four models (charagram, charcnn, lstm-last and lstm-avg) could be launched in this bash file

## Data
Data folder contains ``data`` ``alias`` and ``kb``
#### ``data``: all train, val, test data
```
English_Wikipedia_ID ||| English_Wikipedia_title ||| Wikipedia_title_of_train/test_lang ||| Entity_type
e.g. 3378263 ||| John Michael Talbot ||| ጆን ማይክል ታልበት ||| PER
```
#### ``alias``: entity alias from Wikidata
```
Wikidata_ID ||| English_Wikipedia_ID ||| aliases
e.g. Q42 ||| 8091 ||| Douglas Adams ||| Douglas N. Adams || Douglas Noël Adams || Douglas Noel Adams
```

#### ``kb``: all entities in wikipedia (~2M, proper nouns)
```
English_Wikipedia_ID ||| English_Wikipedia_title ||| Entity_type
e.g. 16429160 ||| George Karrys ||| PER
```
Note that all files ended with .ipa are of *phoneme* representations, using [Epitran](https://github.com/dmort27/epitran)
## Reference
```
@article{zhou20tacl,
    title = {Improving Candidate Generation for Low-resource Cross-lingual Entity Linking},
    author = {Shuyan Zhou and Shruti Rijhwani and John Wieting and Jaime Carbonell and Graham Neubig},
    journal = {Transactions of the Association of Computational Linguistics},
    month = {},
    url = {https://arxiv.org/abs/2003.01343},
    year = {2020}
}

```