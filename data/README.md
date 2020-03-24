# Data

## Download
Please download the full data from this [link](https://drive.google.com/file/d/1Uh80mAWeuczNwC-bDbSnpbCSAofIAsii/view?usp=sharing)

Note that for the test data, we only release the data we crawl from Wikipedia.
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
