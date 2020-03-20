# PBEL+ Running

## Training
### Generate training data
To generate the training data, we need 
* download Wikipedia data from the Wikipedia dump
* extract bilingual entity map
* extract bilingual mention entity map
* convert to IPA if necessary
* convert to my model running format

An training data example is in:
> /projects/tir2/users/shuyanzh/lorelei_data/pbel/data/ee_test_en-ti_links{.ipa}

The code for this pipeline is here:
> /home/shuyanzh/workshop/lor_edl/train_pbel.sh

It has comment for each step, might be a little bit messy

### Run PBEL training 
```bash
final_train_file=""
suffix="" # ipa or leave blank
final_dev_file=""
save_path="" # the folder you will use to store data, it will contains the map and the model
lang="" # I often use two character code for the language
file_line="" # "_ipa" or "_graph"
python /home/shuyanzh/workshop/pbel/model_charagram.py \
                --is_train 1\
                --mega 0\
                --use_mid 0\
                --share_vocab 0\
                --trg_encoding_num 1\
                --trg_auto_encoding 0\
                --position_embedding 0\
                --sin_embedding 0 \
                --embed_size 300 \
                --n_gram_threshold 0 \
                --train_file $final_train_file${suffix}\
                --dev_file $final_dev_file${suffix}\
                --map_file "${save_path}/c2i_maps/${lang}-char-cosine-hinge${file_line}"\
                --model_path "${save_path}/models/${lang}-char-cosine-hinge${file_line}" \
                --similarity_measure "cosine" \
                --objective "hinge"\
                --learning_rate 0.1 \
                --trainer "sgd"

```

## Test
* After you get the ner file, you could use 
> /home/shuyanzh/workshop/pbel/preprocessing/run_st.sh

The argument is the number of IL, e.g. 5, 6, 9, 10. This code will generate PBEL style test data. 

* After getting the pbel style test data, you could run the model, an example script is here:
> /home/shuyanzh/workshop/pbel/sh_script/test_st9-tl_ipa.sh

If the test data is too large, it will throw OOM error, so I split the data to 3000 lines per file. In the test script, L49-L50 will iterate all files

