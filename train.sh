DATA=./data
TRAIN_LANG="" # pivoting language
FORM="" # .ipa or blank
ALIAS_FILE=${DATA}/alias/${ALIAS_FILE}${FORM}

# charagram
python main.py \
            --model charagram \
            --is_train 1\
            --trg_encoding_num 1\
            --embed_size 300 \
            --hidden_size 300 \
            --n_gram_threshold 0 \
            --alia_file ${ALIAS_FILE} \
            --train_file ${DATA}/data/ee-me_train_en-${TRAIN_LANG}_links${FORM}\
            --dev_file ${DATA}/data/me_val_en-${TRAIN_LANG}_links${FORM}\
            --map_file ${DATA}/c2i_map/en-${TRAIN_LANG}_charagram \
            --model_path ${DATA}/models/en-${TRAIN_LANG}_charagram \
            --similarity_measure "cosine" \
            --objective "hinge"\
            --learning_rate 1e-3 \
            --trainer "adam" \

# charcnn
python main.py \
            --model charcnn \
            --is_train 1\
            --trg_encoding_num 1\
            --embed_size 1024 \
            --hidden_size 4800 \
            --n_gram_threshold 0 \
            --alia_file ${ALIAS_FILE} \
            --train_file ${DATA}/data/ee-me_train_en-${TRAIN_LANG}_links${FORM}\
            --dev_file ${DATA}/data/me_val_en-${TRAIN_LANG}_links${FORM}\
            --map_file ${DATA}/c2i_map/en-${TRAIN_LANG}_charagram \
            --model_path ${DATA}/models/en-${TRAIN_LANG}_charagram \
            --similarity_measure "cosine" \
            --objective "hinge"\
            --learning_rate 1e-3 \
            --trainer "adam" \
            --pooling "sum"

# lstm or avglstm
TYPE="lstm" # or "avglstm"
python main.py \
            --model ${TYPE} \
            --is_train 1\
            --trg_encoding_num 1\
            --embed_size 64 \
            --hidden_size 1024 \
            --n_gram_threshold 0 \
            --alia_file ${ALIAS_FILE} \
            --train_file ${DATA}/data/ee-me_train_en-${TRAIN_LANG}_links${FORM}\
            --dev_file ${DATA}/data/me_val_en-${TRAIN_LANG}_links${FORM}\
            --map_file ${DATA}/c2i_map/en-${TRAIN_LANG}_charagram \
            --model_path ${DATA}/models/en-${TRAIN_LANG}_charagram \
            --similarity_measure "cosine" \
            --objective "hinge"\
            --learning_rate 1e-3 \
            --trainer "adam" \


