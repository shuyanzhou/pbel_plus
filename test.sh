DATA="./data"
TRAIN_LANG=""
TEST_LANG="" # test language
FORM="" # .ipa or blank
ALIAS_FILE=${DATA}/alias/${ALIAS_FILE}${FORM}
KB_FILE=${DATA}/kb/en_kb${FORM}

python main.py \
              --model charagram \
              --is_train 0 \
              --trg_encoding_num 10 \
              --n_gram_threshold 0 \
              --embed_size 300 \
              --hidden_size 300 \
              --alia_file ${ALIAS_FILE} \
              --method "pivoting"\
              --test_epoch "best" \
              --kb_file ${KB_FILE} \
              --test_file ${DATA}/data/me_test_en-${TEST_LANG}_links${FORM}\
              --no_pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_base.result${FORM} \
              --pivot_file ${DATA}/data/en-${TRAIN_LANG}_links${FORM} \
              --pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_pivot.result${FORM} \
              --batch_size 256 \
              --similarity_measure "cosine"\
              --objective "hinge" \
              --pivot_is_src 1 \
              --pivot_is_mid 0 \

# charcnn
python main.py \
              --model charcnn \
              --is_train 0 \
              --trg_encoding_num 10 \
              --n_gram_threshold 0 \
              --embed_size 1024 \
              --hidden_size 4800 \
              --alia_file ${ALIAS_FILE} \
              --method "pivoting"\
              --test_epoch "best" \
              --kb_file ${KB_FILE} \
              --test_file ${DATA}/data/me_test_en-${TEST_LANG}_links${FORM}\
              --no_pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_base.result${FORM} \
              --pivot_file ${DATA}/data/en-${TRAIN_LANG}_links${FORM} \
              --pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_pivot.result${FORM} \
              --batch_size 256 \
              --similarity_measure "cosine"\
              --objective "hinge" \
              --pivot_is_src 1 \
              --pivot_is_mid 0 \
              --pooling "sum"

# lstm
TYPE="lstm"
python main.py \
              --model ${TYPE} \
              --is_train 0 \
              --trg_encoding_num 10 \
              --n_gram_threshold 0 \
              --embed_size 64 \
              --hidden_size 1024 \
              --alia_file ${ALIAS_FILE} \
              --method "pivoting"\
              --test_epoch "best" \
              --kb_file ${KB_FILE} \
              --test_file ${DATA}/data/me_test_en-${TEST_LANG}_links${FORM}\
              --no_pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_base.result${FORM} \
              --pivot_file ${DATA}/data/en-${TRAIN_LANG}_links${FORM} \
              --pivot_result ${DATA}/results/en-${TRAIN_LANG}_${TEST_LANG}_pivot.result${FORM} \
              --batch_size 256 \
              --similarity_measure "cosine"\
              --objective "hinge" \
              --pivot_is_src 1 \
              --pivot_is_mid 0