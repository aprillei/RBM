#!/bin/bash

LEARNED_FEATURES="../../learned_features"
SCRIPTS="scripts"
MATLAB_EXEC="matlab"
DOWNLOAD_SCRIPT=$SCRIPTS/"downloadData.py"
DATASET="dataset/20Newsgroups.mat"
PROCESS_20NEWS_SCRIPT="process_20News"
PICKLE_DATA_SCRIPT=$SCRIPTS/"pklInput.py"
DL_LOAD_DATA_SCRIPT=$SCRIPTS/"load_data.py"
DL_TRAIN_MODEL_SCRIPT=$SCRIPTS/"rbm.py"
LIBSVM_EASY="easy.py"
RESULTS="../../results/"

NUM_FEATURES=1000
NUM_HIDDEN_FEATURES=100
NUM_TRAINING_EXAMPLES=1064 #2245
NUM_TEST_EXAMPLES=708 #1494
LEARNING_RATE='0.1'
NUM_ITERATIONS_RBM=50
NUM_ITERATIONS_DL=20
K=15
BATCH_SIZE=20
PARAS="parameters/paras.para"

TRAIN="dataset/train"
TEST="dataset/test"
TRAIN_FEATURES=$TRAIN".fea"
TRAIN_LABELS=$TRAIN".labels"
TEST_FEATURES=$TEST".fea"
TEST_LABELS=$TEST".labels"

RAW_TRAIN="train.svm"
RAW_TEST="test.svm"

DL_TRAIN="learned_features_dl_implementation_train"
DL_TEST="learned_features_dl_implementation_test"

LEARNED_TRAIN="learned_features_train"
LEARNED_TEST="learned_features_test"

LIBSVM_DIRECTORY="libsvm-3.20"
LIBSVM_TOOLS="tools"

RAW_RESULTS_FILE=$RESULTS"raw_results"
DL_RESULTS_FILE=$RESULTS"dl_results"
RBM_RESULTS_FILE=$RESULTS"rbm_results"

mkdir dataset
mkdir learned_features
mkdir results
mkdir parameters

# Download dataset
python $DOWNLOAD_SCRIPT $DATASET

# Process the dataset so as to keep the NUM_FEATURES, NUM_TRAINING_EXAMPLES and NUM_TEST_EXAMPLES
# and write them to text file
echo Keeping $NUM_TRAINING_EXAMPLES training examples, and $NUM_TEST_EXAMPLES test examples, each with $NUM_FEATURES features
cd $SCRIPTS
$MATLAB_EXEC -nojvm -nodisplay -nosplash -r "$PROCESS_20NEWS_SCRIPT($NUM_FEATURES,$NUM_TRAINING_EXAMPLES,$NUM_TEST_EXAMPLES);exit"
cd ../

# Format the dataset so that it can be used by the deeplearning.net implementation
echo Formatting dataset so that it can be read by the deeplearning.net RBM implementation
python $PICKLE_DATA_SCRIPT $TRAIN_FEATURES $TRAIN_LABELS $TEST_FEATURES $TEST_LABELS

# Train RBM using the http://deeplearning.net/tutorial/rbm.html implementation
#echo Training the deeplearning.net RBM
python $DL_TRAIN_MODEL_SCRIPT $NUM_FEATURES $NUM_HIDDEN_FEATURES $LEARNING_RATE $NUM_ITERATIONS_DL $BATCH_SIZE $K

# Train RBM, format raw input and RBM models into libsvm format
make

# Write parameters to parameter file
rm $PARAS
touch $PARAS
echo $NUM_FEATURES >> $PARAS
echo $NUM_HIDDEN_FEATURES >> $PARAS
echo $K >> $PARAS
echo $NUM_ITERATIONS_RBM >> $PARAS
echo $LEARNING_RATE >> $PARAS
echo $BATCH_SIZE >> $PARAS

time ./main $TRAIN $TEST $NUM_TRAINING_EXAMPLES $NUM_TEST_EXAMPLES $NUM_FEATURES $NUM_HIDDEN_FEATURES $LEARNING_RATE $NUM_ITERATIONS_RBM $K $BATCH_SIZE $PARAS

# Call the easy.py script from libsvm
# Edited the libsvm scale script to scale between 0 and 1
cd $LIBSVM_DIRECTORY
make
cd $LIBSVM_TOOLS
echo training svm on raw data
python $LIBSVM_EASY $LEARNED_FEATURES/$RAW_TRAIN $LEARNED_FEATURES/$RAW_TEST > $RAW_RESULTS_FILE &

echo training svm on deeplearning.net trained RBM
python $LIBSVM_EASY $LEARNED_FEATURES/$DL_TRAIN $LEARNED_FEATURES/$DL_TEST > $DL_RESULTS_FILE &

echo training svm on RBM
python $LIBSVM_EASY $LEARNED_FEATURES/$LEARNED_TRAIN $LEARNED_FEATURES/$LEARNED_TEST > $RBM_RESULTS_FILE &
wait
