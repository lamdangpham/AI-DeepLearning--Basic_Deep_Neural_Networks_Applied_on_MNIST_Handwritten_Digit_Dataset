# Basic Deep Neural Networks Applied on MNIST Handwritting Dataset

1/ Hierarchy:
 + 01_dnn_minist	: MPL architecture
     + network : Learning model description
     + step02_training.py : Training & Testing Process
     + run.sh : Bash script to call 'step02_training.py'
 ---> Other architectures show similar files

 + 02_cnn_mnist:         CNN architecture	
 + 03_cnn_mixup_minist:  CNN architecture with mixup data augmentation
 + 04_rnn_mnist:         RNN architecture (Bidirectional RNN with GRU cell)
 + 05_crnn_mnist:        CRNN architecture
 + 06_cnn_rnn_parallell: Parallel CNN & RNN architecture
 + README.md
 
2/ Notes:
 + All networks base on tensorflow framework 1.X
 + Dataset : MNIST
 
3/ conda env setting:
conda create --name test python=2.7
conda activate test
pip install tensorflow-cpu==1.15
pip install -U scikit-learn
