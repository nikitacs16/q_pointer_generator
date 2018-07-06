export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py27
cd /storage/home/nikita
#java HelloWorld &> outjava
# Change to the directory in which your code is present
cd /storage/home/nikita/d_pointer_generator
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
#touch /storage/home/sidarora/stanford-corenlp-full-2018-02-27/temp12345 &> out1
#ls -l /storage/home/sidarora/stanford-corenlp-full-2018-02-27 &> out2
python -u small_run_this.py &> out
