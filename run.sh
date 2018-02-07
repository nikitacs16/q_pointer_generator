export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py27
# Change to the directory in which your code is present
cd /storage/home/nikita/d_pointer_generator
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
python -u run_this.py &> out
