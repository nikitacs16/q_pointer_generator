import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='config.yaml')
args = parser.parse_args()
os.system('python run_summarization.py --mode=restore_best_model --config_file '+str(args.config_file))
os.system('python run_summarization.py --mode=decode --config_file '+str(args.config_file))

