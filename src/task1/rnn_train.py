import yaml
from src.task1.trainer import run_task

def main(config_path, mode):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_task("rnn", cfg, mode)