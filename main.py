import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# TODO: You can rename the module for each of the tasks in the registry as per your method of implementation
REGISTRY = {
    "task1_rnn": {
        "module": "src.task1.rnn_train",
        "default_config": os.path.join("config", "task1", "rnn.yaml"),
    },
    "task1_lstm": {
        "module": "src.task1.lstm_train",
        "default_config": os.path.join("config", "task1", "lstm.yaml"),
    },
    "task2_bilstm": {
        "module": "src.task2.bilstm_train",
        "default_config": os.path.join("config", "task2", "bilstm.yaml"),
    },
    "task2_ssm": {
        "module": "src.task2.ssm_train",
        "default_config": os.path.join("config", "task2", "ssm.yaml"),
    },
    "task3_bilstm": {
        "module": "src.task3.pipeline",
        "default_config": os.path.join("config", "task3", "bilstm.yaml"),
    },
    "task3_ssm": {
        "module": "src.task3.pipeline",
        "default_config": os.path.join("config", "task3", "ssm.yaml"),
    },
}

# DO NOT TOUCH THIS FUNCTION, ADHERE TO THE INSTRUCTIONS IN "Instructions.md"
def main():
    parser = argparse.ArgumentParser(description="INLP Assignment 3")
    subparsers = parser.add_subparsers(dest="command")

    for name, info in REGISTRY.items():
        sub = subparsers.add_parser(name, help=f"Run {name}")
        sub.add_argument("--config", type=str, default=info["default_config"], help="Path to config file")
        sub.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="evaluate", help="Run mode")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    info = REGISTRY[args.command]
    module = __import__(info["module"], fromlist=["main"])
    module.main(config_path=args.config, mode=args.mode)

if __name__ == "__main__":
    main()
