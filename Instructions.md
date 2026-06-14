# Introduction to NLP: Assignment 3

We have provided you with the environment for the assignment to ensure uniformity. Ensure you have `uv` installed. Then run `uv sync` to create the environment.

Your code should be run from the `main.py` file using the task subcommands:

```bash
uv run main.py <task> [--config CONFIG_PATH] [--mode {train,evaluate,both}]
```

Available tasks:

| Subcommand     | Description                              | Default Config              |
|----------------|------------------------------------------|-----------------------------|
| `task1_rnn`    | Train/evaluate RNN for decryption        | `config/task1/rnn.yaml`     |
| `task1_lstm`   | Train/evaluate LSTM for decryption       | `config/task1/lstm.yaml`    |
| `task2_bilstm` | Train/evaluate Bi-LSTM for MLM           | `config/task2/bilstm.yaml`  |
| `task2_ssm`    | Train/evaluate SSM for NWP               | `config/task2/ssm.yaml`     |
| `task3_bilstm` | Error correction pipeline using bilstm   | `config/task3/bilstm.yaml`  |
| `task3_ssm`    | Error correction pipeline using ssm      | `config/task3/ssm.yaml`     |

Available modes:

| Mode       | Description                               |
|------------|-------------------------------------------|
| `train`    | Train the model                           |
| `evaluate` | Run inference on trained model (default)  |
| `both`     | Train and then evaluate                   |

Examples:

```bash
uv run main.py task1_rnn
uv run main.py task1_rnn --mode train
uv run main.py task1_rnn --mode evaluate
uv run main.py task1_lstm --mode both
uv run main.py task2_ssm --mode evaluate
uv run main.py task3_bilstm
```

Ensure that the code can be run by the above commands to ensure no penalty during our testing. Adhere to the given directory structure.

In each directory, we have kept a `.gitkeep` file as well, that contains the instruction on which files/code is to be housed in it.
