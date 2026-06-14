[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tTL6Bg44)

# Assignment3_boilerPlate

Deadline: 27 March 2026

## Available Commands

Here is a comprehensive list of all possible commands you can run for each task in this codebase.

### Task 1: Decryption (Character-Level)

#### RNN Model

- **Train:** `uv run main.py task1_rnn --mode train`
- **Evaluate (default):** `uv run main.py task1_rnn --mode evaluate`
- **Train & Evaluate:** `uv run main.py task1_rnn --mode both`

#### LSTM Model

- **Train:** `uv run main.py task1_lstm --mode train`
- **Evaluate (default):** `uv run main.py task1_lstm --mode evaluate`
- **Train & Evaluate:** `uv run main.py task1_lstm --mode both`

### Task 2: Language Modeling (Word-Level)

#### Bi-LSTM Model (MLM)

- **Train:** `uv run main.py task2_bilstm --mode train`
- **Evaluate (default):** `uv run main.py task2_bilstm --mode evaluate`
- **Train & Evaluate:** `uv run main.py task2_bilstm --mode both`

#### SSM Model (NWP/Causal LM)

- **Train:** `uv run main.py task2_ssm --mode train`
- **Evaluate (default):** `uv run main.py task2_ssm --mode evaluate`
- **Train & Evaluate:** `uv run main.py task2_ssm --mode both`

### Task 3: Error Correction Pipeline

#### Pipeline using Bi-LSTM LM

- **Complete:** `uv run main.py task3_bilstm`

#### Pipeline using SSM LM

- **Complete:** `uv run main.py task3_ssm`
