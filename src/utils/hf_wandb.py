import os

import torch
import wandb
from huggingface_hub import HfApi, hf_hub_download


def init_wandb(project: str, config: dict, name: str | None = None) -> wandb.sdk.wandb_run.Run:
    return wandb.init(project=project, config=config, name=name)


def log_wandb(metrics: dict, step: int | None = None) -> None:
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    wandb.finish()


def push_to_hub(
    path: str,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo or os.path.basename(path),
        repo_id=repo_id,
        token=token,
    )


def pull_from_hub(
    repo_id: str,
    filename: str,
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def save_and_push(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    torch.save(model.state_dict(), local_path)
    return push_to_hub(local_path, repo_id, filename, token)


def load_from_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    device: str = "cpu",
    token: str | None = None,
) -> torch.nn.Module:
    path = pull_from_hub(repo_id, filename, local_dir, token)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
