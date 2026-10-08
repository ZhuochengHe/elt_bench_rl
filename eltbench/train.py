"Configure and launch Tinker RL training for ELT-Bench."

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from . import benchmark
from .env import (
    ELTEnvConfig,
    ELTEnvGroupBuilder,
    load_task,
    renderer_for,
    rollout_termination,
)
from .task_selection import select_trainable_tasks
from .workspace import new_run_tag

ROOT = Path(__file__).resolve().parents[1]


PG_CONTAINER_HOST = os.environ.get("ELT_PG_CONTAINER_HOST", "elt-postgres")
PG_CONTAINER_PORT = int(os.environ.get("ELT_PG_CONTAINER_PORT", "5432"))
PG = {
    "host": os.environ.get("ELT_PG_HOST", "localhost"),
    "port": int(os.environ.get("ELT_PG_PORT", "5433")),
    "user": os.environ.get("ELT_PG_USER", "postgres"),
    "password": os.environ.get("ELT_PG_PASSWORD", "testelt"),
}

TARGET_DB = os.environ.get("ELT_WAREHOUSE_DB", "elt_warehouse")

BASE_SCHEMA = os.environ.get("ELT_BASE_SCHEMA", "eltbench")

BOOTSTRAP_DB = os.environ.get("ELT_PG_BOOTSTRAP_DB", "postgres")


def local_destination_config() -> dict:
    return {
        "host": PG_CONTAINER_HOST,
        "port": PG_CONTAINER_PORT,
        "user": PG["user"],
        "password": PG["password"],
        "database": TARGET_DB,
        "schema": BASE_SCHEMA,
        "connect_timeout": int(os.environ.get("ELT_PG_CONNECT_TIMEOUT", "15")),
        "bootstrap_database": BOOTSTRAP_DB,
    }


def local_harness_config() -> dict:
    return {
        "host": PG["host"],
        "port": PG["port"],
        "connect_timeout": int(os.environ.get("ELT_PG_CONNECT_TIMEOUT", "15")),
    }


def list_trainable_tasks(
    max_models: int = 1, destination: str = benchmark.LOCAL_DESTINATION
) -> list[str]:
    spec_root = benchmark.benchmark_dir(destination)
    gt_root = benchmark.GT_ROOT
    tasks = []
    for gt_dir in sorted(gt_root.iterdir()):
        if not gt_dir.is_dir() or not (spec_root / gt_dir.name).is_dir():
            continue
        n_models = len(list(gt_dir.glob("*.csv")))
        if 0 < n_models <= max_models:
            tasks.append(gt_dir.name)
    return tasks


# ------------------------------------------------------------------ Dataset
class ELTRLDataset(RLDataset):
    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[ELTEnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int = 4
    max_turns: int = 50
    max_trajectory_tokens: int = 30000
    max_tool_calls: int = 60

    rollout_timeout_seconds: float | None = 3600.0
    renderer_name: str | None = None
    max_models: int = 1
    max_tasks: int = 0
    test_frac: float = 0.0
    task_name: str | None = None

    eval_batch_size: int = 0
    seed: int = 0
    destination: str = benchmark.LOCAL_DESTINATION

    check_loading: bool = False

    load_mode: str = "auto"
    keep_namespaces: bool = False
    runs_root: str | None = None
    staging_root: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if self.keep_namespaces:
            os.environ["ELT_KEEP_NAMESPACES"] = "1"
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)

        renderer = renderer_for(
            self.model_name_for_tokenizer, tokenizer, self.renderer_name
        )
        names = select_trainable_tasks(
            list_trainable_tasks(
                max_models=self.max_models, destination=self.destination
            ),
            task_name=self.task_name,
            seed=self.seed,
            max_tasks=self.max_tasks,
        )
        if not names:
            raise ValueError(
                "No trainable tasks found; check repo/elt-bench and repo/evaluation/gt."
            )

        n_test = int(len(names) * self.test_frac) if self.test_frac > 0 else 0
        test_names, train_names = names[:n_test], names[n_test:]
        if not train_names:
            raise ValueError(
                "test_frac assigned every task to the test split; reduce it."
            )

        run_tag = new_run_tag()
        train_ds = ELTRLDataset(
            [
                self._build(n, tokenizer, renderer, self.group_size, run_tag)
                for n in train_names
            ],
            self.batch_size,
        )
        test_ds = None
        if test_names:
            eval_bs = self.eval_batch_size or min(len(test_names), 2)
            test_ds = ELTRLDataset(
                [self._build(n, tokenizer, renderer, 1, run_tag) for n in test_names],
                max(1, min(eval_bs, len(test_names))),
            )
        return train_ds, test_ds

    def _build(
        self, name: str, tokenizer, renderer, group_size: int, run_tag: str
    ) -> ELTEnvGroupBuilder:
        task = load_task(
            name,
            destination=self.destination,
            load_tables=None if self.check_loading else [],
        )
        cfg = ELTEnvConfig(
            task=task,
            model_name=self.model_name_for_tokenizer,
            destination=self.destination,
            group_size=group_size,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_tool_calls=self.max_tool_calls,
            rollout_timeout_seconds=self.rollout_timeout_seconds,
            renderer_name=self.renderer_name,
            base_destination_config=(
                local_destination_config()
                if self.destination == benchmark.LOCAL_DESTINATION
                else {}
            ),
            harness_destination_config=(
                local_harness_config()
                if self.destination == benchmark.LOCAL_DESTINATION
                else {}
            ),
            credential_path=(
                None
                if self.destination == benchmark.LOCAL_DESTINATION
                else benchmark.credential_path_for(self.destination)
            ),
            load_mode=self.load_mode,
            runs_root=Path(self.runs_root) if self.runs_root else ROOT / "runs",
            staging_root=Path(self.staging_root) if self.staging_root else None,
            run_tag=run_tag,
        )
        return ELTEnvGroupBuilder(cfg, tokenizer, renderer=renderer)


# ------------------------------------------------------------------ CLI
@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None

    learning_rate: float = 4e-5

    batch_size: int = 4
    group_size: int = 4
    max_turns: int = 50
    max_trajectory_tokens: int = 30000
    max_tool_calls: int = 60

    rollout_timeout_seconds: float | None = 3600.0
    max_tokens: int = 4096
    seed: int = 0
    eval_every: int = 0
    save_every: int = 20
    max_steps: int | None = None

    rollout_error_tolerance: bool = True

    destination: str = benchmark.LOCAL_DESTINATION

    check_loading: bool | None = None

    load_mode: str = "auto"

    kl_penalty_coef: float = 0.0
    kl_reference_model: str | None = None
    max_models: int = 1
    max_tasks: int = 0
    task_name: str | None = None
    test_frac: float = 0.0

    eval_batch_size: int = 0

    remove_constant_reward_groups: bool = True
    keep_namespaces: bool = False
    runs_root: str | None = None
    staging_root: str | None = None

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def resolve_check_loading(cli_config: CLIConfig) -> bool:
    if cli_config.check_loading is not None:
        return cli_config.check_loading
    return cli_config.destination != benchmark.LOCAL_DESTINATION


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )

    check_loading = resolve_check_loading(cli_config)
    if cli_config.keep_namespaces:
        os.environ["ELT_KEEP_NAMESPACES"] = "1"

    builder = ELTDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        max_turns=cli_config.max_turns,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_tool_calls=cli_config.max_tool_calls,
        rollout_timeout_seconds=cli_config.rollout_timeout_seconds,
        renderer_name=renderer_name,
        max_models=cli_config.max_models,
        max_tasks=cli_config.max_tasks,
        task_name=cli_config.task_name,
        test_frac=cli_config.test_frac,
        eval_batch_size=cli_config.eval_batch_size,
        seed=cli_config.seed,
        destination=cli_config.destination,
        check_loading=check_loading,
        load_mode=cli_config.load_mode,
        keep_namespaces=cli_config.keep_namespaces,
        runs_root=cli_config.runs_root,
        staging_root=cli_config.staging_root,
    )

    model_short = cli_config.model_name.lower().replace("/", "-")
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"elt_{model_short}_{cli_config.destination}_bs{cli_config.batch_size}"
        f"_gs{cli_config.group_size}_seed{cli_config.seed}"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{stamp}"
    )
    log_path = cli_config.log_path or str(ROOT / "runs" / "train" / run_name)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )

    kl_reference_config = None
    if cli_config.kl_penalty_coef > 0:
        kl_reference_config = train.KLReferenceConfig(
            base_model=cli_config.kl_reference_model or cli_config.model_name
        )

    config = train.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_elt_bench",
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        lora_rank=cli_config.lora_rank,
        max_steps=cli_config.max_steps,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_reference_config=kl_reference_config,
        termination=rollout_termination(),
        rollout_error_tolerance=cli_config.rollout_error_tolerance,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
    )
    await train.main(config)


if __name__ == "__main__":
    _cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(_cfg))
