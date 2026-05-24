import click

from .config import HatConfig, hat_config_from_yaml


@click.group()
def cli():
    """Hugging Hat CLI."""
    pass

@cli.command()
@click.argument("model_name_or_path", type=str)
@click.option("--config-yaml", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--local-files-only/--download", default=True, show_default=True)
def doctor(model_name_or_path: str, config_yaml: str | None, local_files_only: bool) -> None:
    """Inspect a Hugging Face causal LM for hat attachment compatibility."""
    try:
        from .model import HatEnabledModel
    except ModuleNotFoundError as e:
        raise click.ClickException(str(e)) from e

    config = None
    if config_yaml is not None:
        with open(config_yaml, "r", encoding="utf-8") as f:
            config = hat_config_from_yaml(f.read())
    else:
        config = HatConfig()

    try:
        model = HatEnabledModel.from_pretrained(model_name_or_path, config=config, local_files_only=local_files_only)
    except ModuleNotFoundError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"layers_path: {model.layers_path}")
    click.echo(f"hidden_size: {model.hidden_size}")
    click.echo(f"num_layers: {model.num_layers}")
    click.echo(f"router_layer_idx: {model.router_layer_idx}")
    click.echo(f"thinker_layer_idx: {model.thinker_layer_idx}")
    click.echo(f"critic_layer_idx: {model.critic_layer_idx}")


@cli.command(name="train-thinker")
@click.argument("model_name_or_path", type=str)
@click.option("--data", "data_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--dataset", type=str, default=None)
@click.option("--split", type=str, default=None)
@click.option("--dataset-config", type=str, default=None)
@click.option("--prompt-field", type=str, default="prompt", show_default=True)
@click.option("--completion-field", type=str, default="completion", show_default=True)
@click.option("--output-dir", "output_dir", type=click.Path(file_okay=False), required=True)
@click.option("--thinker-steps", type=int, default=4, show_default=True)
@click.option("--max-length", type=int, default=1024, show_default=True)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--epochs", "num_epochs", type=int, default=1, show_default=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--grad-accum", "grad_accum_steps", type=int, default=1, show_default=True)
@click.option("--precision", type=click.Choice(["fp32", "fp16"]), default="fp32", show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--log-every", type=int, default=10, show_default=True)
@click.option("--save-every", type=int, default=None)
@click.option("--config-yaml", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--local-files-only/--download", default=True, show_default=True)
@click.option("--resume-from", type=click.Path(exists=True, file_okay=False), default=None)
def train_thinker_cmd(
    model_name_or_path: str,
    data_path: str | None,
    dataset: str | None,
    split: str | None,
    dataset_config: str | None,
    prompt_field: str,
    completion_field: str,
    output_dir: str,
    thinker_steps: int,
    max_length: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    max_steps: int | None,
    grad_accum_steps: int,
    precision: str,
    device: str | None,
    seed: int | None,
    log_every: int,
    save_every: int | None,
    config_yaml: str | None,
    local_files_only: bool,
    resume_from: str | None,
) -> None:
    """Train the Thinker Hat end-to-end from a JSONL file or HF dataset."""
    if bool(data_path) == bool(dataset):
        raise click.UsageError(
            "exactly one of --data or --dataset is required"
        )
    if dataset and not split:
        raise click.UsageError("--split is required when using --dataset")

    try:
        from .model import HatEnabledModel
    except ModuleNotFoundError as e:
        raise click.ClickException(str(e)) from e
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as e:
        raise click.ClickException(str(e)) from e

    from .data import load_hf_dataset, load_jsonl
    from .train import TrainConfig, train_thinker

    if config_yaml is not None:
        with open(config_yaml, "r", encoding="utf-8") as f:
            hat_config = hat_config_from_yaml(f.read())
    else:
        hat_config = HatConfig()

    try:
        model = HatEnabledModel.from_pretrained(
            model_name_or_path, config=hat_config, local_files_only=local_files_only
        )
    except ModuleNotFoundError as e:
        raise click.ClickException(str(e)) from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )

    click.echo(f"layers_path: {model.layers_path}")
    click.echo(f"thinker_layer_idx: {model.thinker_layer_idx}")
    click.echo(f"router_layer_idx: {model.router_layer_idx}")
    click.echo(f"critic_layer_idx: {model.critic_layer_idx}")
    click.echo(f"thinker_steps: {thinker_steps}")
    click.echo(f"step_set: {list(model.config.thinker.step_set)}")
    click.echo(f"device: {device if device is not None else 'auto'}")
    click.echo(f"precision: {precision}")
    click.echo(f"batch_size: {batch_size}")
    click.echo(f"max_length: {max_length}")

    train_config = TrainConfig(
        max_length=max_length,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum_steps,
        thinker_steps=thinker_steps,
        device=device,
        precision=precision,  # type: ignore[arg-type]
        seed=seed,
        log_every=log_every,
        save_every=save_every,
        resume_from=resume_from,
    )

    if dataset:
        records = list(
            load_hf_dataset(
                dataset,
                split=split,
                config=dataset_config,
                prompt_field=prompt_field,
                completion_field=completion_field,
            )
        )
    else:
        records = list(load_jsonl(data_path))
    train_thinker(model, records, tokenizer, train_config, output_dir=output_dir)


if __name__ == "__main__":
    cli()
