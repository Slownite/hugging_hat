import click

from .config import HatConfig, hat_config_from_yaml


@click.group()
def cli():
    """Hugging Hat CLI."""
    pass

@cli.command()
def hello():
    """Say hello."""
    click.echo("Hello from hugging-hat!")

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

if __name__ == "__main__":
    cli()
