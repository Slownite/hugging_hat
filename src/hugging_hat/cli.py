import click

@click.group()
def cli():
    """Hugging Hat CLI."""
    pass

@cli.command()
def hello():
    """Say hello."""
    click.echo("Hello from hugging-hat!")

if __name__ == "__main__":
    cli()
