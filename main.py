import click
import sys
import os
from trading_model.train import cli as train_cli

@click.group()
def main_cli():
    """Trading Model CLI"""
    pass

# We can directly use the CLI from train.py which now contains all subcommands
main_cli = train_cli

if __name__ == "__main__":
    main_cli()
