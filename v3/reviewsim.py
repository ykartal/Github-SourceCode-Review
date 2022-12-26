import click
import pandas as pd
from libs.feature_extraction.vectorizers import vectorizers


@click.group()
def cli():
    """ReviewSim: A tool for automated code review"""
    pass

@cli.command()
@click.option('--type', '-t', type=click.Choice(vectorizers.keys()), help="Vectorizer to use")
@click.option('--model', '-m', type=str, help="Path to model")
@click.option('--data', '-d', type=str, help="Path to data")
@click.option('--save', '-s', type=str, help="Path to save vectors")
def vectorize(type, model, data, save):
    vectorizer = vectorizers[type](model)
    df = pd.read_csv(data)
    codes = df["code"]
    vectors = vectorizer.transform(codes)
    vectorizer.save_vectors(vectors, f"{save}/{type}")

@cli.command()
def review(type, history, code):
    vectorizer = vectorizers[type]()
    df

if __name__ == '__main__':
   cli()