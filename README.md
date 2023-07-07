# Automating Modern Code Review Processes with Code Similarity Measurement

This repository contains research code for the paper ["Automating Modern Code Review Processes with Code Similarity Measurement"](http://dx.doi.org/10.2139/ssrn.4450324). Our research aims to automating code review processes by measuring code similarity.

Data and models available at [Mega.nz](https://mega.nz/folder/kv4GDDJa#pcgag7752nVLPumSilU_yg).

## Architecture

![Architecture](public/architecture.png)

## Auxiliary Libraries

In `libs` directory, you can find auxiliary libraries that we used in our experiments.

1. **Vectorizers:** TF-IDF, Bag-of-Words, Word2Vec, Doc2Vec, Transformers
2. **Metrics:** Text similarity, vector distance.

## Evaluation

`experiment.py` contains necessary codes to reproduce the experiments in the paper.
You can find example usage in `experiment_runner.ipynb`.

## Example

![Example](public/example.png)

## Results

![Vectorizer Comparison](public/results_chart.png)
![Model Comparison](public/results_table.png)

## Citation

If you use this repository for your work, please don't forget to give proper credit by including the accompanying [paper](https://dx.doi.org/10.2139/ssrn.4450324) in your references.

    @article{kartal2023,
      title={Automating Modern Code Review Processes with Code Similarity Measurement},
      author={Kartal, Yusuf and Akdeniz, Kaan and Özkan, Kemal},
      journal={Information and Software Technology},
      year={2023}
    }
