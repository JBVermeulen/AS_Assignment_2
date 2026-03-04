# README File

Created by: EPA133a Group 11

|    Name     | Student Number |
| :---------: | :------------- |
| Annette Dorresteijn | 5868629 |
| Evi de Kok | 5878179 |
| Jonathan Vermeulen | 5144434 |
| Scipio Bruijn  | 5868181 |
| Stijn Keukens | 5072700 |

## Introduction

Every project should have a README file to help a first-time user understand what it is about and how they might be able to use it. This file is where you (as a group) shall provide the information needed by the TAs to evaluate and grade your work.

If you are looking for information about the Demo model of Assignment 2, navigate to the [model/README.md](model/README.md) in the [model](model) directory. Have **fun** modeling in Python!

## How to Use

### Project Preparation
1. Create and activate a virtual environment (`conda` or `venv`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running
Run the model from `model/model_run.py` (from the project root):

```bash
python model/model_run.py
```

- Set `SINGLE_RUN = True` to run one simulation.
- Output is printed saved to `experiment/model_results.csv`.
- Set `SINGLE_RUN = False` to run the full scenario analysis (Scenarios 0-8, 10 replications each).
- Output is saved to `experiment/scenario0.csv` through `scenario8.csv`.

To visualize the model, run `model/model_viz.py`.
This runs a single simulation with adjustable bridge breakdown probabilities.

### Format

Most README files for data or software projects are now written in Markdown format, like this document. There are some different flavours, but they are easy to write. See here for more information https://www.markdownguide.org/basic-syntax

Most IDEs can render Markdown files directly.
