# Dataset Toolkit

Choose a CSV file and get a measurement summary, a two-dimensional chart, and
a CSV containing the chart coordinates. No notebook editing is required.

## Setup

Run these commands once from the project folder (Python 3.10 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Analyse a dataset

Put CSV files with a header row in `data`, then run:

```bash
python main.py
```

Enter the number beside a dataset, then press Enter. Enter `Q` to quit.
Open the chart printed at the end of the analysis. Results are saved in
`results/<dataset>_pca.png` and `results/<dataset>_pca.csv`; running the same
dataset again replaces its previous results. Paths are relative to the toolkit,
so the application also works when started from another folder.

## What the analysis means

PCA combines numeric measurements into two coordinates per row. The reported
percentage tells you how much of the original variation the chart retains;
a lower percentage means more information is lost in the two-dimensional view.
Nearby points have similar measurements, but the chart does not prove that
groups are meaningful or predict outcomes.

The application uses numeric columns with varying values, excluding the
optional `target` group label and unnamed CSV index columns. Missing or
infinite numeric values are replaced with their column median. Text columns,
constant columns, and entirely empty columns are not measurements. At least
two rows and two usable measurements are required. Remove numeric IDs or other
labels before analysis if they should not influence the chart.

Like the original notebook, measurements are not scaled before PCA. Columns
with larger units or variation can dominate the chart; compare compatible
measurements or prepare scaled data when that matters. If `target` exists,
it is preserved in the output and used to colour the chart's groups.

The notebook remains an interactive walkthrough of the Iris example.

## Check the toolkit

```bash
python -m unittest discover -s tests -v
```