# Project Idea

This project compares two IR approaches for claim verification.
We use the [ClimateFever](https://huggingface.co/datasets/tdiggelm/climate_fever) dataset, which contains claims and evidences that support or refute a claim. 

The `data/` folder contains raw and processed data:
- `raw/` The initial dataset in csv and json form 
- `processed/` Separated evidences, claims and the mapping between the claims and the evidences
- `embeddings/` Claim- and Evidence embeddings for retrieval in Approach 2

The `src/` folder contains the python file and jupyter notebooks.
- `a1/` Files for approach 1. The `pipeline.ipynb` can be used to start the pipelin
- `a2/` Files for approach 2. The `pipeline.ipynb` can be used to start the pipelin
- `data/` Files for preprocessing the data. This includes cleaning the dataset and creating the files in `data/processed/
- `eval/` Files for the evaluation of both approaches and their comparison
- `config.py` general settings.

# Setup to start experiments

Download the necessary models:

`cd src/a1 && python3 bart_large_pull`

`cd src/a2 && python3 climatebert_pull`

Start pipelines for each approach (pipeline.ipynb).

# UI for claim verfication

Run the demo to get a interactive Gradio Based interface.

```bash
python demo.py
```

Access the interface at:

```bash
http://127.0.0.1:7860
```
