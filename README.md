# MNIST Conditioned GAN

## Project summary 

## Example results

## Features

## Setup (Windows)

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the project's `.venv` interpreter with **Python: Select Interpreter**.

Run training with:

```powershell
python train.py
```

MNIST is downloaded automatically to `data/` the first time the training script runs.

## Generate a class-cycle GIF

After training has saved `weights/generator_cgan.pth`, generate a 0-to-9 class
cycle with:

```powershell
python -m visualisations.class_cycle --seed 4 --frames 20
```

The GIF is saved to `outputs/class_cycle/class_cycle.gif` by default. Use
`--weights_path` and `--output_dir` to choose different paths.

## Animate training progress

Training saves a sample grid named `epoch_NNN.png` for each epoch in
`outputs/`. After training, combine those images into a looping GIF with:

```powershell
python -m visualisations.training_progress
```

The GIF is saved to `outputs/training_progress.gif` by default. Use
`--input_dir`, `--output_path`, or `--duration` to customize the input folder, output path, or delay per epoch frame.

## Visualise class-label embeddings

After training, create both a static 2D PCA plot and an interactive 3D HTML
plot with:

```powershell
python -m visualisations.class_similarity
```

By default, the plots are saved under `outputs/class_similarity/`. Open
`embeddings_pca_3d.html` in a browser and drag to rotate the 3D view. Use
`--weights_path` and `--output_dir` to choose different paths.

## Notes