# Glass Breaking Simulator

Exactly what it says on the tin. Select from 11 models to break, set parameters
for fracturing and movement, then click anywhere to explode.

[Demo Video](https://youtu.be/AIpXEkMYnbU)

## Install requirements

```commandline
conda install -c conda-forge pyvista numpy
```

```commandline
pip install pygame
```

You may also need to install `vtk` separately before installing `pyvista` (especially on M1 Macs),
which (for Mac) can be done via Homebrew.

## Run
```commandline
python main.py
```