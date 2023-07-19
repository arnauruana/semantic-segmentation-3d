# Semantic Segmentation 3D

Final project developed for the [UPC-AIDL](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/posgrado-artificial-intelligence-deep-learning/) postgraduate course. The main objective is to create a [deep learning](https://en.wikipedia.org/wiki/Deep_learning) model able to perform [semantic segmentation](https://paperswithcode.com/task/semantic-segmentation) over a given urban-dense [point cloud](https://en.wikipedia.org/wiki/Point_cloud).

<p align="center">
  <img src="https://www.cvlibs.net/datasets/kitti-360/images/example/3d/semantic/02400.jpg" width=800/>
</p>

## Table of Contents

* [Documentation](#documentation)
* [Installation](#installation)
  * [Repository](#repository)
  * [Environment](#environment)
  * [Requirements](#requirements)
  * [Dataset](#dataset)
  * [Models](#models)
* [Usage](#usage)
  * [PointNet](#pointnet)
  * [GraphNet](#graphnet)
* [Uninstalling](#uninstalling)
* [Acknowledgements](#acknowledgements)
* [License](#license)
* [Authors](#authors)

## Documentation

For detailed information about this project, check out the [documentation](./doc/) folder. We strongly recommend reading it.

## Installation

### Repository

First of all, we need to execute these commands in order to install the repository locally.

1.- Download the repository:

```bash
git clone git@github.com:arnauruana/semantic-segmentation-3d.git
```

> Note that the previous command uses the `ssh` protocol. You can also use `https` if you don't have any linked ssh key.

2.- Enter the downloaded folder:

```bash
cd semantic-segmentation-3d
```

### Environment

If you don't want to use `conda` environments you can [skip](#requirements) this section, otherwise you will need to download it manually from [here](https://docs.conda.io/en/latest/miniconda.html).

Once installed, you can run the following commands to set up the execution environment.

1.- Create the environment:

```bash
conda create --name aidl-project python=3.11 --yes
```

2.- Activate the environment:

```bash
conda activate aidl-project
```

3.- Install the `Python` requirements from the [next](#requirements) section.

### Requirements

Unless you followed the previous [instructions](#environment), you must install `Python` (either manually from this [link](https://www.python.org/downloads/) or using your favorite package manager).

```bash
sudo pacman -S python --needed --noconfirm
```

> Please, note that the previous command will only work on Arch-based Linux distributions and, most likely, you will need to adapt it for yours.

In both cases, you need to execute this line to install all the `Python` requirements:

```bash
pip install -r requirements.txt
```

### Dataset

The original dataset, formally called [IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/), is a public dataset containing a point cloud from a $200$ meter street in Paris, gathered using the [LiDAR](https://www.ibm.com/topics/lidar) technology. It contains $12$ million points labeled with their respective classes and identifiers, making it suitable for either [semantic segmentation](https://paperswithcode.com/task/semantic-segmentation) or [instance segmentation](https://paperswithcode.com/task/instance-segmentation) tasks.

The already filtered, normalized and split version of this dataset can be found in the following [link](https://drive.google.com/file/d/1fYo03lGE9E0yDbgx-KzCnDr8_wUdinKU/view?usp=drive_link). This is the actual dataset we have worked with.

Once downloaded and placed it under the root directory of this project, you can decompress it using the following command:

```bash
unzip data.zip
```

> You can also use any program capable of decompressing files. Such as [7-zip](https://www.7-zip.org/), [WinRAR](https://www.rarlab.com/) or [WinZip](https://www.winzip.com/en/) among others.

If you prefer the raw data and split them manually you must download this [file](https://drive.google.com/file/d/1PM99MkXoIozVFTkaWvOb6u-e0SYjpAtH/view?usp=drive_link) instead.

### Models

We also provide two pre-trained models called `pointnet.pt` and `graphnet.pt`. You can download them from this [link](https://drive.google.com/file/d/12ZLOiM_fZqekVBLHnlOXUPlvdOqbTA53/view?usp=sharing).

Once downloaded and placed the file under the root directory of this project, you can decompress it using the following command:

```bash
unzip models.zip
```

## Usage

If you haven't downloaded the already processed and split version of the data and you want to manually do it, use this command:

```bash
python src/split.py
```

Additionally inside the source code, there are two submodules that can be executed independently from each other depending on the model you want to use.

### PointNet

1.- Train the model:

```bash
python src/pointnet/train.py
```

2.- Test the model:

```bash
python src/pointnet/test.py
```

3.- Infer the model:

```bash
python src/pointnet/infer.py
```

### GraphNet

1.- Train the model:

```bash
python src/graphnet/train.py
```

2.- Test the model:

```bash
python src/graphnet/test.py
```

3.- Infer the model:

```bash
python src/graphnet/infer.py
```

## Uninstalling

In order to uninstall the project and all its previously installed dependencies, you can execute these commands:

1.- Uninstall python requirements

```bash
pip uninstall -r requirements.txt
```

2.- Remove conda environment.

```bash
conda deactivate
conda remove --name aidl-project --all --yes
```

3.- Remove repository:

```bash
cd ..
rm -rf semantic-segmentation-3d
```

## License

This project is licensed under the `MIT License`. See the [LICENSE](./LICENSE.md) file for further information about it.

## Acknowledgements

**Dataset**: [IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/)

**Paper**: Bruno Vallet, Mathieu Brédif, Andrés Serna, Beatriz Marcotegui, Nicolas Paparoditis. TerraMobilita/IQmulus urban point cloud analysis benchmark. Computers and Graphics, Elsevier, 2015, Computers and Graphics, 49, pp.126-133. (<https://hal.archives-ouvertes.fr/hal-01167995v1>)

## Authors

Owners:

* [Arnau Ruana](https://github.com/arnauruana)
* [Rafel Palomo](https://github.com/RafelPalomo)
* [Soraya Gonzalez](https://github.com/SorayaGonzalezSanchez)

Supervisors:

* [Mariona Carós](https://github.com/marionacaros)
