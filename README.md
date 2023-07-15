# Semantic Segmentation 3D

Final project developed for the [UPC-AIDL](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/posgrado-artificial-intelligence-deep-learning/) postgraduate course.

<p align="center">
  <img *img* src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fpythonawesome.com%2Fcontent%2Fimages%2F2019%2F08%2FLiDAR-Bonnetal.jpg&f=1&nofb=1&ipt=ca2ab3059b21edf04bfd01bfa4b86b9373682fa7b63e4d9e7f5e59c970273874&ipo=images"/>
</p>

## Index

* [Description](#description)
* [Installation](#installation)
  * [Repository](#repository)
  * [Environment](#environment)
  * [Requirements](#requirements)
  * [Dataset](#dataset)
* [Uninstalling](#uninstalling)
* [License](#license)
* [Authors](#authors)

## Description

The objective of this project is to create a [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) model able to perform [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation) over a given [point cloud](https://en.wikipedia.org/wiki/Point_cloud).

## Installation

### Repository

First of all, we need to execute these commands in order to install the repository locally.

1.- Download the repository:

```bash
git clone git@github.com:arnauruana/semantic-segmentation-3d.git
```

2.- Enter the downloaded folder:

```bash
cd semantic-segmentation-3d
```

### Environment

If you don't want to use `conda` environments you can [skip](#requirements) this section, otherwise you will need to download it manually from [here](https://docs.conda.io/en/latest/miniconda.html).

Once installed, we can run the following commands to set up the execution environment.

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

In both cases, we need to execute this line to install all the `Python` requirements:

```bash
pip install -r requirements.txt
```

### Dataset

The original dataset, formally called [IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/), is a public dataset containing a point cloud from a $200$ meter street in Paris, gathered using the [LiDAR](https://www.ibm.com/topics/lidar) technology. All its $12$ million points are labeled with their respective classes and identifiers making it suitable for either [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation) or [Instance Segmentation](https://paperswithcode.com/task/instance-segmentation) tasks.

The filtered and split version of this dataset can be found in the following [link](https://drive.google.com/file/d/1fYo03lGE9E0yDbgx-KzCnDr8_wUdinKU/view?usp=drive_link). This is the actual dataset we have worked with.

Once downloaded and placed it under the root directory of this project, you can decompress it using the following command:

```bash
unzip data.zip
```

> You can also use any program capable of decompressing files. Such as [7-zip](https://www.7-zip.org/), [WinRAR](https://www.rarlab.com/) or [WinZip](https://www.winzip.com/en/) among others.

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

## Authors

* [Arnau Ruana](https://github.com/arnauruana) | developer
* [Mariona Carós](https://github.com/marionacaros) | supervisor
* [Rafel Palomo](https://github.com/RafelPalomo) | developer
* [Soraya Gonzalez](https://github.com/SorayaGonzalezSanchez) | developer
