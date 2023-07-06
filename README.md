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
conda create --name aidl-project python --yes
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

## License

This project is licensed under the `MIT License`. See the [LICENSE](./LICENSE.md) file for further information about it.

## Authors

* [Arnau Ruana](https://github.com/arnauruana) | developer
* [Mariona Carós](https://github.com/marionacaros) | supervisor
* [Rafel Palomo](https://github.com/RafelPalomo) | developer
* [Soraya Gonzalez](https://github.com/SorayaGonzalezSanchez) | developer
