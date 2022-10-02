# Deep Learning Based Image Segmentation
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Mainguide.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DOShXZQoOCqU"
      },
      "source": [
        "#**Welcome to Deep Learning based image segmentation repository**\n",
        "\n",
        "Image Segementation is the category of Digital Image processsing in which pixel level classification is carried out to classify/identify image segments/attributes eventually leading to pixel level mask for differnt objects contained in an image.\n",
        "\n",
        "Before the advent of machine learning/ Deep learning, It was carried out using conventional image processing techniques of k means clustering, Active contour and histogram based bundling.\n",
        "\n",
        "Deep learning along with CNN's has considerably improved segmentation. \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3ox0xFOOQ9i4"
      },
      "source": [
        "##**Hierrarchy of Repository**\n",
        "This repository contains different image segmentation deep learning based models along with some utility files.\n",
        "\n",
        "The **dataset** folder is named as Dataset containg folder partitions of Train, Test and Validation along with images and ground truth labels accordingly.\n",
        "\n",
        "The folder **Models** contain individual model files defining model summary and architecture with regards to convolution, pooling and activation layers etc.\n",
        "\n",
        "The **frontend** directory has base model of CNN which is resnet 101 being used here in these experiments.\n",
        "\n",
        "**Utilities** Directory provides some usefull features and functions required for preprocessing and evaluation of the work.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "68Q3sM7nTXO5"
      },
      "source": [
        "##**Requirments and Dependencies**\n",
        "The project is made using following libraries:\n",
        "\n",
        "\n",
        "1.   Python 2.7\n",
        "2.   Tensorflow 1.15\n",
        "3.   Keras\n",
        "4.   cv2\n",
        "5.   Numpy\n",
        "6.   os,sys,argparse\n",
        "7.   Tensorflow gpu\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w1ylNq6zVSoR"
      },
      "source": [
        "###**Usefull Command Line Arguments to run the snippet.**\n",
        "The script is made user friendly using argument parsing mechanism of python. Below are some usefull arguments to pass through:\n",
        "\n",
        "--model (The name of model to train, test and predict).\n",
        "\n",
        "--num_epochs (The total no of iterations ).\n",
        "\n",
        "--dataset (The path or name of datset).\n",
        "\n",
        "\n",
        "--image (the name of image just for prediction)\n",
        "\n",
        "\n",
        "--checkpoint_path(the paths to saved checkpoints)\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RmMtsfcOWXC3"
      },
      "source": [
        "!python train.py --model PSPNet --num_epochs 400 --dataset Dataset"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lMc_ScJSXu5t"
      },
      "source": [
        "**Testing model** requires the path to saved weights in checkpoints folder."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qUpq39nPX2Xp"
      },
      "source": [
        "!pyton test.py --model PSPNet --checkpoint path checkpoints/0300/model.ckpt --dataset Dataset"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WSWhM0LMYPEb"
      },
      "source": [
        "**Prediction** requires an optional argument of image path to be predicted upon."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JG0L8xxaYWUa"
      },
      "source": [
        "!python predict.py --model PSPNet --checkpoint path checkpoints/0300/model.ckpt --image image path"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zu1UnLQlZDbA"
      },
      "source": [
        "#**Note**.\n",
        "Please run this code on GPU as the model takes a lot of time to get trained. A GPU bigger than 12 GB is recommended. In order to run this notebook, kindly change the runtime of googlecolab to GPU and save the settings."
      ]
    }
  ]
}
