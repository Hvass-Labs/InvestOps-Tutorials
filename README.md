# InvestOps Tutorials

[Original repository on GitHub](https://github.com/Hvass-Labs/InvestOps-Tutorials)


## Introduction

[InvestOps](https://github.com/Hvass-Labs/InvestOps) is a Python package with
tools for investing and these are short tutorials on how to use them.


## Tutorials

1. Portfolio Diversification ([Notebook](https://github.com/Hvass-Labs/InvestOps-Tutorials/blob/master/01_Portfolio_Diversification.ipynb)) ([Google Colab](https://colab.research.google.com/github/Hvass-Labs/InvestOps-Tutorials/blob/master/01_Portfolio_Diversification.ipynb))


## Run in Google Colab

If you do not want to install anything on your own computer, then the
Notebooks can be viewed, edited and run entirely on the internet by using
[Google Colab](https://colab.research.google.com).

You click the "Google Colab"-link next to the tutorials listed above.
You can view the Notebook on Colab but in order to run it you need to login
using your Google account.

Most of the required Python packages should already be installed on Google
Colab, and there is a `!pip install` command near the top of each Notebook,
which can be un-commented and run so as to install the required Python
packages for that particular Notebook, like this:

    !pip install investops


## Run on Your Own Computer

If you want to run these tutorials on your own computer, then it is
recommended that you download the whole repository from GitHub,
instead of just downloading the individual Python Notebooks.


### Git Clone

The easiest way to download and install it is by using git from the command-line:

    git clone https://github.com/Hvass-Labs/InvestOps-Tutorials.git

This creates the directory `InvestOps-Tutorials` and downloads all the files to it.

This also makes it easy to update the files, simply by executing this
command inside that directory:

    git pull


### Zip-File

You can also [download](https://github.com/Hvass-Labs/InvestOps-Tutorials/archive/refs/heads/main.zip)
the contents of the GitHub repository as a Zip-file and extract it manually.


### Installation

If you want to run these tutorials on your own computer, then it is best
to use a virtual environment when installing the required packages,
so you can easily delete the environment again. You write the following
in a Linux terminal:

    virtualenv investops-env

Or you can use [Anaconda](https://www.anaconda.com/download) instead of a virtualenv:

    conda create --name investops-env python=3

Then you switch to the virtual environment and install the required packages.

    source activate investops-env
    pip install -r requirements.txt

When you are done working on the project you can deactivate the virtualenv:

    source deactivate


### Run 

Once you have installed the required Python packages in a virtual environment,
you run the following command from the `InvestOps-Tutorials` directory to view,
edit and run the Notebooks:

    source activate investops-env
    jupyter notebook


## License (MIT)

This is published under the
[MIT License](https://github.com/Hvass-Labs/InvestOps-Tutorials/blob/master/LICENSE)
which allows very broad use for both academic and commercial purposes.

You are very welcome to modify and use this source-code in your own project.
Please keep a link to the [original repository](https://github.com/Hvass-Labs/InvestOps-Tutorials).
