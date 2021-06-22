<p align="center">
    <img width="200" height="200" src="https://github.com/Sarailidis/Interactive-Decision-Trees/blob/main/logo.png">
    <h1 align="center" style="color:rgb(49,112,223);"> Interactive Decision Trees </h1>
</p>

[![DOI](https://zenodo.org/badge/355970266.svg)](https://zenodo.org/badge/latestdoi/355970266)

# Description
This repository contains the "InteractiveDT" package. The package consists of two Python modules and a Jupyter Lab notebook:
1. iDT (python module) which contains the necessary classes and functions that enable the experts to interact with the DT and incorporate their scientific knowledge.
2. iDTGUIfun (python module) which incorporates the functions and classes defined in iDT into widgets to create user interfaces that support the experts in their interactions with the DT.
3. The two python modules are used in a Jupyter Lab notebook which is the Graphical User Interface for Interactive Construction and analysis of Decision Trees (DT). 

Moreover, there are three workflows (datasets are also provided in the workflows folders) for anyone who wants to get familiar with the toolbox.


# Getting Started

It is highly recommended to install the Anaconda Navigator (https://www.anaconda.com/products/individual-b) before proceeding with the installation of this package.

The python modules and Jupyter Lab notebook require certain packages (and versions) to be installed in order to run. Therefore, it is advised to install the package in a new virtual environment. Below there are guidelines to create a new virtual environment and install the package there.


## How to install

1. Clone this repository. For more information on how to clone a github repository please follow the link https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository

2. Open the Anaconda Prompt
3. Create a new virtual environment, where the package and all its dependencies will be installed, by copy pasting and running the following command in the Anaconda Prompt:

        conda create -n InteractiveDecisionTrees anaconda
        
    __Note:__ Depending on the computer the creation of the virtual environment may take a while to complete!

4. Activate the new created environment by copy pasting and running the following command in the Anaconda Prompt. 

        conda activate InteractiveDecisionTrees

    This will force Anaconda to switch from the base environment to the new environment ensuring that after the installation of this package, the packages stored in the base environment remain untouched.

5. Change the working directory to the cloned folder by using the following command:

        cd [add path to repository]

6. Then copy paste and run the following command in the Anaconda prompt. This will install the 'InteractiveDT' package in the new environment.

        pip install .
        
    __Note:__ Depending on the computer the installation of the package may take a while to complete!

7. After installing the 'InteractiveDT' package the node.js and npm packages need to be installed. This can been done by copy pasting and runing the following command in the Anaconda prompt

        conda install -c conda-forge nodejs
 
8. Finally, the JupyterLab renderer support and widgets extensions need to be installed. This can be done by copy pasting and running the following commands in the Anaconda prompt:
    
    a. JupyterLab renderer support:
    
        jupyter labextension install jupyterlab-plotly@4.14.3
        
    Depending on the computer this may take a while.

    b. Jupyter widgets extension:
    
        jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
       
    Depending on the computer this may take a while.
        
9. Once the installation is completed type and run the following command to run Jupyter Lab application:

        jupyter lab

To use the Graphical User Interface you need to run the InteractiveDecisionTrees.ipynb file in Jupyter Lab.

For those who are unfamiliar with Jupyter Lab I provide some guidelines on how to use Jupyter Lab on https://github.com/Sarailidis/Interactive-Decision-Trees/wiki/How-to-run-a-workflow. 

# Contact
If you have any questions or feedback, or if you spotted an error or bug, please email Georgios Sarailidis (g.sarailidis@bristol.ac.uk)


# Acknowledgements
This work was supported by the Engineering and Physical Sciences Research Council in the UK via grant EP/L016214/1 awarded for the Water Informatics: Science and Engineering (WISE) Centre for Doctoral Training, which is gratefully acknowledged.

Thanks to the authors of the paper Almeida et al, (2017) NHESS (https://doi.org/10.5194/nhess-17-225-2017) for kindly providing me with their dataset created by the landslide stability model CHASM. This dataset is used in the first case study workflow. 
Thanks to Fanny Sarrazin for providing me with the revised version of the dataset she created using the V2Karst model in https://research-information.bris.ac.uk/en/studentTheses/understanding-the-sensitivity-of-karst-groundwater-recharge-to-cl.  This dataset is used in the second and third case studies workflows as a a basis for creating sample datasets through random sampling.

Thanks to Dan Power, Sebastian Gnann and Stamatis Batelis for providing helpful feedback and to Demetrios Poursanidis for designing the logo.

# Credits
Sarailidis, G., Wagener, T. and Pianosi, F. (2021). Integrating Scientific Knowledge into Machine Learning using Interactive Decision Trees. submitted in Environmental Modelling and Software.

# License
This software is distributed under the GNU Public License Version 3. See https://www.gnu.org/licenses/gpl-3.0.en.html for details.
