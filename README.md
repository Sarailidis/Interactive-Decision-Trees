<p align="center">
    <img width="200" height="200" src="https://github.com/Sarailidis/Interactive-Decision-Trees/blob/main/logo.png">
    <h1 align="center" style="color:rgb(49,112,223);"> Interactive Decision Trees </h1>
</p>

# Description
This repository contains the "InteractiveDT" package. The package consists of two Python modules and a Jupyter Lab notebook:
1. iDT (python module) which contains the necessary classes and functions that enable the experts to interact with the DT and incorporate their scientific knowledge.
2. iDTGUIfun (python module) which incorporates the functions and classes defined in iDT into widgets to create user interfaces that support the experts in their interactions with the DT.
3. The two python modules are used in a Jupyter Lab notebook which is the Graphical User Interface for Interactive Construction and analysis of Decision Trees (DT). 

Moreover, there are three workflows (datasets are also provided in the workflows folders) for anyone who wants to get familiar with the toolbox.


# Getting Started

It is highly recommended to install the Anaconda Navigator (https://www.anaconda.com/products/individual-b) before proceeding with the installation of this package.

The python modules and Jupyter Lab notebook require certain packages (and versions) to be installed in order to run. Therefore, it is advised to install the package in a new virtual environment. Below there are guidelines to create a new virtual environment and install the package there.

__Note:__ Depending on the computer the creation of the virtual environment and the installation of the package may take a while to complete!

## How to install

1. Clone this repository. For more information on how to clone a github repository please follow the link https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository

2. Open the Anaconda Prompt
3. Create a new virtual environment, where the package and all its dependencies will be installed, by copy pasting and running the following command in the Anaconda Prompt:

        conda create -n InteractiveDecisionTrees anaconda

4. Activate the new created environment by copy pasting and running the following command in the Anaconda Prompt. 

        conda activate InteractiveDecisionTrees

This will force Anaconda to switch from the base environment to the new environment ensuring that after the installation of this package, the packages stored in the base environment remain untouched.

5. Change the working directory to the cloned folder by using the following command:

        cd path_to_the_folder

6. Then copy paste and run the following command in the Anaconda prompt. This will install the 'InteractiveDT' package in the new environment.

        pip install .

7. After installing the 'InteractiveDT' package the node.js and npm packages need to be installed. This can been done by copy pasting and runing the following command in the Anaconda prompt

        conda install -c conda-forge nodejs
 
8. Finally, the JupyterLab renderer support and widgets extensions need to be installed. This can be done by copy pasting and running the following commands in the Anaconda prompt:
    
    a. JupyterLab renderer support:
    
        jupyter labextension install jupyterlab-plotly@4.14.3

    b. Jupyter widgets extension:
    
        jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
        
Once the installation is completed __do not close the Anaconda Prompt.__ Follow the guidelines of the next section "How to run a workflow"

## How to run a workflow

For those who are unfamiliar with Jupyter Lab below I provide some guidelines on what to do after the installation. These include how to open the Jupyter Lab application and run a workflow.

1. The Graphical User Interface and the workflows are running only on Jupyter Lab. In the Anaconda prompt (used for the installation of the package) type and run the following:
        
        jupyter lab

   This should open the Jupyter Lab application and it should look similar to this:

![image](https://user-images.githubusercontent.com/58266471/122534240-04585100-d02b-11eb-9e7a-6931bdb970ed.png)

2. On the left hand side find and click on the folder icon:

![image](https://user-images.githubusercontent.com/58266471/122539344-2accbb00-d030-11eb-99bf-a216f2fce302.png)

3. This should show you the contents of the repository:

![image](https://user-images.githubusercontent.com/58266471/122567556-75126400-d051-11eb-8884-13f6cb2e16d4.png)

4. Click on the "Workflows" folder. The contents of this folder should be three subfloders with the 3 case studies workflows:

![image](https://user-images.githubusercontent.com/58266471/122568408-62e4f580-d052-11eb-9436-1e3278f7fcf8.png)

5. Click on the one you want to practice. For example, click on the 1st_case_study and it will show you the contents on the 1st_case_study which should be as below:

![image](https://user-images.githubusercontent.com/58266471/122574646-89a62a80-d058-11eb-8b4f-4921f7f9a93a.png)

6. Finally, click on the file named "Workflow_1st_case_study.ipynb" and it should open a notebook that looks like this:

![image](https://user-images.githubusercontent.com/58266471/122576179-143b5980-d05a-11eb-912d-29839928a1fc.png)


# Contact
If you have any questions or feedback, or if you spotted an error or bug, please email Georgios Sarailidis (g.sarailidis@bristol.ac.uk)


# Acknowledgements
This work was supported by the Engineering and Physical Sciences Research Council in the UK via grant EP/L016214/1 awarded for the Water Informatics: Science and Engineering (WISE) Centre for Doctoral Training, which is gratefully acknowledged.

Thanks to Dan Power, Sebastian Gnann and Stamatis Batelis for providing helpful feedback and to Demetrios Poursanidis for designing the logo.

# Credits
(cite the paper)

# License
