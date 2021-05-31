# Interactive-Decision-Trees
This repository contains modules in Python and Jupyter Lab Notebooks for Interactive Construction and analysis of Decision Trees (DT). Python modules provide functions that enable users to incorporate their domain knowledge by interacting with the DT (e.g. manually change split points) and the InteractiveDecisionTrees Jupyter Lab notebook provides basic user-interface options for interacting with the DT for those who don't like programming. The online documentation for Interactive Decision Trees can be found here: (put link)


# Requirements

The python modules and Jupyter Lab notebooks require the following packages to be installed in the following versions:

|Package      |   Version |
|:-----------:|:---------:|    
|sklearn:     |  '0.22.1' |
|plotly :     |   '4.7.1' | 
|ipywidgets:  |   '7.5.1' |  
|python-igraph:      |   '0.8.2' |
|chart-studio:|   '1.1.0' |
|pandas:      |   '1.2.4' |
|numpy:       |  '1.18.1' |
|matplotlib:  |   '3.2.1' |

The notebooks make use of the plotly and ipywidgets libraries for the interactive plots. To run the notebooks in Jupyter Lab then:

1) Install the jupyterlab and ipywidgets packages:

    - Using pip:
        
        $ pip install jupyterlab "ipywidgets>=7.5"

    - or conda:

        $ conda install jupyterlab "ipywidgets>=7.5"

2) Then run the following commands to install the required JupyterLab extensions (note that this will require node to be installed):

    - JupyterLab renderer support:
    
        jupyter labextension install jupyterlab-plotly@4.14.3

    - Jupyter widgets extension:
    
        jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3



# Contact
If you have any questions or feedback, or if you spotted an error or bug, please email Georgios Sarailidis (g.sarailidis@bristol.ac.uk)

# Download
(Link on github to download)

# Acknowledgements
Thanks to Dan Power and Stamatis Batelis for providing helpful feedback.

This work was supported by the Engineering and Physical Sciences Research Council in the UK via grant EP/L016214/1 awarded for the Water Informatics: Science and Engineering (WISE) Centre for Doctoral Training, which is gratefully acknowledged.

# Credits
(cite the paper)

# License
