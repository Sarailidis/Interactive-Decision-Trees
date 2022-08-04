"""
Created on Tue Apr  7 11:34:07 2020

@author: Georgios Sarailidis

Description:
This module wraps the functions and classes defined in module iDT into widgets. The widgets are used to create Graphical User Interfaces (GUI) that enable the expert to:
i)   Import a csv file (options for random sampling and splitting in train and test sets included)
ii)  Specify the classes names and pick a color for each class
iii) (Pre)Group features (variables) by type and pick color for each group
iv)  Select important features (variables) of the dataset
v)   Create new composite features (variables) from existing ones
vi)  Manual pruning 
v)   Manually change feature (variable) and point (threshold) to split.
vi)  Manually change a leaf node class

This module is used in a Jupyter Lab notebook called InteractiveDecisionTrees which is the GUI that support the user to interactively build Decision Trees.
However, the users can select whatever GUI they want and/or need from this module. 
"""



#Libraries necessary for various uses 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
from collections import OrderedDict

#Libraries necessary for data analysis and ML methods
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
#Libraries necessary for evaluation metrics and plots
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Libraries necesaary for widgets
import ipywidgets as widgets
from ipywidgets import Layout, interact, interactive

import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Import the libraries I created for interactive construction and analysis of decision trees
from InteractiveDT import iDT

#Libraries necessary for uploading data
import io # This module provides the Python interfaces to stream handling

#Libraries necessary for random sampling
import random

#Import library necessary for various system functions
import os
import datetime
import sys




#Suppress useless warnings
import warnings
#ignore by message
warnings.filterwarnings("ignore")





# Define function necessary for importing files:
def import_file(filename, sample_size=None, random_sampling=False, header=None, index_col=None, sep = ',', train_test_splitting=True, test_size=0.25, random_state=None):
    '''
    Description: This function is used to import files from the working dierctory
    
    Inputs:
    filename:              The name of the file including the file format specification (e.g. Dataset.csv). 
    sample_size:           If random_sampling is true then it is an integer denoting the sample size to be used. Other wise it should be None. Default to None
    random_sampling:       If True then a random sample from the dataset will be imported. If False the whole dataset will be imported.
    header:                Integer or string denoting the row to be used as a header
    index_col:             Integer denoting the column to be used as row names
    sep:                   The delimeter to use. Default to ','
    train_test_splitting:  If True the data will be splitted in train and test sets. If False no splitting will be applied. Default to true
    test_size:             Float indicating the percentage of the dataset to be used for test set.
    random_state:          An integer number to be used as seed for random state. This will force the alghorithm to produce the same results.
    
    '''
    if random_sampling == False:
        dataset_df=iDT.Data_Preprocessing.Data_Preparation(filename=filename, header=None if header==-1 else header, index_col=None if index_col==-1 else index_col, sep = sep, train_test_splitting=train_test_splitting, 
                                                             test_size=test_size, random_state=random_state)
        
    elif random_sampling == True:
        filename= filename
        n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
        s = sample_size #desired sample size
        #check for invalid user input:
        if s>=n:
            print('Dataset not loaded! \nSample size must not be larger than the population.')
            dataset_df = pd.DataFrame()
        else:
            skip = sorted(random.sample(range(1,n),n-s)) #the 0-indexed header will not be included in the skip list #sorted(random.sample(range(1,n+1),n-s))
            for i in [0,1]:
                if i in skip:
                    skip.remove(i)
            dataset_sample = pd.read_csv(filename, skiprows=skip,  header=None if header==-1 else header, index_col=None if index_col==-1 else index_col) #pd.read_csv(filename, skiprows=skip, index_col=index_col)

            dataset_df=iDT.Data_Preprocessing.Data_Preparation(filename=dataset_sample, header=None if header==-1 else header, index_col=None if index_col==-1 else index_col, train_test_splitting=train_test_splitting, 
                                                                 test_size=test_size, random_state=random_state)
    
    return dataset_df









# Create function that will allow user to select important features for the dataset and also provides the option for adding some extra randomly selected features:
def select_features(dataset, important_features, random_features=False, total_features=None):
    '''
    Description: This function enables the user to specify which are the important features according to his/her expertise
    
    Inputs:
    dataset:              The available dataset (pandas data frame object)
    important_features:   List containing the names of the features the user considers important
    random_features:      If True the algorithm will randomly select a certain number of features (the number is specified by the user) and add them to the important features list.
    total_features:       The total number of features the newly created dataset should contain.
    
    Outputs:
    The new input dataset in dataframe format. The new input dataset consists of the important features specified by the user. If random_features = True the new input dataset will contain also 
    some randomly selected features. 
    '''

    if random_features==False:
        inputs_reduced=dataset.loc[:, important_features]
    elif random_features==True:
        #Find the ids of the selected features
        imp_feat_indexes=[]
        for imp_ids in important_features:
            imp_feat_indexes.append(dataset.columns.get_loc(imp_ids))

        complete_id_list=list(range(0,len(dataset.columns)))

        list_to_pick_ids=list(set(complete_id_list).difference(imp_feat_indexes))

        rand_ids=[]
        for i in range(0, total_features-len(important_features)):
            index=random.choice(list_to_pick_ids)
            rand_ids.append(index)
            list_to_pick_ids.remove(index)

        features_ids_list=imp_feat_indexes+rand_ids

        #Reformulate the dataset
        inputs_reduced=dataset.iloc[:,features_ids_list]
    return inputs_reduced





    
    
    

# Define the function for creating new variables in the dataset:       
def new_feature(Features_color_groups, equation, new_feature_name, features_labels, dataset, group_name):
    '''
    Description
    This function creates a new variable (feature) in the dataset (out of the already existing ones) based on a user defined equation
    
    Inputs:
    Features_color_groups:  The dictionary containing the following key-value pairs:
                            'Groups & parameters': dictionary containing the following key-value pairs:
                                                   'Group Name': list of strings representing the features names
                            'Colors of groups': dictionary containing the following key-value pairs:
                                                'Group Name': list with a string denoting the colour of the group
    equation:               The equation that defines the new variable (feature).
    new_feature_naem:       The name of the new variable (feature)
    features_labels:        List containing the names of the existing features.
    dataset:                The existing dataset.
    '''
    from numpy import inf, nan

    #Reform the original dataset
    dataset[new_feature_name]=dataset.eval(equation)
    features_labels.append(new_feature_name)
    #Make necessary checks for missing values of inf values
    dataset[new_feature_name].replace(to_replace=-inf, value=np.mean(dataset[new_feature_name][np.isfinite(dataset[new_feature_name])]), inplace=True)  #Replace -inf values
    dataset[new_feature_name].replace(to_replace=inf,  value=np.mean(dataset[new_feature_name][np.isfinite(dataset[new_feature_name])]), inplace=True)  #Replace inf values
    dataset[new_feature_name].replace(to_replace=nan,  value=np.mean(dataset[new_feature_name][np.isfinite(dataset[new_feature_name])]), inplace=True)  #Replace nan values
    
    #np.mean(dataset[new_feature_name][np.isfinite(dataset[new_feature_name])]) --> this is to make the inf values seen as finite and be able to calculate the mean of the array
    
    #Assign the new feature to a new color group
    if any(Features_color_groups.values()):
        Features_color_groups['Groups & parameters'][group_name].append(new_feature_name)

    

    
    
    
    
    
    
class InteractiveDecisionTreesGUI():
    '''
    Description:
    This class contains all the necessary methods for the implementation of the GUI for the Interactive Construction and Analysis of Decision Trees
    
    Inputs:
    Data_dict:                An empty dictionary. Data_dict = {}
    classes_dict:             Dictionary containing the following key-value pairs:
                                 'Classes Labels': A list containing strings (or integers) corresponding to each class
                                 'Classes Colors': A list contining the colour code of each class
    Features_color_groups:    The dictionary containing the following key-value pairs:
                                 'Groups & parameters': dictionary containing the following key-value pairs:
                                                        'Group Name': list of strings representing the features names
                                 'Colors of groups': dictionary containing the following key-value pairs:
                                                        'Group Name': list with a string denoting the colour of the group
    '''
    def __init__(self, Data_dict, classes_dict, Features_color_groups):
        self.Data_dict = Data_dict
        self.classes_dict = classes_dict
        self.Features_color_groups = Features_color_groups 
    
    # Define the necessary function for updtating the tree parameters and the corresponfing plot.    
    def interactive_plot(self, criterion, max_depth, max_features, max_leaf_nodes, min_impurity_decrease,min_samples_leaf, min_samples_split, random_state, splitter, nodes_coloring, edges_shape, 
                         plot_width=1400, plot_height=800, mrk_size=15, txt_size=15, opacity_edges=0.7, opacity_nodes=1, Best_first_Tree_Builder=True):
        '''
        Description:
        This function updates tree pramaeters specified by the expert and in the same time it update the plot.

        Inputs:                     
        criterion:                  The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        max_depth:                  The maximum depth of the tree.
        max_features:               The number of features to consider when looking for the best split:
        max_leaf_nodes:             Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. 
        min_impurity_decrease:      A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        min_samples_leaf:           The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training 
                                    samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
        min_samples_split:          The minimum number of samples required to split an internal node:
        random_state:               Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best". When max_features < n_features, 
                                    the algorithm will select max_features at random at each split before finding the best split among them. But the best found split may vary across different runs, 
                                    even if max_features=n_features. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random.
        splitter:                   The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
        nodes_coloring:             The options available to color the nodes. 'Impurity' for coloring the nodes based on their impurities.'Classes' for coloring of nodes based on their classes. 
                                    'Features_color_groups' for nodes coloring based on user defined groups
        edges_shape:                The options available for the shapes of edges. 'Lines' for lines and 'Lines-steps' for lines with steps.
        Best_first_Tree_Builder:    Whether the tree was built in best first way. Deafult to True.
        plot_width:                 An integer denoting the width of the plot. Default to 1400
        plot_height:                An integer denoting the height of the plot. Default to 800
        mrk_size:                   An integer denoting the size of the markers appearing in the plot and legend. Default to 15
        txt_size:                   An integer denoting the size of the text appearing in the plot and legend. Default to 15
        opacity_edges:              A float indicating the opacity of the edges in the plot 
        opacity_nodes:              A float indicating the opacity of the nodes in the plot
        show_figure:                Whether to show the figure or not. It can be either True or False. Default to True.
        '''
        #Checking for invalid user inputs
        maximum_features = len(self.Data_dict['features'])
        if max_features>maximum_features:
            print(f"Max_features must not exceed the maximum number of available features in the dataset {maximum_features}. A value > {maximum_features} is given. Enter a value <= {maximum_features}.") 
        elif max_features<=0:
            print(f"Max_features must be greater than 0. A value <= 0 is given. Enter a value > 0.")
        elif min_samples_leaf<1:
            print(f"Min_samples_leaf must be greater than or equal to 1. A value < 1 is given. Enter a value >= 1")
        elif min_samples_split<=1:
            print(f"Min_samples_split must be greater than 1. A value <= 1 is given. Enter a value > 1.")
        elif max_depth<1:
            print(f"Max depth must be greater than 0. A value <= 0 is given. Enter a value > 1.")
        elif max_leaf_nodes<= 1:
            print(f"Max leaf nodes must be greater than 1. A value <= 1 is given. Enter a value > 1.") 
        elif (nodes_coloring == 'Features_color_groups' and len(self.Features_color_groups['Groups & parameters']) == 0):
            print(f"No variables assigned to groups. To use this nodes coloring option, assign variables to groups using the corresponding tool in the Preprocessing Stage tab ")
        else:
            #Classify a tree with the default values for the various paramaters.
            #In this way the user will be able to manipulate all the paramaters.       
            tree_base=tree.DecisionTreeClassifier(ccp_alpha=0, class_weight=None, criterion=criterion, max_depth=max_depth, max_features=max_features, 
                                                  max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=None, min_samples_leaf=min_samples_leaf, 
                                                  min_samples_split=min_samples_split, min_weight_fraction_leaf=0.0, random_state=random_state, 
                                                  splitter=splitter)
            tree_base.fit(self.Data_dict['x'], self.Data_dict['y']) #it was: tree_base.fit(x,y) and iDT.TreeStructure(Tree=tree_base, features=features,  X=x, Y=outputs_complete_dataset, classes_labels=classes_labels, outputs_train=y)
            TreeStructureBase=iDT.TreeStructure(Tree=tree_base, features=self.Data_dict['features'],  X=self.Data_dict['x'], Y=self.Data_dict['outputs_complete_dataset'], 
                                                  classes_dict=self.classes_dict, outputs_train=self.Data_dict['outputs_train'])
            iDT.Plot_Tree(TreeStructureBase, classes_dict=self.classes_dict, criterion=criterion, nodes_coloring=nodes_coloring, edges_shape=edges_shape, 
                            User_features_color_groups=None if nodes_coloring != 'Features_color_groups' else self.Features_color_groups, 
                            plot_width=plot_width, plot_height=plot_height, mrk_size=mrk_size, txt_size=txt_size, opacity_edges=opacity_edges, opacity_nodes=opacity_nodes, show_figure=True)
    
    #Create the appropriate widgets for importing files
    def ImportDatasetGUI(self):
        '''
        This method creates the necessary widgets, functions and the GUI that enable the expert to load the file containing the dataset. Then it assigns all the widgets in a box
        '''
        #Widget for file name
        filename_widget=widgets.Text(value='', placeholder='Type the file name', description='File', disabled=False)
        #Widget for random sampling
        Random_sampling_widget=widgets.Checkbox(value=False, description='Random Sampling', disabled=False)
        #Widget for sample size
        Sample_size_widget=widgets.IntText(value=0, description='Sample Size', style = {'description_width': 'initial'}, disabled=False)
        #Widget for header
        self.Train_test_splitting=widgets.Checkbox(value=False, description='Split in train-test sets')
        #Widget for test_size
        test_size_widget=widgets.FloatText(value=0.25, description='Test Size', style = {'description_width': 'initial'}, disabled=False)
        #Widgets for header and index_col and random state
        Header_widget=widgets.IntText(value=-1, description='Header', style = {'description_width': 'initial'}, disabled=False)
        Index_col_widget=widgets.IntText(value=-1, description='Column', style = {'description_width': 'initial'}, disabled=False)
        Random_state_widget=widgets.IntText(value=None, description='Random State seed', style = {'description_width': 'initial'}, disabled=False)
        Sep_widget=widgets.Text(value=',', placeholder='Type the delimeter', description='Delimeter', disabled=False)
        #Create a button to enable files importing
        Import_files_button=widgets.Button(description='Import file', layout=Layout(width='35%'))
        Import_files_out=widgets.Output()
        #Create the function to be executed and which enables the loading of the dataset file with the button click
        def load_file(ldf, train_test_splitting=self.Train_test_splitting.value):
            Import_files_out.clear_output()
            with Import_files_out:
                #Check for invalid user inputs:
                #Empty file name:
                if filename_widget.value == '':
                    print("Filename must not be empty. An empty string is given. Type a name.")
                #Invalid filename. Not in the working directory.
                elif filename_widget.value not in os.listdir():
                    print("File doesn't exist in the current working directory. Type a valid filename")
                else:
                    n = sum(1 for line in open(filename_widget.value)) - 1 #number of records in file (excludes header)
                    #Invalind sample size
                    #User input a sample size equal or less than zero.
                    if (Random_sampling_widget.value == True and Sample_size_widget.value <=0):
                        print('Dataset not loaded! \nWhen Random Sampling is enabled, sample size must be greater than 1. A value <= 1 is given. Enter a value > 1.')
                    elif (Random_sampling_widget.value == True and Sample_size_widget.value<=1):
                        print(f"Dataset not loaded! \nWhen Random Sampling is enabled, sample size must be greater than 1. A value <= 1 is given. Enter a value > 1.")
                    elif (Random_sampling_widget.value == True and Sample_size_widget.value>=n):
                        print(f"Dataset not loaded! \nSample size must not be larger than the population {n}. A value >= {n} is given. Enter a value < {n}") 
                    #Invalid test size
                    else:
                        if (self.Train_test_splitting.value == True and Random_sampling_widget.value == True and test_size_widget.value <= 0):
                            print(f"Dataset not loaded! Test size dataset must be in the range [0,1]. Enter any value within [0,1].")
                        elif (self.Train_test_splitting.value == True and Random_sampling_widget.value == False and test_size_widget.value <= 0):
                            print(f"Dataset not loaded! Test size dataset must be in the range [0,1]. Enter any value within [0,1].")
                        elif (self.Train_test_splitting.value == True and Random_sampling_widget.value == True and test_size_widget.value > 1):
                            print(f"Dataset not loaded! Test size dataset must be in the range [0,1]. Enter any value within [0,1].")
                        elif (self.Train_test_splitting.value == True and Random_sampling_widget.value == False and test_size_widget.value > 1):
                            print(f"Dataset not loaded! Test size dataset must be in the range [0,1]. Enter any value within [0,1].")
                        else:
                            dataset_dict=import_file(filename=filename_widget.value, sample_size=Sample_size_widget.value, random_sampling=Random_sampling_widget.value, 
                                                 header=None if Header_widget.value==-1 else Header_widget.value, index_col=None if Index_col_widget.value==-1 else Index_col_widget.value,
                                                 sep = Sep_widget.value, train_test_splitting=self.Train_test_splitting.value, test_size=test_size_widget.value, 
                                                 random_state=Random_state_widget.value)

                            self.Data_dict['x']=dataset_dict['x'] 
                            self.Data_dict['y']=dataset_dict['y']
                            self.Data_dict['z']=dataset_dict['z']
                            self.Data_dict['w']=dataset_dict['w']
                            self.Data_dict['features']=dataset_dict['features']
                            self.Data_dict['outputs_complete_dataset']=dataset_dict['outputs_complete_dataset']
                            self.Data_dict['inputs_complete_dataset']=dataset_dict['inputs_complete_dataset']
                            self.Data_dict['outputs_train']=dataset_dict['outputs_train']

                            #Convert -inf and inf values to nan values
                            self.Data_dict['x'].replace(-np.inf, np.nan)
                            self.Data_dict['x'].replace(np.inf, np.nan)

                            #Find the indexes of NaN values
                            indexes=[]
                            for lbl in self.Data_dict['x'].columns:
                                idx=self.Data_dict['x'][self.Data_dict['x'].loc[:,lbl].isnull()].index.tolist()
                                if len(idx)>0:
                                    indexes.append(idx)

                            #Flatten the list (make a single list from a list of lists)        
                            indexes = [item for items in indexes for item in items]
                            #Remove duplicates from the list
                            indexes=list(set(indexes))

                            #Drop the corresponding NaN values from the dataset
                            #First from the output dataset
                            if train_test_splitting == True:
                                #check for invalid user inputs
                                if test_size_widget.value <= 0:
                                    print("Test size dataset must be greater than 0. A value <= 0 is given. Enter a value > 0.")
                                else:
                                    self.Data_dict['y'].drop(indexes, inplace=True)
                                    self.Data_dict['outputs_complete_dataset'].drop(indexes, inplace=True)
                                    #Then from the input dataset
                                    self.Data_dict['x'].dropna(inplace=True)
                                    self.Data_dict['inputs_complete_dataset'].dropna(inplace=True)
                            elif train_test_splitting == False:
                                self.Data_dict['y'].drop(indexes, inplace=True)
                                #Then from the input dataset
                                self.Data_dict['x'].dropna(inplace=True)

                            print("Dataset loaded")
                    

        #Assign the above function to the button click
        Import_files_button.on_click(load_file)
        #Filename box 
        Filename_box=widgets.HBox([filename_widget])
        #File management option widgets
        #Random sampling box
        Random_sampling_box=widgets.HBox([Random_sampling_widget, Sample_size_widget])
        #Header and column box
        Header_column_sep_box=widgets.HBox([Header_widget, Index_col_widget, Sep_widget]) 
        #Train-test splitting box 
        Train_test_box=widgets.HBox([self.Train_test_splitting, test_size_widget])
        #File Options Box
        File_options_box=widgets.VBox([Random_sampling_box, Header_column_sep_box, Train_test_box, Random_state_widget])
        #All import files options in a box
        Import_files_all_options_box=widgets.VBox([Filename_box, File_options_box])
        #Import File button output message Box
        Import_files_out_box = widgets.HBox([Import_files_button, Import_files_out])
        #Import file box
        self.Import_file_box=widgets.VBox([Import_files_all_options_box, Import_files_out_box])
        
#         self.Import_file_box=widgets.VBox([Filename_box, File_options_box, Import_files_button, Import_files_out_box]) # Import_files_button, Import_files_out
        
        
    #Create the appropriate widgets for storing the classes_labels and their corresponding colors
    def DefineClassesGUI(self):
        '''
        This method creates the necessary widgets, functions and the GUI that enable the expert to pick a color for each class.
        '''
        #Create an empty list to store the labels and colors defined by the user
        classes_labels=[]
        classes_colors=[]
        #Create a widget for typing the class label
        class_label_widget = widgets.Text(value='', placeholder='Type the class name', description='Class Name', style = {'description_width': 'initial'}, disabled=False)
#         class_label_widget = widgets.Dropdown(options=list(set(self.Data_dict['y'])), value=list(set(self.Data_dict['y']))[0], description='Class Label',  disabled=False)
        #Create a widget to enable user to pick color for each class
        pick_class_color_widget=widgets.ColorPicker(concise=False, description='Pick a color', value='blue', disabled=False)
        #Create a button to add the class label to a list
        add_label_button=widgets.Button(description='Add Class Label', layout=Layout(width='35%'))
        add_label_out=widgets.Output()
        def add_label(addlbl):
            add_label_out.clear_output()
            with add_label_out:
                #check for invalid user inputs
                if class_label_widget.value == '':
                    print("The class name must not be empty. An empty class name is given. Type a name for the class.")
                else:
                    classes_labels.append(class_label_widget.value)
                    self.classes_dict['Classes Labels']=classes_labels
                    print('Class {} Added'.format(class_label_widget.value))
            return self.classes_dict
        #Assign the above function to the button click        
        add_label_button.on_click(add_label)
        #Create a button to assign a color to each class
        assign_color_class_button=widgets.Button(description='Assign color to the Class', layout=Layout(width='35%'))
        ass_col_class_out=widgets.Output()
        def assign_color_class(asscolcl):
            ass_col_class_out.clear_output()
            with ass_col_class_out:
                classes_colors.append(pick_class_color_widget.value)
                self.classes_dict['Classes Colors']=classes_colors
                print('Color {} assigned to Class {}'.format(pick_class_color_widget.value,class_label_widget.value))
            return self.classes_dict
        #Assign the above function to the button click
        assign_color_class_button.on_click(assign_color_class)
        #Wrap class labels widgets to a box
        label_button_box=widgets.HBox([class_label_widget, add_label_button])
        #Class Labels button output message box
        label_box_out=widgets.VBox([add_label_out])
        #Label Box
        label_box=widgets.VBox([label_button_box, add_label_out])
        #Color button box
        color_class_button_box=widgets.HBox([pick_class_color_widget, assign_color_class_button])
        #Color button output message box
        color_class_out_box=widgets.VBox([ass_col_class_out])
        #Wrap color widgets to a box
        color_class_box=widgets.VBox([color_class_button_box, color_class_out_box])

        self.classes_labels_box=widgets.VBox([label_box, color_class_box])
    
    #Create the appropriate widgets for pregrouping the features and assigning colors to the groups
    def PregroupFeaturesGUI(self):
        '''
        This method creates the necessary widgets, functions and the GUI that enable the expert to pre-group variables and pick a color for each group.
        '''
        #Create a widgets for typing the group name
        group_names_widget=widgets.Text(value='', placeholder='Type the group name', description='Group', style = {'description_width': 'initial'}, disabled=False)
        #Create a widget for picking the group color
        pick_group_color_widget=widgets.ColorPicker(concise=False, description='Pick a color', value='blue', style = {'description_width': 'initial'}, disabled=False)
        #Create widgets for assigning parameters for each group
        style = {'description_width': 'initial'}
        assign_features_to_group_widget=widgets.Text(value='', placeholder='Assign variables to group', description='Variables', style = {'description_width': 'initial'}, disabled=False)
        #Create button to assign features to each group
        Assign_features_to_groups_button=widgets.Button(description='Assign variables to Group', layout=Layout(width='35%'))
        ass_feat_group_out=widgets.Output()
        #Create the function that will be executed when we will click on the assign features to groups
        def assign_features(assft):
            ass_feat_group_out.clear_output()
            with ass_feat_group_out:
                #check for invalid user inputs:
                #Empty group name
                if group_names_widget.value == '':
                    print("Group name must not be empty. An empty string is given. Type a name for the group")
                #Empty variables names
                elif assign_features_to_group_widget.value == '':
                    print("Variables must not be empty. Type the variables names (as they're named in the dataset and space separated).")
                #Invalid variables names
                for i in assign_features_to_group_widget.value.split():
                    if i not in self.Data_dict['features']:
                        print(f"Variable {i} not in dataset. Type variable names included in the dataset (space separated)")
                self.Features_color_groups['Groups & parameters'][group_names_widget.value]=assign_features_to_group_widget.value.split()
                print('Features assigned to Group {}'.format(group_names_widget.value))
            return self.Features_color_groups
        #Assign the above function to the button click        
        Assign_features_to_groups_button.on_click(assign_features)
        #Create a box that contains assign_features_to_group_widget and assign features to groups button
        Assign_features_to_groups_button_box=widgets.HBox([assign_features_to_group_widget, Assign_features_to_groups_button], box_style='info')
        #Create a box that contains assign_features_to_group_widget and assign features to groups button
        Assign_features_to_groups_out_box=widgets.VBox([ass_feat_group_out], box_style='info')
        #Create a box that contains assign_features_to_group_widget and assign features to groups button
        Assign_features_to_groups_box=widgets.VBox([Assign_features_to_groups_button_box, Assign_features_to_groups_out_box], box_style='info')
        
        #Create button to assign color to each group
        Assign_color_to_groups_button=widgets.Button(description='Assign Color to Group', layout=Layout(width='35%'))
        ass_col_group_out=widgets.Output()
        #Create the function that will be executed when we will click on the assign color to group button
        def assign_color(asscl):
            ass_col_group_out.clear_output()
            with ass_col_group_out:
                self.Features_color_groups['Colors of groups'][group_names_widget.value]=[pick_group_color_widget.value]
                print('Color {} assigned to Group {}'.format(pick_group_color_widget.value, group_names_widget.value))
            return self.Features_color_groups
        #Assign the above function to the button click
        Assign_color_to_groups_button.on_click(assign_color)
        #Create a box that contains pick_group_color_widget and Assign_color_to_groups_button
        Assign_color_to_groups_button_box=widgets.HBox([pick_group_color_widget, Assign_color_to_groups_button], box_style='info')
        #Create a box that contains the output message for pick color button
        Assign_color_to_groups_out_box=widgets.VBox([ass_col_group_out], box_style='info')
        #Create a box that contains pick_group_color_widget and Assign_color_to_groups_button
        Assign_color_to_groups_box=widgets.VBox([Assign_color_to_groups_button_box, Assign_color_to_groups_out_box], box_style='info')
        #Assemble all the above widgets to one
        self.Group_feat_col_box=widgets.VBox([group_names_widget, Assign_features_to_groups_box, Assign_color_to_groups_box],  box_style='info')
    
    #Create the appropriate widgets for selecting the important features of the dataset
    def SelectFeaturesGUI(self):
        '''
        This method creates the necessary widgets, functions and the GUI that enable the expert to select/denote the important features of the dataset.
        '''
        #Widget to enable the user to select important features
        select_features_widget=widgets.SelectMultiple(options=self.Data_dict['features'], value=[self.Data_dict['features'][0]], description='Features', disabled=False)
        #Widget to enable the user to select the total number of features
        total_features_widget=widgets.IntText(value=0, description='Total Features', style = {'description_width': 'initial'}, disabled=False)
        #Widget to enable or not the random selection of features
        random_features_widget=widgets.Checkbox(value=False, description='Random Features', disabled=False)
        
        #Create a button to start grouping parameters
        Select_features_button=widgets.Button(description='Select Features')
        Select_features_out=widgets.Output()
        
        #Create the function that will be executed when we will click on the update button
        self.ImpFeat_datestamp = 0
        def feature_selection(ftsel):
            Select_features_out.clear_output()
            with Select_features_out:
                self.Data_dict['x']=select_features(self.Data_dict['x'], important_features=list(select_features_widget.value), random_features=random_features_widget.value,
                                                    total_features=total_features_widget.value)
                self.Data_dict['features']=list(self.Data_dict['x'].columns)
                self.ImpFeat = list(OrderedDict.fromkeys(self.Data_dict['features']))
                self.ImpFeat_datestamp = datetime.datetime.now().time()
                print('Features Selected')
        #Assign the above function to the button click
        Select_features_button.on_click(feature_selection)     
        
        #Assign them all to a box
        self.Features_Selection_Box= widgets.VBox([select_features_widget, random_features_widget, total_features_widget, Select_features_button, Select_features_out], box_style='info')
        
    #Create the appropriate widgets for controlling tree size, creating new composite features, manually change split points features to split, manual pruning, manually cgange leaf node class
    def InterConAnalDTGUI(self):
        '''
        This method creates the necessary widgets, functions, classes and the GUI that enable the expert to:
        1) Control the tree size
        2) Create new composite features from existing ones.
        3) Manually change split points or features to split
        4) Manually prune the DT
        5) Manually change leaf node classes 
        6) Check the classification perfomance of the DT on the train and test sets (if previously split)
        7) Check the confusion matrix of the DT on the train and test sets (if previously split)

        '''
        #Create widgets for controlling the tree structure
        criterion_widget=widgets.Dropdown(options=['gini', 'entropy'], value='gini', description='criterion', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        max_depth_widget=widgets.IntText(value=2, description='max_depth', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        max_features_widget=widgets.IntText(value=2, description='max_features', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        max_leaf_nodes_widget=widgets.IntText(value=2, description='max_leaf_nodes', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        min_imp_decr_widget=widgets.FloatText(value=0.00001, description='min_impurity_decrease', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        min_samples_leaf_widget=widgets.IntText(value=2, description='min_samples_leaf', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        min_samples_split_widget=widgets.IntText(value=2, description='min_samples_split', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        random_state_widget=widgets.IntText(value=0, description='Random State seed', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        splitter_widget=widgets.Dropdown(options=['best','random'], value='best', description='splitter', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        nodes_coloring_widget=widgets.Dropdown(options=['Impurity', 'Classes', 'Features_color_groups'], value='Impurity', description='nodes coloring', style = {'description_width': 'initial'}, 
                                               layout=Layout(width='20%'), disabled=False)
        edges_shape_widget=widgets.Dropdown(options=['Lines', 'Lines-steps'], value='Lines', description='edges_shape', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        #Create widgets for formatting the graph
        plot_width_widget  = widgets.IntText(value=1400, description='plot_width', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        plot_height_widget = widgets.IntText(value=800, description='plot_height', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        mrk_size_widget = widgets.IntText(value=15, description='marker size', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        txt_size_widget = widgets.IntText(value=15, description='text size', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        opacity_edges_widget = widgets.FloatText(value=0.7, description='edges opacity', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        opacity_nodes_widget = widgets.FloatText(value=1.0, description='nodes opacity', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        
        #Interact function for tuning of tree parameters.
        inter_3=interact(self.interactive_plot,
                         criterion=criterion_widget, max_depth=max_depth_widget, max_features=max_features_widget, max_leaf_nodes=max_leaf_nodes_widget, min_impurity_decrease=min_imp_decr_widget,      
                         min_samples_leaf=min_samples_leaf_widget, min_samples_split=min_samples_split_widget, random_state= [None if random_state_widget.value==0 else random_state_widget], 
                         splitter=splitter_widget, nodes_coloring=nodes_coloring_widget, edges_shape=edges_shape_widget, plot_width= plot_width_widget, plot_height=plot_height_widget, 
                         mrk_size=mrk_size_widget, txt_size =txt_size_widget, opacity_edges=opacity_edges_widget, opacity_nodes=opacity_nodes_widget);
        inter_3.widget.layout.display = 'none'

        #Create a layout for interact function
        controls_label = widgets.HBox([widgets.Label(value="Decision Tree Structure Controlling Parameters and Plot Formatting")])
        controls_widgs = widgets.HBox(inter_3.widget.children[:-1], layout = Layout(flex_flow='row wrap'))
        controls = widgets.VBox([controls_label, controls_widgs])
        controls.layout.border = "1px solid"
        inter_output = inter_3.widget.children[-1]

        #Widgets for the creation of the new variable
        #Widget for the variable name
        Variable_name_widget=widgets.Text(value='', placeholder='Type Variable Name', description='Variable Name:', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        #Widget to assign the new variable to a color group
        Group_name_widget=widgets.Text(value='', placeholder='Type the group Name', description='Group Name', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)

        #Widget to write a new equation
        Equation_widget=widgets.Text(value='', placeholder='Type equation', description='Equation:', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        Equation_button=widgets.Button(description='Create Feature')
        eq_out=widgets.Output()

        #Create the functions that will be executed when we will click on the buttons
        self.NewFeat_datestamp = 0
        def create_feature(updt):
            eq_out.clear_output()
            with eq_out:
                #Checking for errors caused by invalid user inputs:
                if Variable_name_widget.value == '':
                    print(f"New variable's name must not be empty. Type a name for the new variable")
                #Check for errors in the eval function. If there are errors (e.g. invalid input variables) raise and exception.
                else:
                    try:
                        new_feature(self.Features_color_groups, Equation_widget.value, Variable_name_widget.value, self.Data_dict['features'], self.Data_dict['x'], Group_name_widget.value)    
                    except:
                        print(f"Input variable(s) not in dataset. Type an equation using the dataset's variables")
                    #If there are no errors excecute the code
                    else:
                        new_feature(self.Features_color_groups, Equation_widget.value, Variable_name_widget.value, self.Data_dict['features'], self.Data_dict['x'], Group_name_widget.value)
                        new_feature(self.Features_color_groups, Equation_widget.value, Variable_name_widget.value, self.Data_dict['features'], self.Data_dict['inputs_complete_dataset'], Group_name_widget.value)
                #         Data_dict['inputs_complete_dataset'][Variable_name_widget.value] = Data_dict['x'][Variable_name_widget.value]
                        self.Data_dict['z'][Variable_name_widget.value]=self.Data_dict['inputs_complete_dataset'].loc[self.Data_dict['z'].index,Variable_name_widget.value]
                        self.Data_dict['features']=list(OrderedDict.fromkeys(self.Data_dict['features']))
                        self.NewFeatList = self.Data_dict['features']
                        self.NewFeat_datestamp = datetime.datetime.now().time()


        #Assign the above function to the button click
        Equation_button.on_click(create_feature)

        #Define functions and widgets for changing feature and split points at specific nodes
        Node_id_widget=widgets.IntText(value=None, description='Node_id', style = {'description_width': 'initial'}, layout=Layout(width='20%'))

        #Create a widget to enable the expert to select, from a dropdown menu of available features, a feature to split 
        Feature_to_Split_widget=widgets.Dropdown(options=self.Data_dict['features'], value=self.Data_dict['features'][0], description='Features', style = {'description_width': 'initial'},
                                                 layout=Layout(width='20%'), disabled=False)

        #The list of features in the above dropdown menu will need to be updated in two cases:
        # 1) In case the expert select a set of important features then the list of avail. features should be reduced to that set
        # 2) In case the expert creates a new composite feature then this new feature should be incorporated in the list of available features.

        #Create a button to Update the list of available features to choose from:
        Updt_Feat_List_button=widgets.Button(description='Update Features')
        featlist_out=widgets.Output()
        def update_FeaturesList(ftlst):
            featlist_out.clear_output()
            with featlist_out:
                if isinstance(self.ImpFeat_datestamp, datetime.time) and self.NewFeat_datestamp == 0:
                    Feature_to_Split_widget.options = self.ImpFeat
                elif isinstance(self.ImpFeat_datestamp, datetime.time) and isinstance(self.NewFeat_datestamp, datetime.time) and self.ImpFeat_datestamp > self.NewFeat_datestamp:
                    Feature_to_Split_widget.options = self.ImpFeat
                elif isinstance(self.NewFeat_datestamp, datetime.time) and self.ImpFeat_datestamp == 0:
                    Feature_to_Split_widget.options = self.NewFeatList 
                elif isinstance(self.ImpFeat_datestamp, datetime.time) and isinstance(self.NewFeat_datestamp, datetime.time) and self.ImpFeat_datestamp < self.NewFeat_datestamp:
                    Feature_to_Split_widget.options = self.NewFeatList

        #Assign the above function to the button click
        Updt_Feat_List_button.on_click(update_FeaturesList)

        #Set the feature to split widget to observe the update button function            
        Feature_to_Split_widget.observe(update_FeaturesList)
        Feature_box_label = widgets.HBox([widgets.Label(value="Creation of New Composite Variables")])
        Feature_box_widgs= widgets.HBox([Variable_name_widget, Group_name_widget, Equation_widget, Equation_button, eq_out, Updt_Feat_List_button, featlist_out]) #, Refresh_plot_button, ref_out
        Feature_box=widgets.VBox([Feature_box_label, Feature_box_widgs])
        Feature_box.layout.border = "1px solid"

        Split_point_widget=widgets.FloatText(value=None, description='Split Point', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        max_leaf_nodes_left_widget=widgets.IntText(value=2, description='Max_leaf_nodes_left_subtree', style = {'description_width': 'initial'}, layout=Layout(width='20%'))
        max_leaf_nodes_right_widget=widgets.IntText(value=2, description='Max_leaf_nodes_right_subtree', style = {'description_width': 'initial'}, layout=Layout(width='20%'))

        #Create class that contains functions to enable the feeding of the output of each click (which is the modified tree) as input for the next click of the button.
        class modify_nodes():
            def __init__(self, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups, tree_is_pruned = False, Pruned_tree = None): 
                #The self.Data_dict is the dataset as the user has imported. So we need to pass that as input to this class.
                #And, now in the following line we write Data_dict and not self.Data_dict because we already have defined that as an argument in the class: Data_dict = self.Data_dict
                #The above are valid also for classes_dict.
                if tree_is_pruned == False:
                    Tree_modf=tree.DecisionTreeClassifier(class_weight=None, criterion=inter_3.widget.kwargs['criterion'], max_depth=inter_3.widget.kwargs['max_depth'], 
                                                          max_features=inter_3.widget.kwargs['max_features'], max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], 
                                                          min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], min_impurity_split=None, 
                                                          min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'], 
                                                          min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_weight_fraction_leaf=0.0, 
                                                          random_state=inter_3.widget.kwargs['random_state'], splitter=inter_3.widget.kwargs['splitter'])
                    Tree_modf.fit(Data_dict['x'],Data_dict['y']) 
                    self.Tree_to_modify = Tree_modf
                    self.Tree_to_modify_str = iDT.TreeStructure(self.Tree_to_modify, Data_dict['features'], Data_dict['x'], Data_dict['outputs_complete_dataset'],
                                                                  Data_dict['outputs_train'], classes_dict)
                elif tree_is_pruned == True: #This is to include the case where the tree has been: 1) pruned before modified 2) both pruned and modified in previous steps and needs to be modified again
                    self.Tree_to_modify = Pruned_tree
                    self.Last_modified_tree = Pruned_tree
            def first_modification(self, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups):
                change_thres_data=iDT.specify_feature_split_point(Tree=self.Tree_to_modify, features=Data_dict['features'], X=Data_dict['x'], Y=Data_dict['outputs_complete_dataset'],
                                                                    outputs_train=Data_dict['y'], classes_dict=classes_dict, node_id=Node_id_widget.value, feature=Feature_to_Split_widget.value,
                                                                    new_threshold=Split_point_widget.value, print_rules=False, Best_first_Tree_Builder=True, criterion=inter_3.widget.kwargs['criterion'],
                                                                    splitter=inter_3.widget.kwargs['splitter'], max_depth_left=inter_3.widget.kwargs['max_depth'],
                                                                    max_depth_right=inter_3.widget.kwargs['max_depth'], 
                                                                    min_samples_split_left=inter_3.widget.kwargs['min_samples_split'], min_samples_split_right=inter_3.widget.kwargs['min_samples_split'],
                                                                    min_samples_leaf_left=inter_3.widget.kwargs['min_samples_leaf'], min_samples_leaf_right=inter_3.widget.kwargs['min_samples_leaf'], 
                                                                    min_weight_fraction_leaf_left=0.0, min_weight_fraction_leaf_right=0.0, max_features=inter_3.widget.kwargs['max_features'], 
                                                                    random_state=1650, max_leaf_nodes_left=max_leaf_nodes_left_widget.value, max_leaf_nodes_right=max_leaf_nodes_right_widget.value, 
                                                                    min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], min_impurity_split=None, class_weight=None, 
                                                                    ccp_alpha=0.0, 
                                                                    nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'],
                                                                    User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else Features_color_groups, 
                                                                    opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], 
                                                                    plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                                                    mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], show_figure=True)
                #Merge the two tree produced from the above function
                change_thres_data.merge_subtrees() 
                change_thres_data.merged_tree_graph()
                change_thres_data.unified_plot() 
                self.Last_modified_tree = change_thres_data.unified_tree_df
            def modify(self, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups):
                change_thres_data=iDT.specify_feature_split_point(Tree=self.Last_modified_tree, features=Data_dict['features'], X=Data_dict['x'], Y=Data_dict['outputs_complete_dataset'],
                                                                    outputs_train=Data_dict['y'], classes_dict=classes_dict, node_id=Node_id_widget.value, feature=Feature_to_Split_widget.value,
                                                                    new_threshold=Split_point_widget.value, print_rules=False, Best_first_Tree_Builder=True, criterion=inter_3.widget.kwargs['criterion'],
                                                                    splitter=inter_3.widget.kwargs['splitter'], 
                                                                    max_depth_left=inter_3.widget.kwargs['max_depth'], max_depth_right=inter_3.widget.kwargs['max_depth'], 
                                                                    min_samples_split_left=inter_3.widget.kwargs['min_samples_split'], min_samples_split_right=inter_3.widget.kwargs['min_samples_split'],
                                                                    min_samples_leaf_left=inter_3.widget.kwargs['min_samples_leaf'], min_samples_leaf_right=inter_3.widget.kwargs['min_samples_leaf'], 
                                                                    min_weight_fraction_leaf_left=0.0, min_weight_fraction_leaf_right=0.0, max_features=inter_3.widget.kwargs['max_features'], 
                                                                    random_state=1650, max_leaf_nodes_left=max_leaf_nodes_left_widget.value, max_leaf_nodes_right=max_leaf_nodes_right_widget.value, 
                                                                    min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], min_impurity_split=None, class_weight=None, 
                                                                    ccp_alpha=0.0, 
                                                                    nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'],
                                                                    User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else Features_color_groups,
                                                                    opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], 
                                                                    plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                                                    mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], show_figure=True)
                #Merge the two trees produced from the above function
                change_thres_data.merge_subtrees() 
                change_thres_data.merged_tree_graph()
                change_thres_data.unified_plot()
                self.Last_modified_tree = change_thres_data.unified_tree_df

        #Create button to apply changes
        Apply_changes_button = widgets.Button(description='Apply Changes')
        Apply_changes_out = widgets.Output()
        #Create a refresh widget in case the user wants to undo all the modifications 
        Refresh_widget_mdf = widgets.Checkbox(value=False, description='Refresh')
        self.counter = 0
        #Create a widget that enables the user to check if the tree is pruned and if yes use this last_pruned tree to apply anny modifications
        tree_is_pruned_widget = widgets.Checkbox(value=False, description='Tree is Pruned')
        #Create the function that will be executed when we will click on the update button
        def apply_changes(appch):
            Apply_changes_out.clear_output()
            with Apply_changes_out:
                #Checking for errors caused by invalid user inputs
                #Invalid max_leaf nodes for left subtree
                if max_leaf_nodes_left_widget.value <= 1:
                    print("Maximum leaf nodes in the new left subtree must be greater than 1. A value <= 1 is given. Enter a value > 1.")
                #Invalid max_leaf nodes for right subtree
                elif max_leaf_nodes_right_widget.value <= 1:
                    print("Maximum leaf nodes in the new right subtree must be greater than 1. A value <= 1 is given. Enter a value > 1.")
                #Invalid split point value
                elif (Split_point_widget.value > max(self.Data_dict['x'][Feature_to_Split_widget.value]) or Split_point_widget.value < min(self.Data_dict['x'][Feature_to_Split_widget.value])):
                    print(f"Splitting threshold must be in the range [{min(self.Data_dict['x'][Feature_to_Split_widget.value])}, {max(self.Data_dict['x'][Feature_to_Split_widget.value])}]. A value outside this range is given. Enter a value within the range.")
                else:
                    global counter
                    global base
                    if tree_is_pruned_widget.value == False:
                        if Refresh_widget_mdf.value == True:
                            #Check for invalid split point value
                            max_val = max(self.Data_dict['x'][Feature_to_Split_widget.value])
                            min_val = min(self.Data_dict['x'][Feature_to_Split_widget.value])
                            if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                            else:
                                base=modify_nodes()
                                base.first_modification()
                                self.counter+=1
                        elif Refresh_widget_mdf.value == False:
                            #Check for invalid split point value
                            if self.counter==0:
                                max_val = max(self.Data_dict['x'][Feature_to_Split_widget.value])
                                min_val = min(self.Data_dict['x'][Feature_to_Split_widget.value])
                                if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                    print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                                else:
                                    base=modify_nodes()
                                    base.first_modification()
                                    self.counter+=1
                            elif self.counter>0:
                                #Check for invalid split point value
                                NodeData = iDT.get_nodes_data(base.Last_modified_tree, self.Data_dict['x'])
                                min_val = min(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                max_val = max(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                    print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                                else:
                                    base.modify()  
                                    self.counter+=1
                    elif tree_is_pruned_widget.value == True:
                        if Refresh_widget_mdf.value == True:
                            #Check for invalid split point value
                            NodeData = iDT.get_nodes_data(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                            min_val = min(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                            max_val = max(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                            base=modify_nodes(tree_is_pruned = tree_is_pruned_widget.value, Pruned_tree = base_prn.Last_Pruned_tree)
                            if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                            else:
                                base.first_modification()
                                self.counter+=1
                        elif Refresh_widget_mdf.value == False:
                            if self.counter==0:
                                #Check for invalid split point value
                                NodeData = iDT.get_nodes_data(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                                min_val = min(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                max_val = max(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                base=modify_nodes(tree_is_pruned = tree_is_pruned_widget.value, Pruned_tree = base_prn.Last_Pruned_tree)
                                if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                    print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                                else:
                                    base=modify_nodes(tree_is_pruned = tree_is_pruned_widget.value, Pruned_tree = base_prn.Last_Pruned_tree)
                                    base.first_modification()
                                    self.counter+=1
                            elif self.counter>0:
                                #Check for invalid split point value
                                NodeData = iDT.get_nodes_data(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                                min_val = min(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                max_val = max(NodeData[Node_id_widget.value][Feature_to_Split_widget.value])
                                base=modify_nodes(tree_is_pruned = tree_is_pruned_widget.value, Pruned_tree = base_prn.Last_Pruned_tree)
                                if (Split_point_widget.value > max_val or Split_point_widget.value < min_val):
                                    print(f"Splitting threshold must be in the range [{min_val}, {max_val}]. A value outside this range is given. Enter a value within the range.")
                                else:
                                    base=modify_nodes(tree_is_pruned = tree_is_pruned_widget.value, Pruned_tree = base_prn.Last_Pruned_tree)
                                    base.modify()  
                                    self.counter+=1



        #Assign the above function to the button click
        Apply_changes_button.on_click(apply_changes)

        #Make a box containing sepcify feature and split point widgets
        Specify_box_label = widgets.HBox([widgets.Label(value="Manually change node variable and threshold to split")])
        Specify_box_widgs = widgets.HBox([Node_id_widget, Feature_to_Split_widget, Split_point_widget, max_leaf_nodes_left_widget, max_leaf_nodes_right_widget, 
                                          Apply_changes_button, Refresh_widget_mdf, tree_is_pruned_widget], layout = Layout(flex_flow='row wrap'), box_style='info')
        Specify_feature_split_point_Box = widgets.VBox([Specify_box_label, Specify_box_widgs])
        Specify_feature_split_point_Box.layout.border = "1px solid"

        #Create a widget to enable the expert to input the node id of the branch to be pruned.
        node_to_prune_widget = widgets.IntText(value=None, description='Node to prune',  style = {'description_width': 'initial'})

        #This is a supporting coding block for the prune button. 
        #In this button we need to create a variable that will contain the "original tree" which is the tree that was last modified in the 1st tab.
        #This will create a TreeStructure which is the same as the treestructure in tab and it will be used when the user wants to go back to modify the tree in the 1st tab.

        tree_orig=tree.DecisionTreeClassifier(class_weight=None, criterion=inter_3.widget.kwargs['criterion'], max_depth=inter_3.widget.kwargs['max_depth'], 
                                              max_features=inter_3.widget.kwargs['max_features'], max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], 
                                              min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], min_impurity_split=None, min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'], 
                                              min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_weight_fraction_leaf=0.0, 
                                              random_state=inter_3.widget.kwargs['random_state'], splitter=inter_3.widget.kwargs['splitter'])
        tree_orig.fit(self.Data_dict['x'], self.Data_dict['y']) 
        TreeStructure_orig=iDT.TreeStructure(Tree=tree_orig, features=self.Data_dict['features'],  X=self.Data_dict['x'], Y=self.Data_dict['outputs_complete_dataset'],
                                               outputs_train=self.Data_dict['outputs_train'], classes_dict=self.classes_dict)    


        #We create a class that will contain two methods: 
        #Method first pruning is used to prune the tree plotted in tab Interactive Analysis of Decision Tree and it will store the pruned tree in order to use it then as input to the second method
        #of this class:
        #Method pruning is used for every pruning after the first pruning of the initial tree

        class Pruning:
            def __init__(self, nd_id, Tree_to_prune, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups, modified = False):
                #The self.Data_dict is the dataset as the user has imported. So we need to pass that as input to this class.
                #And, now in the following line we write Data_dict and not self.Data_dict because we already have defined that as an argument in the class: Data_dict = self.Data_dict
                #The above are valid also for classes_dict.
                self.nd_id = nd_id
                if modified == True:
                    self.Tree_to_prune = base.Last_modified_tree
                elif modified == False:
                    tree_orig=tree.DecisionTreeClassifier(class_weight=None, criterion=inter_3.widget.kwargs['criterion'], max_depth=inter_3.widget.kwargs['max_depth'], 
                                                  max_features=inter_3.widget.kwargs['max_features'], max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], 
                                                  min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], min_impurity_split=None, min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'], 
                                                  min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_weight_fraction_leaf=0.0, 
                                                  random_state=inter_3.widget.kwargs['random_state'], splitter=inter_3.widget.kwargs['splitter'])
                    tree_orig.fit(Data_dict['x'], Data_dict['y']) 
                    TreeStructure_orig=iDT.TreeStructure(Tree=tree_orig, features=Data_dict['features'],  X=Data_dict['x'], Y=Data_dict['outputs_complete_dataset'],
                                                           outputs_train=Data_dict['outputs_train'], classes_dict=classes_dict)
                    self.Tree_to_prune = TreeStructure_orig
            def first_pruning(self, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups):
                treebase=iDT.ManualPruning(node_id=self.nd_id, TreeObject=self.Tree_to_prune, classes_dict=classes_dict, criterion=inter_3.widget.kwargs['criterion'], 
                                             nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                             User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else Features_color_groups, 
                                             Best_first_Tree_Builder=inter_3.widget.kwargs['Best_first_Tree_Builder'], 
                                             txt_size=inter_3.widget.kwargs['txt_size'], mrk_size=inter_3.widget.kwargs['mrk_size'], 
                                             plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'])
                self.Last_Pruned_tree=treebase.Pruned_tree
            def pruning(self, nd_id, modified_tree=None, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups):
                if tree_is_pruned_widget.value == False:  
                    treebase=iDT.ManualPruning(node_id=nd_id, TreeObject=self.Last_Pruned_tree, classes_dict=classes_dict, criterion=inter_3.widget.kwargs['criterion'], 
                                                 nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                                 User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else Features_color_groups, 
                                                 Best_first_Tree_Builder=inter_3.widget.kwargs['Best_first_Tree_Builder'],
                                                 txt_size=inter_3.widget.kwargs['txt_size'], mrk_size=inter_3.widget.kwargs['mrk_size'], 
                                                 plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'])
                    self.Last_Pruned_tree=treebase.Pruned_tree   
                elif tree_is_pruned_widget.value == True: #This is to include the case where the tree has been both modified and pruned in previous steps 
                    self.Last_Pruned_tree=modified_tree
                    treebase=iDT.ManualPruning(node_id=nd_id, TreeObject=self.Last_Pruned_tree, classes_dict=classes_dict, criterion=inter_3.widget.kwargs['criterion'], 
                                                 nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                                 User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else Features_color_groups, 
                                                 Best_first_Tree_Builder=inter_3.widget.kwargs['Best_first_Tree_Builder'],
                                                 txt_size=inter_3.widget.kwargs['txt_size'], mrk_size=inter_3.widget.kwargs['mrk_size'], 
                                                 plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'])
                    self.Last_Pruned_tree=treebase.Pruned_tree  


        #Create a refresh widget that helps the user to undo all the pruning.        
        Refresh_widget = widgets.Checkbox(value=False, description='Refresh')
        #Create the widget that helps the user to specify whether the tree has been previously modified
        Tree_is_modified_widget = widgets.Checkbox(value=False, description='Modified')
        #Create a Button widget to enable the user to prune the tree
        Prune_button=widgets.Button(description='Prune')
        prune_out=widgets.Output()        
        self.counter_prn=0
        #Define the function that will be executed when the prune button is clicked
        def Prune(prn):
            prune_out.clear_output()
            with prune_out:
                global counter_prn
                global base_prn
                #Check for invalid user inputs:
                if Tree_is_modified_widget.value == False:
                    if Refresh_widget.value == True:
                        base_prn=Pruning(node_to_prune_widget.value, TreeStructure_orig, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups,
                                             modified = Tree_is_modified_widget.value)
                        #Check for invalid user inputs
                        #Node id given greater than total number of nodes
                        if ((isinstance(base_prn.Tree_to_prune, iDT.TreeStructure) and node_to_prune_widget.value > base_prn.Tree_to_prune.n_nodes) or
                            (isinstance(base_prn.Tree_to_prune, pd.DataFrame) and node_to_prune_widget.value > len(base_prn.Tree_to_prune.loc[:,:]))):
                            print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                        #Node id given not a leaf node
                        elif ((isinstance(base_prn.Tree_to_prune, iDT.TreeStructure) and node_to_prune_widget.value in base_prn.Tree_to_prune.leaves) or
                            (isinstance(base_prn.Tree_to_prune, pd.DataFrame) and node_to_prune_widget.value in list(base_prn.Tree_to_prune.loc[base_prn.Tree_to_prune.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                            print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                        else:
#                             base_prn=Pruning(node_to_prune_widget.value, TreeStructure_orig, Data_dict = self.Data_dict, classes_dict = self.classes_dict, Features_color_groups = self.Features_color_groups,
#                                              modified = Tree_is_modified_widget.value)
                            base_prn.first_pruning()
                            self.counter_prn+=1
                    elif Refresh_widget.value == False:
                        if self.counter_prn==0:
                            base_prn=Pruning(node_to_prune_widget.value, TreeStructure_orig, Data_dict = self.Data_dict, classes_dict = self.classes_dict, 
                                             Features_color_groups = self.Features_color_groups, modified = Tree_is_modified_widget.value)
                            #Check for invalid user inputs
                            #Node id given greater than total number of nodes
                            if ((isinstance(base_prn.Tree_to_prune, iDT.TreeStructure) and node_to_prune_widget.value > base_prn.Tree_to_prune.n_nodes) or
                                (isinstance(base_prn.Tree_to_prune, pd.DataFrame) and node_to_prune_widget.value > len(base_prn.Tree_to_prune.loc[:,:]))):
                                print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                            #Node id given greater than total number of nodes
                            elif ((isinstance(base_prn.Tree_to_prune, iDT.TreeStructure) and node_to_prune_widget.value in base_prn.Tree_to_prune.leaves) or
                                  (isinstance(base_prn.Tree_to_prune, pd.DataFrame) and node_to_prune_widget.value in list(base_prn.Tree_to_prune.loc[base_prn.Tree_to_prune.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                                print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                            else:
#                                 base_prn=Pruning(node_to_prune_widget.value, TreeStructure_orig, Data_dict = self.Data_dict, classes_dict = self.classes_dict, 
#                                                  Features_color_groups = self.Features_color_groups, modified = Tree_is_modified_widget.value)
                                base_prn.first_pruning()
                                self.counter_prn+=1
                        elif self.counter_prn>0:
                            #Check for invalid user inputs
                            #Node id given greater than total number of nodes
                            if ((isinstance(base_prn.Last_Pruned_tree, iDT.TreeStructure) and node_to_prune_widget.value > base_prn.Last_Pruned_tree.n_nodes) or
                                (isinstance(base_prn.Last_Pruned_tree, pd.DataFrame) and node_to_prune_widget.value > len(base_prn.Last_Pruned_tree.loc[:,:]))):
                                print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                            #Node id given greater than total number of nodes
                            elif ((isinstance(base_prn.Last_Pruned_tree, iDT.TreeStructure) and node_to_prune_widget.value in base_prn.Last_Pruned_tree.leaves) or
                                  (isinstance(base_prn.Last_Pruned_tree, pd.DataFrame) and node_to_prune_widget.value in list(base_prn.Last_Pruned_tree.loc[base_prn.Last_Pruned_tree.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                                print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                            else:
                                base_prn.pruning(node_to_prune_widget.value)  
                                self.counter_prn+=1
                elif Tree_is_modified_widget.value == True:
                    if Refresh_widget.value == True:
                        #Check for invalid user inputs
                        #Node id given greater than total number of nodes
                        if ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value > base.Last_modified_tree.n_nodes) or
                            (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value > len(base.Last_modified_tree.loc[:,:]))):
                            print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                        #Node id given greater than total number of nodes
                        elif ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value in base.Last_modified_tree.leaves) or
                              (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value in list(base.Last_modified_tree.loc[base.Last_modified_tree.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                            print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                        else:
                            base_prn=Pruning(node_to_prune_widget.value, base.Last_modified_tree, Data_dict = self.Data_dict, classes_dict = self.classes_dict, 
                                             Features_color_groups = self.Features_color_groups, modified = Tree_is_modified_widget.value)
                            base_prn.first_pruning()
                            self.counter_prn+=1
                    elif Refresh_widget.value == False:
                        if self.counter_prn==0:
                            #Check for invalid user inputs
                            #Node id given greater than total number of nodes
                            if ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value > base.Last_modified_tree.n_nodes) or
                                (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value > len(base.Last_modified_tree.loc[:,:]))):
                                print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                            #Node id given greater than total number of nodes
                            elif ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value in base.Last_modified_tree.leaves) or
                                (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value in list(base.Last_modified_tree.loc[base.Last_modified_tree.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                                print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                            else:
                                base_prn=Pruning(node_to_prune_widget.value, base.Last_modified_tree, Data_dict = self.Data_dict, classes_dict = self.classes_dict, 
                                                 Features_color_groups = self.Features_color_groups, modified = Tree_is_modified_widget.value)
                                base_prn.first_pruning()
                                self.counter_prn+=1
                        elif self.counter_prn>0:
                            #Check for invalid user inputs
                            #Node id given greater than total number of nodes
                            if ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value > base.Last_modified_tree.n_nodes) or
                                (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value > len(base.Last_modified_tree.loc[:,:]))):
                                print("The id of the node to be pruned must not exceed the total number of nodes in the tree. Enter a valid node id")
                            #Node id given greater than total number of nodes
                            elif ((isinstance(base.Last_modified_tree, iDT.TreeStructure) and node_to_prune_widget.value in base.Last_modified_tree.leaves) or
                                (isinstance(base.Last_modified_tree, pd.DataFrame) and node_to_prune_widget.value in list(base.Last_modified_tree.loc[base.Last_modified_tree.loc[:,'nodes_thresholds'] == -2, 'Id']))):
                                print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given. Enter a test (internal) node id")
                            else:
                                base_prn.pruning(node_to_prune_widget.value, base.Last_modified_tree)  
                                self.counter_prn+=1

        #Assign the above function to prune button click
        Prune_button.on_click(Prune)        
        #Make a box containing everything useful for the pruning interaction
        Pruning_label = widgets.HBox([widgets.Label(value="Manually prune DT nodes and branches")])
        Pruning_widgs = widgets.VBox([node_to_prune_widget, Prune_button, Tree_is_modified_widget, Refresh_widget, prune_out], box_style='info')
        Pruning_box = widgets.VBox([Pruning_label, Pruning_widgs])
        Pruning_box.layout.border = "1px solid"

        #Create a button for manually changing classes to leaves nodes
        Change_class_button = widgets.Button(description='Change Class')
        change_leaf_node_class_out = widgets.Output()     

        #Create necessary widgets
        leaf_node_id_widget = widgets.IntText(value=None, description='Leaf Node ID',  style = {'description_width': 'initial'})
        new_class_widget = widgets.Text(value='', placeholder='Type the new class', description='Class', style = {'description_width': 'initial'}, disabled=False)
        # Create a widget for denoting what changes (if any) have been made to the tree
        Last_tree_interaction = ['Last modified', 'Last pruned']
        Last_tree_interaction_widget = widgets.Dropdown(options=Last_tree_interaction, value=Last_tree_interaction[0], description='Tree State', style = {'description_width': 'initial'},
                                                        layout=Layout(width='20%'), disabled=False)

        def change_leaf_node_class(chg_cl):
            change_leaf_node_class_out.clear_output()
            with change_leaf_node_class_out: 
                if self.counter == 0 and self.counter_prn == 0:
                    global base
                    base = modify_nodes()
                    #Check for invalid user inputs
                    #Node id given is not a leaf
                    if (isinstance(base.Tree_to_modify_str, iDT.TreeStructure) and base.Tree_to_modify_str.feature_labels[leaf_node_id_widget.value] != 'leaf_node' or
                        isinstance(base.Tree_to_modify_str, pd.DataFrame) and base.Tree_to_modify_str.loc[leaf_node_id_widget.value,'nodes_labels'] != 'leaf_node'):
                        print("The node id must be a leaf node. A test node id is given. Enter a leaf node id")
                    elif(new_class_widget.value not in self.classes_dict['Classes Labels']):
#                     elif (isinstance(base.Tree_to_modify_str, iDT.TreeStructure) and new_class_widget.value not in base.Tree_to_modify_str.Node_classes or
#                           isinstance(base.Tree_to_modify_str, pd.DataFrame) and new_class_widget.value not in list(base.Tree_to_modify_str.loc[leaf_node_id_widget.value,'nodes_classes'])):
                        print("The new class does not exist in the dataset. Define the new class using the corresponding tool in the preprocessing stage tab")
                    else:
                        iDT.change_class(leaf_node_id_widget.value, base.Tree_to_modify_str, new_class_widget.value)
                        iDT.Plot_Tree(base.Tree_to_modify_str, self.classes_dict, criterion=inter_3.widget.kwargs['criterion'], Best_first_Tree_Builder=True, 
                                        nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                        User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else self.Features_color_groups,
                                        plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                        mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], 
                                        opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], show_figure=True)
                elif self.counter != 0 and self.counter_prn == 0:
                    if (isinstance(base.Last_modified_tree, iDT.TreeStructure) and base.Last_modified_tree.feature_labels[leaf_node_id_widget.value] != 'leaf_node' or
                        isinstance(base.Last_modified_tree, pd.DataFrame) and base.Last_modified_tree.loc[leaf_node_id_widget.value,'nodes_labels'] != 'leaf_node'):
                        print("The node id must be a leaf node. A test node id is given. Enter a leaf node id")
                    elif(new_class_widget.value not in self.classes_dict['Classes Labels']):
                        print("The new class does not exist in the dataset. Define the new class using the corresponding tool in the preprocessing stage tab.")
                    else:
                        iDT.change_class(leaf_node_id_widget.value, base.Last_modified_tree, new_class_widget.value)
                        iDT.Plot_Tree(base.Last_modified_tree, self.classes_dict, criterion=inter_3.widget.kwargs['criterion'], Best_first_Tree_Builder=True,
                                        nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                        User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else self.Features_color_groups,
                                        plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                        mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], 
                                        opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], show_figure=True)
                elif self.counter != 0 and self.counter_prn != 0:
                    if Last_tree_interaction_widget.value == 'Last pruned':
                        if (isinstance(base_prn.Last_Pruned_tree, iDT.TreeStructure) and base_prn.Last_Pruned_tree.feature_labels[leaf_node_id_widget.value] != 'leaf_node' or
                            isinstance(base_prn.Last_Pruned_tree, pd.DataFrame) and base_prn.Last_Pruned_tree.loc[leaf_node_id_widget.value,'nodes_labels'] != 'leaf_node'):
                            print("The node id must be a leaf node. A test node id is given. Enter a leaf node id")
                        elif (new_class_widget.value not in self.classes_dict['Classes Labels']):
                            print("The new class does not exist in the dataset. Define the new class using the corresponding tool in the preprocessing stage tab")
                        else:
                            iDT.change_class(leaf_node_id_widget.value, base_prn.Last_Pruned_tree, new_class_widget.value)
                            iDT.Plot_Tree(base_prn.Last_Pruned_tree, self.classes_dict, criterion=inter_3.widget.kwargs['criterion'], Best_first_Tree_Builder=True,
                                            nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                            User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else self.Features_color_groups,
                                            plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'],
                                            mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], 
                                            opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], show_figure=True)
                    elif Last_tree_interaction_widget.value == 'Last modified':
                        if (isinstance(base.Last_modified_tree, iDT.TreeStructure) and base.Last_modified_tree.feature_labels[leaf_node_id_widget.value] != 'leaf_node' or
                            isinstance(base.Last_modified_tree, pd.DataFrame) and base.Last_modified_tree.loc[leaf_node_id_widget.value,'nodes_labels'] != 'leaf_node'):
                            print("The node id must be a leaf node. A test node id is given. Enter a leaf node id")
                        elif(new_class_widget.value not in self.classes_dict['Classes Labels']):
                            print("The new class does not exist in the dataset. Define the new class using the corresponding tool in the preprocessing stage tab")
                        else:
                            iDT.change_class(leaf_node_id_widget.value, base.Last_modified_tree, new_class_widget.value)
                            iDT.Plot_Tree(base.Last_modified_tree, self.classes_dict, criterion=inter_3.widget.kwargs['criterion'], Best_first_Tree_Builder=True,
                                            nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                            User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else self.Features_color_groups,
                                            plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                            mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], 
                                            opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], show_figure=True)
                elif self.counter == 0 and self.counter_prn != 0:
                    if (isinstance(base_prn.Last_Pruned_tree, iDT.TreeStructure) and base_prn.Last_Pruned_tree.feature_labels[leaf_node_id_widget.value] != 'leaf_node' or
                        isinstance(base_prn.Last_Pruned_tree, pd.DataFrame) and base_prn.Last_Pruned_tree.loc[leaf_node_id_widget.value,'nodes_labels'] != 'leaf_node'):
                        print("The node id must be a leaf node. A test node id is given. Enter a leaf node id")
                    elif(new_class_widget.value not in self.classes_dict['Classes Labels']):  
                        print("The new class does not exist in the dataset. Define the new class using the corresponding tool in the preprocessing stage tab")
                    else:
                        iDT.change_class(leaf_node_id_widget.value, base_prn.Last_Pruned_tree, new_class_widget.value)
                        iDT.Plot_Tree(base_prn.Last_Pruned_tree, self.classes_dict, criterion=inter_3.widget.kwargs['criterion'], Best_first_Tree_Builder=True, 
                                        nodes_coloring=inter_3.widget.kwargs['nodes_coloring'], 
                                        edges_shape=inter_3.widget.kwargs['edges_shape'], 
                                        User_features_color_groups=None if inter_3.widget.kwargs['nodes_coloring'] != 'Features_color_groups' else self.Features_color_groups,
                                        plot_width=inter_3.widget.kwargs['plot_width'], plot_height=inter_3.widget.kwargs['plot_height'], 
                                        mrk_size=inter_3.widget.kwargs['mrk_size'], txt_size=inter_3.widget.kwargs['txt_size'], 
                                        opacity_edges=inter_3.widget.kwargs['opacity_edges'], opacity_nodes=inter_3.widget.kwargs['opacity_nodes'], show_figure=True)

        #Assign the above function to prune button click
        Change_class_button.on_click(change_leaf_node_class)        
        #Make a box containing everything useful for the pruning interaction
        Changing_class_label = widgets.HBox([widgets.Label(value="Manually change leaf node class")])
        Changing_class_inputs_box = widgets.HBox([Last_tree_interaction_widget, leaf_node_id_widget, new_class_widget, Change_class_button], box_style='info')
        Changing_class_widgs = widgets.VBox([Changing_class_inputs_box, change_leaf_node_class_out], box_style='info')
        Changing_class_box = widgets.VBox([Changing_class_label, Changing_class_widgs])
        Changing_class_box.layout.border = "1px solid"

        #Assign all the widgets with their functionalities into a box
        self.Box=widgets.VBox([Feature_box, controls, Specify_feature_split_point_Box, inter_output, Apply_changes_out, Pruning_box, Changing_class_box], box_style='info')
    
#     def EvaluationGUI(self):
        # Create a widget for denoting what changes (if any) have been made to the tree
        Tree_states_list = ['No expert tree interactions', 'Tree was last modified', 'Tree was last pruned']
        Tree_state_widget = widgets.Dropdown(options=Tree_states_list, value=Tree_states_list[0], description='Tree State', style = {'description_width': 'initial'}, layout=Layout(width='20%'),
                                                 disabled=False)

        #Create button to calculate accuracy on train and tests sets
        Calculate_accuracies_button = widgets.Button(description='Calculate accuracy')
        accuracy_out=widgets.Output()
        #Write the function to be exectuted by clicking on the button
        def calculate_accuracy(calcacc):
            accuracy_out.clear_output()
            with accuracy_out:
                if Tree_state_widget.value == 'No expert tree interactions':
                    tree_orig=tree.DecisionTreeClassifier(criterion=inter_3.widget.kwargs['criterion'], splitter=inter_3.widget.kwargs['splitter'], max_depth=inter_3.widget.kwargs['max_depth'],
                                                          min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'],
                                                          min_weight_fraction_leaf=0.0, max_features=inter_3.widget.kwargs['max_features'], random_state=inter_3.widget.kwargs['random_state'], 
                                                          max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'], 
                                                          min_impurity_split=None, class_weight=None,ccp_alpha=0.0) 
                    tree_orig.fit(self.Data_dict['x'], self.Data_dict['y']) 
                    if self.Train_test_splitting.value == True:
                        Accuracies = {'Train': [accuracy_score(self.Data_dict['y'], tree_orig.predict(self.Data_dict['x']))],
                                      'Test' : [accuracy_score(self.Data_dict['w'], tree_orig.predict(self.Data_dict['z']))]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                    elif self.Train_test_splitting.value == False:
                        Accuracies = {'Dataset': [accuracy_score(self.Data_dict['y'], tree_orig.predict(self.Data_dict['x']))]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                elif Tree_state_widget.value == 'Tree was last modified':
                    if self.Train_test_splitting.value == True:
                        #Make predictions on train set:
                        predicted_iDT_train=iDT.classify(base.Last_modified_tree, self.Data_dict['x'])
                        predicted_iDT_train.classify()
                        #Make predictions on test set:      
                        predicted_iDT_test=iDT.classify(base.Last_modified_tree, self.Data_dict['z'])
                        predicted_iDT_test.classify()

                        Accuracies = {'Train': [accuracy_score(self.Data_dict['y'], predicted_iDT_train.predicted_classes)],
                                      'Test' : [accuracy_score(self.Data_dict['w'], predicted_iDT_test.predicted_classes)]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                    elif self.Train_test_splitting.value == False:
                        predicted_iDT=iDT.classify(base.Last_modified_tree, self.Data_dict['x'])
                        predicted_iDT.classify()
                        Accuracies = {'Dataset': [accuracy_score(self.Data_dict['y'], predicted_iDT.predicted_classes)]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                elif Tree_state_widget.value == 'Tree was last pruned':
                    if self.Train_test_splitting.value == True:
                        #Make predictions on train set:
                        predicted_iDT_train=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                        predicted_iDT_train.classify()
                        #Make predictions on test set:      
                        predicted_iDT_test=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['z'])
                        predicted_iDT_test.classify()
                        Accuracies = {'Train': [accuracy_score(self.Data_dict['y'], predicted_iDT_train.predicted_classes)],
                                      'Test' : [accuracy_score(self.Data_dict['w'], predicted_iDT_test.predicted_classes)]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                    elif self.Train_test_splitting.value == False:
                        predicted_iDT=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                        predicted_iDT.classify()
                        Accuracies = {'Dataset': [accuracy_score(self.Data_dict['y'], predicted_iDT.predicted_classes)]}
                        self.Accuracies = pd.DataFrame(Accuracies, index = ['Classification Accuracy'])
                display(self.Accuracies)

        #Assign the above function to calculate accuracy button click
        Calculate_accuracies_button.on_click(calculate_accuracy)
        Tree_state_box = widgets.HBox([Tree_state_widget, self.Train_test_splitting], box_style='info')
        #Make a box containing everything useful for the calculate accuracy interaction
        Accuracy_Box=widgets.VBox([Tree_state_box, Calculate_accuracies_button, accuracy_out], box_style='info')

        #Create button to calculate accuracy on train and tests sets
        ConfusionMatrix_button = widgets.Button(description='Plot Confusion Matrix')
        ConfusionMatrix_out=widgets.Output()
        #Write the function to be exectuted by clicking on the button
        def plot_confusion_matrix(pltcm):
            ConfusionMatrix_out.clear_output()
            with ConfusionMatrix_out:
                if len(self.classes_dict['Classes Labels']) == 0:
                    print("You havent assigned labels to classes. To plot a confusion matrix please first assign labels to classes using the appropriate tool in the Preprocessing Stage tab.")
                else:
                    if Tree_state_widget.value == 'No expert tree interactions':
                        tree_orig=tree.DecisionTreeClassifier(criterion=inter_3.widget.kwargs['criterion'], splitter=inter_3.widget.kwargs['splitter'], max_depth=inter_3.widget.kwargs['max_depth'],
                                                              min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'],
                                                              min_weight_fraction_leaf=0.0, max_features=inter_3.widget.kwargs['max_features'], random_state=inter_3.widget.kwargs['random_state'],
                                                              max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'],
                                                              min_impurity_split=None, class_weight=None,ccp_alpha=0.0) 
                        tree_orig.fit(self.Data_dict['x'],self.Data_dict['y']) 
                        if self.Train_test_splitting.value == True:
                            CM_train=confusion_matrix(self.Data_dict['y'], tree_orig.predict(self.Data_dict['x']), labels=self.classes_dict['Classes Labels'])
                            CM_train_display=ConfusionMatrixDisplay(CM_train, self.classes_dict['Classes Labels'])
                            CM_test=confusion_matrix(self.Data_dict['w'], tree_orig.predict(self.Data_dict['z']), labels=self.classes_dict['Classes Labels'])
                            CM_test_display=ConfusionMatrixDisplay(CM_test, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Train': CM_train_display, 'Test': CM_test_display}
                            f, axes = plt.subplots(1, 2, figsize=(20, 5), sharey='row')
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                                if i!=0:
                                    CM_display_dict.ax_.set_ylabel('')
                            plt.subplots_adjust(wspace=0.40, hspace=0.1)
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()
                        elif self.Train_test_splitting.value == False:
                            CM=confusion_matrix(self.Data_dict['y'], tree_orig.predict(self.Data_dict['x']), labels=self.classes_dict['Classes Labels'])
                            CM_display=ConfusionMatrixDisplay(CM, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Confusion Matrix': CM_display}
                            f, axes = plt.subplots(1, 1, figsize=(20, 5))
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()
                    elif Tree_state_widget.value == 'Tree was last modified':
                        if self.Train_test_splitting.value == True:
                            #Make predictions on train set:
                            predicted_iDT_train=iDT.classify(base.Last_modified_tree, self.Data_dict['x'])
                            predicted_iDT_train.classify()
                            #Make predictions on test set:      
                            predicted_iDT_test=iDT.classify(base.Last_modified_tree, self.Data_dict['z'])
                            predicted_iDT_test.classify()
                            #Calculate confusion Matrices:
                            CM_train=confusion_matrix(self.Data_dict['y'], predicted_iDT_train.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_train_display=ConfusionMatrixDisplay(CM_train, self.classes_dict['Classes Labels'])
                            CM_test=confusion_matrix(self.Data_dict['w'], predicted_iDT_test.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_test_display=ConfusionMatrixDisplay(CM_test, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Train': CM_train_display, 'Test': CM_test_display}
                            f, axes = plt.subplots(1, 2, figsize=(20, 5), sharey='row')
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                                if i!=0:
                                    CM_display_dict.ax_.set_ylabel('')
                            plt.subplots_adjust(wspace=0.40, hspace=0.1)
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()
                        elif self.Train_test_splitting.value == False:
                            predicted_iDT=iDT.classify(base.Last_modified_tree, self.Data_dict['x'])
                            predicted_iDT.classify()
                            #Calculate confusion Matrices:
                            CM=confusion_matrix(self.Data_dict['y'], predicted_iDT.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_display=ConfusionMatrixDisplay(CM, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Confusion Matrix': CM_display}
                            f, axes = plt.subplots(1, 1, figsize=(20, 5))
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()
                    elif Tree_state_widget.value == 'Tree was last pruned':
                        if self.Train_test_splitting.value == True:
                            #Make predictions on train set:
                            predicted_iDT_train=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                            predicted_iDT_train.classify()
                            #Make predictions on test set:      
                            predicted_iDT_test=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['z'])
                            predicted_iDT_test.classify()
                            #Calculate confusion Matrices:
                            CM_train=confusion_matrix(self.Data_dict['y'], predicted_iDT_train.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_train_display=ConfusionMatrixDisplay(CM_train, self.classes_dict['Classes Labels'])
                            CM_test=confusion_matrix(self.Data_dict['w'], predicted_iDT_test.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_test_display=ConfusionMatrixDisplay(CM_test, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Train': CM_train_display, 'Test': CM_test_display}
                            f, axes = plt.subplots(1, 2, figsize=(20, 5), sharey='row')
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                                if i!=0:
                                    CM_display_dict.ax_.set_ylabel('')
                            plt.subplots_adjust(wspace=0.40, hspace=0.1)
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()
                        elif self.Train_test_splitting.value == False:
                            predicted_iDT=iDT.classify(base_prn.Last_Pruned_tree, self.Data_dict['x'])
                            predicted_iDT.classify()
                            CM=confusion_matrix(self.Data_dict['y'], predicted_iDT.predicted_classes, labels=self.classes_dict['Classes Labels'])
                            CM_display=ConfusionMatrixDisplay(CM, self.classes_dict['Classes Labels'])
                            CM_display_dict = {'Confusion Matrix': CM_display}
                            f, axes = plt.subplots(1, 1, figsize=(20, 5))
                            for i, (key, CM_display_dict) in enumerate(CM_display_dict.items()):
                                CM_display_dict.plot(ax=axes[i], xticks_rotation=45)
                                CM_display_dict.ax_.set_title(key)
                                CM_display_dict.im_.colorbar.remove()
                                CM_display_dict.ax_.set_xlabel('Predicted')
                            f.colorbar(CM_display_dict.im_, ax=axes)
                            plt.show()

        #Assign the above function to calculate accuracy button click
        ConfusionMatrix_button.on_click(plot_confusion_matrix)
        #Make a box containing everything useful for the plot confusion matrix interaction
        CM_Box=widgets.VBox([ConfusionMatrix_button, ConfusionMatrix_out], box_style='info')
        
        #Create a widget to store the output file name:
        outputfilename_widget = widgets.Text(value='', placeholder='Type the file name', description='File Name', style = {'description_width': 'initial'}, disabled=False)
        #Create a widget with dropdown menu to select the desired format file
        file_format_widget = widgets.Dropdown(options=['pickle', 'csv'], value='pickle', description='File Format', style = {'description_width': 'initial'}, layout=Layout(width='20%'), disabled=False)
        #Create a button to output the classified tree and the related input data
        Output_tree_button = widgets.Button(description='Output DT & Data')
        Output_tree_out=widgets.Output()
        #Write the function to be exectuted by clicking on the button
        def output_DT_and_Data(outdtdt):
            Output_tree_out.clear_output()
            with Output_tree_out:
                #Check for invalid user input name:
                if outputfilename_widget.value == '':
                    print("Filename must not be empty. An empty string is given. Type a name.")
                else:
                    if Tree_state_widget.value == 'No expert tree interactions':
                        tree_orig=tree.DecisionTreeClassifier(criterion=inter_3.widget.kwargs['criterion'], splitter=inter_3.widget.kwargs['splitter'], max_depth=inter_3.widget.kwargs['max_depth'],
                                                              min_samples_split=inter_3.widget.kwargs['min_samples_split'], min_samples_leaf=inter_3.widget.kwargs['min_samples_leaf'],
                                                              min_weight_fraction_leaf=0.0, max_features=inter_3.widget.kwargs['max_features'], random_state=inter_3.widget.kwargs['random_state'],
                                                              max_leaf_nodes=inter_3.widget.kwargs['max_leaf_nodes'], min_impurity_decrease=inter_3.widget.kwargs['min_impurity_decrease'],
                                                              min_impurity_split=None, class_weight=None,ccp_alpha=0.0) 
                        tree_orig.fit(self.Data_dict['x'],self.Data_dict['y'])
                        TreeStructureBase_orig=iDT.TreeStructure(Tree=tree_orig, features=self.Data_dict['features'],  X=self.Data_dict['x'], Y=self.Data_dict['outputs_complete_dataset'], 
                                                                   classes_dict=self.classes_dict, outputs_train=self.Data_dict['outputs_train'])
                        if self.Train_test_splitting.value == True:
                            self.DT = TreeStructureBase_orig.IDs_Depths
                            if file_format_widget.value == 'pickle':
                                #Output DT
                                self.DT.to_pickle(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                                self.Data_dict['z'].to_pickle('Test_input_{}'.format(outputfilename_widget.value))
                                self.Data_dict['w'].to_pickle('Test_classes_{}'.format(outputfilename_widget.value))
                                self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                            elif file_format_widget.value == 'csv':
                                #Output DT
                                self.DT.to_csv(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                                self.Data_dict['z'].to_csv('Test_input_{}.csv'.format(outputfilename_widget.value))
                                self.Data_dict['w'].to_csv('Test_classes_{}.csv'.format(outputfilename_widget.value))
                                self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                        elif self.Train_test_splitting.value == False:
                            if file_format_widget.value == 'pickle':
                                #Output DT
                                self.DT.to_pickle(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                                self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                            elif file_format_widget.value == 'csv':
                                #Output DT
                                self.DT.to_csv(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                                self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                    if Tree_state_widget.value == 'Tree was last modified':
                        self.DT = base.Last_modified_tree
                        if file_format_widget.value == 'pickle':
                                #Output DT
                                self.DT.to_pickle(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                                self.Data_dict['z'].to_pickle('Test_input_{}'.format(outputfilename_widget.value))
                                self.Data_dict['w'].to_pickle('Test_classes_{}'.format(outputfilename_widget.value))
                                self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                        elif file_format_widget.value == 'csv':
                            #Output DT
                            self.DT.to_csv(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Data_dict['z'].to_csv('Test_input_{}.csv'.format(outputfilename_widget.value))
                            self.Data_dict['w'].to_csv('Test_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                    elif self.Train_test_splitting.value == False:
                        if file_format_widget.value == 'pickle':
                            #Output DT
                            self.DT.to_pickle(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                            self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                        elif file_format_widget.value == 'csv':
                            #Output DT
                            self.DT.to_csv(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                    if Tree_state_widget.value == 'Tree was last pruned':
                        self.DT = base_prn.Last_Pruned_tree
                        if file_format_widget.value == 'pickle':
                                #Output DT
                                self.DT.to_pickle(outputfilename_widget.value)
                                #Output the data
                                self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                                self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                                self.Data_dict['z'].to_pickle('Test_input_{}'.format(outputfilename_widget.value))
                                self.Data_dict['w'].to_pickle('Test_classes_{}'.format(outputfilename_widget.value))
                                self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                        elif file_format_widget.value == 'csv':
                            #Output DT
                            self.DT.to_csv(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Data_dict['z'].to_csv('Test_input_{}.csv'.format(outputfilename_widget.value))
                            self.Data_dict['w'].to_csv('Test_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                    elif self.Train_test_splitting.value == False:
                        if file_format_widget.value == 'pickle':
                            #Output DT
                            self.DT.to_pickle(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_pickle('Training_input_{}'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_pickle('Training_classes_{}'.format(outputfilename_widget.value))
                            self.Accuracies.to_pickle('Accuracies_{}'.format(outputfilename_widget.value))
                        elif file_format_widget.value == 'csv':
                            #Output DT
                            self.DT.to_csv(outputfilename_widget.value)
                            #Output the data
                            self.Data_dict['x'].to_csv('Training_input_{}.csv'.format(outputfilename_widget.value)) 
                            self.Data_dict['y'].to_csv('Training_classes_{}.csv'.format(outputfilename_widget.value))
                            self.Accuracies.to_csv('Accuracies_{}.csv'.format(outputfilename_widget.value))
                        
        #Assign the above function to calculate accuracy button click
        Output_tree_button.on_click(output_DT_and_Data)            
        
        #Assemble the output widgets in a box
        Output_widgets_box = widgets.HBox([outputfilename_widget, file_format_widget], box_style='info')
        Output_box = widgets.VBox([Output_widgets_box, Output_tree_button, Output_tree_out], box_style='info')
        
        self.Eval_box = widgets.VBox([Accuracy_Box, CM_Box, Output_box], box_style='info')
        
        
        
        
        