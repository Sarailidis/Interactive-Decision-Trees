
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:34:07 2020

@author: Georgios Sarailidis

Description:
This module contain functions and classes that enable the expert to:
i)   Import a csv file (options for random sampling and splitting in train and test sets included)
ii)  Specify the classes names and pick a color for each class
iii) (Pre)Group features (variables) by type and pick color for each group
iv)  Select important features (variables) of the dataset
v)   Create new composite features (variables) from existing ones
vi)  Manual pruning 
v)   Manually change feature (variable) and point (threshold) to split.
vi)  Manually change a leaf node class
"""

##############################################################################Import necessary libraries######################################################################################################

#Importing libraries useful for every class method contained in this script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

#Import necessary libraries useful for the functions that use several things from sklearn.Tree package
import sklearn
from sklearn.tree._tree import TREE_LEAF         
from sklearn.tree._tree import TREE_UNDEFINED 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#Import necessary libraries for Plot_tree class
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot
import igraph

#import cairo
from igraph import Graph, EdgeSeq

import warnings
from InteractiveDT import iDT
#import plotly.express as px

#Import library necessary for various system functions
import os


############################################################################################################################################################################################################
#Create function to enable the user to define groups of parameters an
def pregroup_features():
    '''Description:
    This class contains a method to create groups from the available features.
    It asks the user to define the desired number of groups. 
    Then, it asks the user to name each group.
    Then, it asks the user to assign a colour to each group
    Finally, it asks the user to assign each available feature to the user defined groups.
    '''
    n_groups=int(input('Assign the number of the groups: '))
    if n_groups>12:
        warnings.warn("Assigning too many groups may compromise the tree legibility because more groups require more colours which makes it difficult to tell apart")
    groups_names=[]
    names_template="Name group {}: "
    groups_colors={}
    colours_template = "Assign colour to group {}"
    for i in range(0,n_groups):
        print(names_template.format(i))
        groups_names.append(input())
        print(colours_template.format(i))
        groups_colors[groups_names[i]]= input()

    #Now the user needs to assign the variables to the corresponding groups
    groups_features={}
    features_groups_template="Assign features to group {} (space separated)"
    for vr in range(0, len(groups_names)):
        print(features_groups_template.format(groups_names[vr]))
        groups_features[groups_names[vr]]=input()  
        groups_features[groups_names[vr]]=groups_features[groups_names[vr]].split()

    features_color_groups_dict={'Groups & parameters': groups_features,
                                  'Colors of groups': groups_colors}
    return features_color_groups_dict









def classes_colors():
    '''Description:
    This function enable the expert to define the classes of the dataset and pick a colour for each class.
    It asks the user to set the number of the classes. 
    Then it asks the user to name each class.
    Finally, it asks the user to assign a colour to each class.
    '''
    n_classes=int(input('Assign the number of the classes: '))
    if n_classes>12:
        warnings.warn("Assigning too many classes may compromise the tree legibility because more classes require more colours which makes it difficult to tell apart")
    classes_names=[]
    names_template="Name class {}: "
    classes_colors=[]
    colours_template = "Assign colour to class {}"
    for i in range(0,n_classes):
        print(names_template.format(i))
        classes_names.append(input())
        print(colours_template.format(i))
        classes_colors.append(input())

#     #Now the user needs to assign the variables to the corresponding groups
#     groups_features={}
#     features_groups_template="Assign features to group {} (space separated)"
#     for vr in range(0, len(groups_names)):
#         print(features_groups_template.format(classes_names[vr]))
#         groups_features[classes_names[vr]]=input()  
#         groups_features[classes_names[vr]]=groups_features[classes_names[vr]].split()

    features_color_groups_dict={'Classes Labels': classes_names,
                                'Classes Colors': classes_colors}
    return features_color_groups_dict









def select_features(dataset, important_features, random_features=False, total_features=None):
    '''
    Description: This function enables the user to specify which are the important features according to his/her expertise
    
    Inputs:
    dataset:              The available dataset (pandas data frame object)
    important_features:   List containing the names of the features the user considers important
    random_features:      If True the algorithm will randomly select a certain number of features (the number is specified by the user) and add them to the important features list. Default to False.
    total_features:       The total number of features the newly created dataset should contain. Default to None.
    
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









def new_feature(equation, new_feature_name, features_labels, dataset, group_name):
    '''
    Description
    This function creates a feature (new variable) in the dataset (out of the already existing ones) based on a user defined equation
    
    Inputs:
    equation:              The equation that defines the new feature (variable).
    new_feature_naem:      The name of the new feature (variable).
    features_labels:       List containing the names of the existing features.
    dataset:               The existing dataset.
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
    Features_color_groups['Groups & parameters'][group_name].append(new_feature_name)









def get_parents_and_leaves_nodes(TreeObject):
    """
    Description:
    This function retrieves the parent and leaves nodes and stores them in a numpy array
    
    Inputs:
    TreeObject:      It can be: 1) An object of class sklearn.tree._classes.DecisionTreeClassifier 2) An object of class pandas dataframe containing the necessary columns (e.g. nodes samples etc)
    
    Ouputs:
    Two numpy arrays containing the ids of the parent and leaves ids.
    """
    
    
    if isinstance(TreeObject, sklearn.tree._classes.DecisionTreeClassifier):
        #Import necessary libs to help us identify leaves
        parents=np.zeros(len(TreeObject.tree_.threshold))      #Pre-allocate necessary variables
        leaves=np.zeros(len(TreeObject.tree_.threshold))       #Pre-allocate necessary variables

        for thres in range(0,len(TreeObject.tree_.threshold)):
            #Look for parents first
            if TreeObject.tree_.threshold[thres]!=TREE_UNDEFINED:
                parents[thres]=thres
            elif TreeObject.tree_.threshold[thres] == TREE_UNDEFINED:
                leaves[thres]=thres
        root=[0] #Keep the root
        parents=parents[parents>0] #Keep only the parents (omit zeros)
        parents=np.concatenate((root, parents)) 
        leaves=leaves[leaves>0] #Keep only the leaves (ommit zeros)
        
    
    elif isinstance(TreeObject, pd.DataFrame):
        #Import necessary libs to help us identify leaves
        parents=np.zeros(len(TreeObject.loc[:, 'nodes_thresholds']))      #Pre-allocate necessary variables
        leaves=np.zeros(len(TreeObject.loc[:, 'nodes_thresholds']))       #Pre-allocate necessary variables

        for thres in range(0,len(TreeObject.loc[:, 'nodes_thresholds'])):
            #Look for parents first
            if TreeObject.loc[thres, 'nodes_thresholds'] != -2:
                parents[thres]=thres
            elif TreeObject.loc[thres, 'nodes_thresholds'] == -2:
                leaves[thres]=thres
        root=[0] #Keep the root
        parents=parents[parents>0] #Keep only the parents (omit zeros)
        parents=np.concatenate((root, parents)) 
        leaves=leaves[leaves>0] #Keep only the leaves (ommit zeros)
    
    return parents, leaves                



                
                
                
                
                
                
def get_left_and_right_children(TreeObject):
    '''
    Description:
    This function retrieves the left and right children and stores them in a numpy array
    
    Inputs:
    TreeObject:   It can be: 1) An object of class sklearn.tree._classes.DecisionTreeClassifier 2) An object of class pandas Data Frame
    
    Outputs:
    Two numpy arrays containing the ids of the left in the places where left children lie and -1 in places of right children (and vice versa)
    '''
    if isinstance(TreeObject, sklearn.tree._classes.DecisionTreeClassifier):
        Children_right=TreeObject.tree_.children_right
        Children_left=TreeObject.tree_.children_left
        
    elif isinstance(TreeObject, pd.DataFrame):
        Children_left = np.zeros(len(TreeObject.loc[:, 'Id']))
        Children_right = np.zeros(len(TreeObject.loc[:, 'Id']))
        links=get_links(TreeObject)
        par, leav = get_parents_and_leaves_nodes(TreeObject)
        for nd in TreeObject.loc[:, 'Id']:
            for ndd in par:
                if nd == links[ndd][0]:
                    Children_left[nd]=nd
                elif nd == links[ndd][1]:
                    Children_right[nd]=nd
        Children_left=Children_left[Children_left>0]
        Children_right=Children_right[Children_right>0]
    
    return Children_left, Children_right
                
                
                                
                
                
                
                
                
                
def get_nodes_data(TreeObject, input_data=None):
    """
    Description:
    This function retrieves the data for each node of the tree and it will store it in a list
    
    Inputs:
    TreeObject:    It can be: 1) An object of class TreeStructure 2) An object of class pandas dataframe
    input_data:    In case TreeObject is TreeStructure then input_data is None. In case TreeObject is pandas dataframe input_data should be another pandas dataframe containg the input dataset
    
    Ouputs:
    A list containing the data for each node in separate (sub)lists
    """
    if isinstance(TreeObject, TreeStructure):
        TreeNodesData=[]
        #Look at the id of every node in the tree
        for nd in range(TreeObject.n_nodes): 
            if nd==0: 
                TreeNodesData=[TreeObject.X] #Assign the whole dataset to the root node       

            elif nd>0:
                #Look at the parents nodes ids
                for ndd in set(TreeObject.Links.keys()): 
                    for ndi in range(0,2):
                        #Check is nd id is equal to the right child of the parent. If so store the data that correspond
                        #to the values that are lower than the threshold of the parent node
                        if nd==TreeObject.Links[ndd][0]: 
                            TreeObject.TreeNodesData.insert(nd, TreeObject.TreeNodesData[TreeObject.Links[ndd][2]][TreeObject.TreeNodesData[TreeObject.Links[ndd][2]].loc[:,TreeObject.feature_labels[TreeObject.Links[ndd][2]]]<=TreeObject.Thresholds[TreeObject.Links[ndd][2]]])
                        #Same as the above condition but for the left child. The values of this node will be larger than
                        #the threshold of the parent node
                        elif nd==TreeObject.Links[ndd][1]: 
                            TreeObject.TreeNodesData.insert(nd, TreeObject.TreeNodesData[TreeObject.Links[ndd][2]][TreeObject.TreeNodesData[TreeObject.Links[ndd][2]].loc[:,TreeObject.feature_labels[TreeObject.Links[ndd][2]]]>TreeObject.Thresholds[TreeObject.Links[ndd][2]]])    
        del(TreeObject.TreeNodesData[len(TreeObject.Thresholds):])

    elif isinstance(TreeObject, pd.DataFrame):
        tree_links=get_links(TreeObject)
        TreeNodesData=[]
        #Look at the id of every node in the tree
        for nd in range(0,len(TreeObject.loc[:,'Id'])): 
            if nd==0: 
                TreeNodesData=[input_data] #Assign the whole dataset to the root node

            elif nd>0:
                #Look at the parents nodes ids
                for ndd in set(tree_links.keys()): #We want the kers to be in order 
                    #Check is nd id is equal to the right child of the parent. If so store the data that correspond
                    #to the values that are lower than the threshold of the parent node
                    if nd==tree_links[ndd][0]: 
                        TreeNodesData.insert(nd, TreeNodesData[tree_links[ndd][2]][TreeNodesData[tree_links[ndd][2]].loc[:,TreeObject['nodes_labels'][tree_links[ndd][2]]]<=TreeObject['nodes_thresholds'][tree_links[ndd][2]]])
                    elif nd==tree_links[ndd][1]:
                        TreeNodesData.insert(nd, TreeNodesData[tree_links[ndd][2]][TreeNodesData[tree_links[ndd][2]].loc[:,TreeObject['nodes_labels'][tree_links[ndd][2]]]>TreeObject['nodes_thresholds'][tree_links[ndd][2]]])
    
    return TreeNodesData





                
                
                

def get_OutputsNodesData(TreeObject, output_data=None, output_train_data=None, TreeNodesData=None):
    """
    Description:
    This function retrieves the output data for each node of the tree and it will store it in a list.
    
    Inputs:
    TreeObject:           It can be: 1) An object of class TreeStructure 2) An object of class pandas dataframe
    output_data:          In case TreeObject is TreeStructure then output_data is None. In case TreeObject is pandas dataframe output_data should be another pandas dataframe containing the complete 
                          output dataset. The output_data is a pandas.series that contains the classes (outcome) of each instance (row in the input dataset)
    output_train_data:    In case TreeObject is TreeStructure then output_train_data is None. In case TreeObject is pandas dataframe output_train_data should be another pandas dataframe containing 
                          the training output dataset.
    TreeNodesData:        In case TreeObject is TreeStructure then TreeNodesData is None. In case TreeObject is pandas dataframe TreeNodesData should be a list of lists. The indexes of the main list 
                          correspond to the different nodes ids of the tree and each sublist contains a pandas dataframes with the data that correpond to that node.                      
    
    Ouputs:
    A list containing the output data for each node in separate (sub)lists
    """
    global OutputsNodesData
    if isinstance(TreeObject, TreeStructure):
        OutputsNodesData=[]
        classes_train=np.array(TreeObject.outputs_train)                   
        for nd in range(TreeObject.n_nodes): #Look at the id of every node in the tree
            if nd==0:
                OutputsNodesData=[TreeObject.classes_train]  #Assign to the root node the classes that correspond to the whole training outputs dataset
            else:
                OutputsNodesData.insert(nd, TreeObject.Y[TreeObject.TreeNodesData[nd].index]) #Assign the appropriate classes looking at the indexes of the original outputs object
    
    if isinstance(TreeObject, pd.DataFrame):
        OutputsNodesData=[]
        classes_train=np.array(output_train_data)                   
        for nd in range(0,len(TreeObject.loc[:,'Id'])): #Look at the id of every node in the tree
            if nd==0:
                OutputsNodesData=[classes_train]  #Assign to the root node the classes that correspond to the whole training outputs dataset
            else:
                OutputsNodesData.insert(nd, output_data[TreeNodesData[nd].index]) #Assign the appropriate classes looking at the indexes of the original outputs object
    
    return OutputsNodesData                



                
                
                
                

                
def get_links(TreeObject):
    '''
    Description: This function finds the links for the tree. Links is a list with pair of values e.g [0,1,2] where place 0 contains the parent node id, place 1 its left child id and place 2 
                 its right child id
    
    Inputs: 
    TreeObject:  It can be: 1) An object of class TreeStructure 2) An object of class pandas dataframe containing at least the columns 'Id', 'Depth', 'Node_samples', 'Node_Values',
                 'nodes_values'
    
    Outputs:
    A dictionary containing the links of the tree
    '''
    if isinstance(TreeObject, TreeStructure):
        
        Links_inter={}   #Create an empty dictionary where we will assign the links for intermediate nodes with 
                         #their parents
        Links={}         #Create an empty dictionary where we will assign the links for all nodes

        #Create a dictionary to store for each depth the nodes ids contained in it
        same_depth={}
        for depth in range(0,TreeObject.max_depth+1):    
            same_depth[depth]=TreeObject.IDs_Depths[TreeObject.IDs_Depths['Depth']==depth]
        #Create a list containing the depths of the tree
        depths=list(same_depth.keys())

        #if Best_first_Tree_Builder==True:
        #Start establishing the links for each node of the tree
        #Look at the ids of parent nodes
        for par in range(0, len(depths)-1):
            #Look at at the nodes included in depth par
            chil_list_one=list(same_depth[par+1].loc[:,'Id'])
            for ndi in chil_list_one:
                #Look at the nodes with id higher than the previously selected one
                chil_list_two=list(same_depth[par+1].loc[ndi+1:,'Id'])
                for ndd in chil_list_two: 
                    if (#Make sure that the children have subsequent ids:
                        TreeObject.IDs_Depths.loc[ndd,'Id']-TreeObject.IDs_Depths.loc[ndi,'Id']==1):
                        ids_list=list(same_depth[par].loc[:,'Id'])
                        #Look for parents
                        for nds in ids_list: 
                            zipped_lists = zip(list(TreeObject.Node_Values[ndi]), list(TreeObject.Node_Values[ndd]))
                            sum_lists = [x + y for (x, y) in zipped_lists]
                            if (#The sum of the samples of children nodes need to be equal to the parent one
                                TreeObject.Node_samples[TreeObject.IDs_Depths.loc[ndi,'Id']]+TreeObject.Node_samples[TreeObject.IDs_Depths.loc[ndd,'Id']]==TreeObject.Node_samples[TreeObject.IDs_Depths.loc[nds,'Id']]
                                #Make sure that the children cannot be at a higher depth than the parent (remember depth=1 higher than depth=2)
                                and TreeObject.IDs_Depths.loc[nds,'Depth']<TreeObject.IDs_Depths.loc[ndi,'Depth'] and TreeObject.IDs_Depths.loc[nds,'Depth']<TreeObject.IDs_Depths.loc[ndd,'Depth']
                                #Make sure that the depth differennce of the parent with the children nodes is equal to 1:
                                and TreeObject.IDs_Depths.loc[ndi,'Depth']-TreeObject.IDs_Depths.loc[nds,'Depth']==1 and TreeObject.IDs_Depths.loc[ndd,'Depth']-TreeObject.IDs_Depths.loc[nds,'Depth']==1
                                #Make sure that the sum of the node_values of each class of the children is equal to the node_values of the corresponding class of the parent
                                and sum_lists == list(TreeObject.Node_Values[nds])):                     
                                Links_inter[nds]=[ndi,ndd,nds] 
        #Concatenate the links dictionaries into one
        Links=Links_inter
                
    elif isinstance(TreeObject, pd.DataFrame): 
        Links_inter={}   #Create an empty dictionary where we will assign the links for intermediate nodes with 
                         #their parents
        Links={}         #Create an empty dictionary where we will assign the links for all nodes

        #Create a dictionary to store for each depth the nodes ids contained in it
        same_depth={}
        for depth in range(0,max(TreeObject['Depth'])+1):    
            same_depth[depth]=TreeObject[TreeObject['Depth']==depth]
        #Create a list containing the depths of the tree
        depths=list(same_depth.keys())

        #Start establishing the links for each node of the tree
        #Look at the ids of parent nodes
        for par in range(0, len(depths)-1):
            #Look at at the nodes included in depth par
            chil_list_one=list(same_depth[par+1].loc[:,'Id'])
            for ndi in chil_list_one:
                #Look at the nodes with id higher than the previously selected one
                chil_list_two=list(same_depth[par+1].loc[ndi+1:,'Id'])
                for ndd in chil_list_two: 
                    ids_list=list(same_depth[par].loc[:,'Id'])
                    #Look for parents
                    for nds in ids_list: 
                        zipped_lists = zip(list(TreeObject.loc[ndi,'nodes_values']), list(TreeObject.loc[ndd,'nodes_values']))
                        sum_lists = [x + y for (x, y) in zipped_lists]
                        if (#The sum of the samples of children nodes need to be equal to the parent one
                            TreeObject.loc[TreeObject.loc[ndi,'Id'], 'Node_Samples']+TreeObject.loc[TreeObject.loc[ndd,'Id'], 'Node_Samples']==TreeObject.loc[TreeObject.loc[nds,'Id'], 'Node_Samples']
                            #Make sure that the children cannot be at a higher depth than the parent (remember depth=1 higher than depth=2)
                            and TreeObject.loc[nds,'Depth']<TreeObject.loc[ndi,'Depth'] and TreeObject.loc[nds,'Depth']<TreeObject.loc[ndd,'Depth']
                            #Make sure that the depth differennce of the parent with the children nodes is equal to 1:
                            and TreeObject.loc[ndi,'Depth']-TreeObject.loc[nds,'Depth']==1 and TreeObject.loc[ndd,'Depth']-TreeObject.loc[nds,'Depth']==1
                            #Make sure that the sum of the node_values of each class of the children is equal to the node_values of the corresponding class of the parent
                            and sum_lists == list(TreeObject.loc[nds,'nodes_values'])):
                            Links_inter[nds]=[ndi,ndd,nds]
        #Concatenate the links dictionaries into one
        Links=Links_inter
    
    return Links

        
        
        
        
                
                
                              
def get_edge_seq(Tree_links):
    '''
    Description: This function stores in a list the edges sequence based on the tree links provided by the user (e.g. if tree_links object is [0,1,2] then the edges_seq object will be [[0,1], [0,2]])
    
    Inputs:
    Tree_links:  A dictionary containing the tree links.
    
    Outputs:
    A list containing the edges sequence    
    '''
    
    edge_seq=[]
    for i in Tree_links.keys():
        edge_seq.append([i, Tree_links[i][0]])
        edge_seq.append([i, Tree_links[i][1]])
    
    return edge_seq










      
def smallest_branch(node_id, TreeObject, Best_first_Tree_Builder=True):
    """
    Description:
    This function will look at all the nodes higher than the one specified by the user
    and will give the node ids that have sum of node samples equal to the one specified by the user

    Inputs:
    node_id:                         the node to be pruned
    TreeStructure:                   Object of class TreeStructure (TreeStructure of the classified tree that the user wants to manually prune)
    Best_first_Tree_Builder=True:    Whether the tree was built in a Best First or Depth First manne. It takes
                                     True or False values. Default to True.

    Outputs:
    A list containing the nodes ids having the same id as the one specified by the user
    """

    def flatten(l, ltypes=(list, tuple)):
        """
        Description:
        This is a helper function. It transforms a list of lists into a single list.
        """
        ltype = type(l)
        l = list(l)
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    i -= 1
                    break
                else:
                    l[i:i + 1] = l[i]
            i += 1
        return ltype(l)

    if isinstance(TreeObject, TreeStructure):
        Nodes_ids_depths=TreeObject.IDs_Depths     #Create an object that will store the nodes ids and depths
        Node_samples=TreeObject.Node_samples
        Node_values=TreeObject.Node_Values
        number_of_nodes=TreeObject.n_nodes
        sup_list=list(range(node_id, number_of_nodes))
    
    elif isinstance(TreeObject, pd.DataFrame):
        Nodes_ids_depths=TreeObject.loc[:,['Id','Depth']]     #Create an object that will store the nodes ids and depths
        Node_samples=TreeObject.loc[:,'Node_Samples']
        Node_values=TreeObject.loc[:,'nodes_values']
        number_of_nodes=len(TreeObject.loc[:,'Id'])
        sup_list=list(range(node_id, number_of_nodes))
    
    
    node_ids_list=[]

    #Check every node id available in the tree
    if Best_first_Tree_Builder==True:
        for i in sup_list:
            for j in sup_list[1:]:
                zipped_lists = zip(list(Node_values[i]), list(Node_values[j]))
                sum_lists = [x + y for (x, y) in zipped_lists]
                #We want to find the children of the node the user has entered. The sum of node samples
                #of Its children will be equal to the node samples of the node specified by the user
                if (Node_samples[i]+Node_samples[j]==Node_samples[node_id]
                and Nodes_ids_depths.loc[i,'Depth']==Nodes_ids_depths.loc[j,'Depth']
                and sum_lists == list(Node_values[node_id])):
                    node_ids_list.append([i,j])
                    node_ids_list=flatten(node_ids_list)

    elif Best_first_Tree_Builder==False:
        for i in sup_list:
            for j in sup_list[1:]:
                #We want to find the children of the node the user has entered. The sum of node samples
                #of Its children will be equal to the node samples of the node specified by the user
                if (Node_samples[i]+Node_samples[j]==Node_samples[node_id]
                and Nodes_ids_depths.loc[i,'Depth']==Nodes_ids_depths.loc[j,'Depth']):
                    node_ids_list.append([i,j])
                    node_ids_list=flatten(node_ids_list)
    
    node_ids_list=list(set(node_ids_list))

    return node_ids_list









def nodes_to_prune(node_id, TreeObject, Best_First_Tree_builder=True): 
    """
    Description:
    This function ouputs a list with the nodes ids that belong to the same branch as the node 
    specified by the user and need to be pruned.

    Inputs:
    node_id:                         the node which defines the branch to be pruned
    TreeObject:                      It can be: 1) an object of class TreeStructure (TreeStructure of the classified tree that the user wants to manually prune) 2) An object of class pandas.DataFrame
    Best_first_Tree_Builder=True:    Whether the tree was built in a Best First or Depth First manne. It takes
                                     True or False values. Default to True.

    Outputs:
    A list with the nodes ids of the branch that need to be pruned.
    """
    def flatten(l, ltypes=(list, tuple)):
        """
        Description:
        This is a helper function. It transforms a list of lists into a single list.
        """
        ltype = type(l)
        l = list(l)
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    i -= 1
                    break
                else:
                    l[i:i + 1] = l[i]
            i += 1
        return ltype(l)

    #Initialize necessary parameters
    if isinstance(TreeObject, TreeStructure):
        first_nodes=smallest_branch(node_id, TreeObject)
        init_list=smallest_branch(node_id, TreeObject)
        par=TreeObject.parents
        chil=TreeObject.leaves
        number_of_nodes=TreeObject.n_nodes
        
    
    elif isinstance(TreeObject, pd.DataFrame):
        first_nodes=smallest_branch(node_id, TreeObject)
        init_list=smallest_branch(node_id, TreeObject)
        par=list(TreeObject.loc[TreeObject.loc[:,'nodes_thresholds'] != -2, 'Id'])
        chil=list(TreeObject.loc[TreeObject.loc[:,'nodes_thresholds'] == -2, 'Id'])
        number_of_nodes=len(TreeObject.loc[:, 'Id'])        
        

    nodelist={}    
    #We start with the init_list which contains the nodes ids of the children of the node specified
    #by the user. We then start a loop in which we look if the children of the node specified by the
    #the user have their own children. If so we find their ids and update the init_list as long as
    #the length of the init_list remains smaller than the number of available nodes in the tree.

    while len(init_list)<number_of_nodes:
        for nd in init_list:
            if nd in par:
                nodelist[nd]=smallest_branch(nd, TreeObject)
                init_list.append(nodelist[nd][0])
                init_list.append(nodelist[nd][1])

    all_nodes=list(nodelist.values())
    all_nodes.append(first_nodes)
    all_nodes.append(node_id)
    nodes_to_prune=flatten(all_nodes)
    #Remove duplicate elements in the list
    nodes_to_prune=list(set(nodes_to_prune)) #it was working without this as well
    
    return nodes_to_prune









def get_left_and_right_banches_nodes(TreeObject, node_id=0):
    '''
    Description: This function finds the nodes that belong to the left and right branches
    
    Inputs: 
    TreeObject:           It can be: 1) An object of class iDT.TreeStructure 2) An object of class iDT.specify_feature_split_point 3) An onject of class pd.DataFrame
    node_id:              An integer denoting the id of the node that we want to calculate the left and right branches
    
    Outputs:
    Left_branch_nodes:    A list with the nodes ids of the left branch
    Right_branch_nodes:   A list with the nodes ids of the right branch
    '''
    if isinstance(TreeObject, TreeStructure):
        if TreeObject.IDs_Depths.loc[1,'Id'] and TreeObject.IDs_Depths.loc[2,'Id'] in TreeObject.leaves:
            Left_branch_nodes=TreeObject.IDs_Depths.loc[1,'Id']
            Right_branch_nodes=TreeObject.IDs_Depths.loc[2,'Id']
        elif TreeObject.IDs_Depths.loc[1,'Id'] in TreeObject.leaves and TreeObject.IDs_Depths.loc[2,'Id'] in TreeObject.parents:
            Left_branch_nodes=TreeObject.IDs_Depths.loc[1,'Id']
            if TreeObject.Links[2][0] in TreeObject.leaves and TreeObject.Links[2][1] in TreeObject.leaves:
                Right_branch_nodes=smallest_branch(2, TreeObject)
                Right_branch_nodes.sort() #Sort the nodes
            else:
                Right_branch_nodes=nodes_to_prune(2, TreeObject)
                Right_branch_nodes.sort() #Sort the nodes
        elif TreeObject.IDs_Depths.loc[1,'Id'] in TreeObject.parents and TreeObject.IDs_Depths.loc[2,'Id'] in TreeObject.leaves:
            if TreeObject.Links[1][0] in TreeObject.leaves and TreeObject.Links[1][1] in TreeObject.leaves:
                Left_branch_nodes=smallest_branch(1, TreeObject)
                Left_branch_nodes.sort() #Sort the nodes
            else:
                Left_branch_nodes=nodes_to_prune(1, TreeObject)
                Left_branch_nodes.sort() #Sort the nodes
            Right_branch_nodes=TreeObject.IDs_Depths.loc[2,'Id']
        else:
            if TreeObject.Links[1][0] in TreeObject.leaves and TreeObject.Links[1][1] in TreeObject.leaves:
                Left_branch_nodes=smallest_branch(1, TreeObject)
                Left_branch_nodes.sort() #Sort the nodes
            else:
                Left_branch_nodes=nodes_to_prune(1, TreeObject)
                Left_branch_nodes.sort() #Sort the nodes
            if TreeObject.Links[2][0] in TreeObject.leaves and TreeObject.Links[2][1] in TreeObject.leaves:
                Right_branch_nodes=smallest_branch(2, TreeObject)
                Right_branch_nodes.sort() #Sort the nodes
            else:
                Right_branch_nodes=nodes_to_prune(2, TreeObject)
                Right_branch_nodes.sort() #Sort the nodes
    elif isinstance(TreeObject, specify_feature_split_point):
        Left_branch_nodes=list(TreeObject.left_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
        Right_branch_nodes=list(TreeObject.right_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
    
    elif isinstance(TreeObject, pd.DataFrame):
        tree_links=get_links(TreeObject)
        parents, leaves=get_parents_and_leaves_nodes(TreeObject)
        if node_id in parents:
            left_chil=tree_links[node_id][0]
            right_chil=tree_links[node_id][1]
            if left_chil in parents and right_chil in parents:
                if TreeObject.loc[tree_links[left_chil][0], 'nodes_labels'] == 'leaf_node' and TreeObject.loc[tree_links[left_chil][1], 'nodes_labels'] == 'leaf_node':
                    Left_branch_nodes=smallest_branch(left_chil, TreeObject)
                    Left_branch_nodes.append(left_chil)
                    Left_branch_nodes.sort()
                else:
                    Left_branch_nodes=nodes_to_prune(left_chil, TreeObject)

                if TreeObject.loc[tree_links[right_chil][0], 'nodes_labels'] == 'leaf_node' and TreeObject.loc[tree_links[right_chil][1], 'nodes_labels'] == 'leaf_node':
                    Right_branch_nodes=smallest_branch(right_chil, TreeObject)
                    Right_branch_nodes.append(right_chil)
                    Right_branch_nodes.sort()
                else:
                    Right_branch_nodes=nodes_to_prune(right_chil, TreeObject)

            elif left_chil in parents and right_chil in leaves:
                if TreeObject.loc[tree_links[left_chil][0], 'nodes_labels'] == 'leaf_node' and TreeObject.loc[tree_links[left_chil][1], 'nodes_labels'] == 'leaf_node':
                    Left_branch_nodes=smallest_branch(left_chil, TreeObject)
                    Left_branch_nodes.append(left_chil)
                    Left_branch_nodes.sort()
                    Right_branch_nodes=right_chil
                else:
                    Left_branch_nodes=nodes_to_prune(left_chil, TreeObject)
                    Right_branch_nodes=right_chil

            elif right_chil in parents and left_chil in leaves:
                if TreeObject.loc[tree_links[right_chil][0], 'nodes_labels'] == 'leaf_node' and TreeObject.loc[tree_links[right_chil][1], 'nodes_labels'] == 'leaf_node':
                    Right_branch_nodes=smallest_branch(right_chil, TreeObject)
                    Right_branch_nodes.append(right_chil)
                    Right_branch_nodes.sort()
                    Left_branch_nodes=left_chil
                else:
                    Right_branch_nodes=nodes_to_prune(right_chil, TreeObject)
                    Left_branch_nodes=left_chil
            
            elif right_chil in leaves and left_chil in leaves:
                Left_branch_nodes=left_chil
                Right_branch_nodes=right_chil
    
    return Left_branch_nodes, Right_branch_nodes









def get_nodes_with_equal_depth(TreeObject, Left_branch_nodes, Right_branch_nodes):
    '''
    Description:    This function finds the nodes for each depth of the tree
    
    Inputs:
    TreeObject:              It can be: 1) An object of class TreeStructure 2) An object of calss dataframe. More specifically the (...).IDs_Depths 
    Left_branch_nodes:       A list with the nodes ids of the left branch
    Right_branch_nodes:      A list with the nodes ids of the right branch
    
    Outputs:
    Same_depth:              A dictionary storing the nodes for each level of the whole tree
    Left_branch_same_depth:  A dictionary storing the nodes for each level of the left branch of the tree 
    Right_branch_same_depth: A dictionary storing the nodes for each level of the left branch of the tree
    '''
    if isinstance(TreeObject, TreeStructure):
        same_depth={}
        Left_branch_same_depth={}
        Right_branch_same_depth={}
        for depth in range(0,TreeObject.max_depth+1):    
            same_depth[depth]=TreeObject.IDs_Depths[TreeObject.IDs_Depths['Depth']==depth]
            if isinstance(Left_branch_nodes, np.int32) and isinstance(Right_branch_nodes, np.int32):
                Left_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id']==Left_branch_nodes] #Left_branch_nodes
                Right_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id']==Right_branch_nodes] #Right_branch_nodes
            elif isinstance(Left_branch_nodes, np.int32) and isinstance(Right_branch_nodes, list):
                Left_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id']==Left_branch_nodes] #Left_branch_nodes
                Right_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Right_branch_nodes)]
            elif isinstance(Left_branch_nodes, list) and isinstance(Right_branch_nodes, np.int32):
                Left_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Left_branch_nodes)]
                Right_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id']==Right_branch_nodes] #Right_branch_nodes
            elif isinstance(Left_branch_nodes, list) and isinstance(Right_branch_nodes, list):
                Left_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Left_branch_nodes)]
                Right_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Right_branch_nodes)]
            
    elif isinstance(TreeObject, pd.DataFrame):
        same_depth={}
        Left_branch_same_depth={}
        Right_branch_same_depth={}
        for depth in range(0,max(TreeObject.loc[:,'Depth'])+1):    
            same_depth[depth]=TreeObject[TreeObject['Depth']==depth]
            Left_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Left_branch_nodes)]
            Right_branch_same_depth[depth]=same_depth[depth].loc[same_depth[depth]['Id'].isin(Right_branch_nodes)]
        
    
    return same_depth, Left_branch_same_depth, Right_branch_same_depth

                        
                        
                        
                        
                        
                        
                        
                        
                        
#Define function to add coordinate columns to IDs_Depths pandas dataframe
def add_coordinates_columns(TreeObject, GraphObject, root_node_samples, x_y_coords_only=False, edges_only=False, tree_links=None):
    '''
    Description:  This function adds new columns to the pandas data frame of IDs and Depth. These columns contain the coordinates of the nodes and the edges.
    
    Inputs:       
    TreeObject:          It can be: 1) An object of class iDT.TreeStructure   2) A An object of class pandas dataframe object.
    Graph Structure:     It can be: 1) An object of class iDT.graph_structure 2) In case TreeObeject is pd.DataFrame Graph object should be an iDT.specify_feature_split_point 
    root_node_samples:   A number denoting the samples contained in the root node. This is helpful for estimating the edges widths.
    edges_only:          If true it should add columns only for edges coordinates. Default to False. (This is useful in the case of manually specifying feature split point in order to calculate the
                         edges coordinates of the merged tree to plot it on the original tree)
    tree_links:          A dictionary containing the links of the tree. Default to None
    '''
    if edges_only==False:
        if isinstance(TreeObject, TreeStructure):
            #Add columns for nodes labels
            TreeObject.IDs_Depths['nodes_labels']=TreeObject.feature_labels
            #Add column for nodesImpurities
            TreeObject.IDs_Depths['nodes_impurities']=list(TreeObject.Impurities.values())
            #Add colums for nodes thresholds
            TreeObject.IDs_Depths['nodes_thresholds']=TreeObject.Thresholds
            #Add columns for nodes values
            TreeObject.IDs_Depths['nodes_values']=list(TreeObject.Node_Values.values())
            #Add columns for nodes classes
            TreeObject.IDs_Depths['nodes_classes']=list(TreeObject.Node_classes.values())
            #Add columns with x & y nodes coordinates:
            TreeObject.IDs_Depths['x_coord']=GraphObject.Xn
            TreeObject.IDs_Depths['y_coord']=GraphObject.Yn

            #Construct the Edges x and y pairs coords
            for i_d in TreeObject.IDs_Depths['Id']:
                if i_d==0:
                    TreeObject.IDs_Depths.loc[i_d,'edges_x_coord'] = 'Root'
                    TreeObject.IDs_Depths.loc[i_d,'edges_y_coord'] = 'Root'
                    Links=TreeObject.Links[i_d]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[0],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[0],'y_coord']]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[1],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[1],'y_coord']]
                if i_d>0 and i_d in TreeObject.parents:
                    Links=TreeObject.Links[i_d]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[0],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[0],'y_coord']]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[1],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[1],'y_coord']] 
            denominator=root_node_samples/10
            for i_d_width in TreeObject.IDs_Depths['Id']:
                if i_d_width==0:
                    TreeObject.IDs_Depths.loc[i_d_width, 'edges_width'] = 'Root'
                elif i_d_width >0:
                    TreeObject.IDs_Depths.loc[i_d_width, 'edges_width'] = TreeObject.IDs_Depths.loc[i_d_width, 'Node_Samples']/denominator
                   

        elif isinstance(TreeObject, pd.DataFrame):
            if x_y_coords_only==False:
                #Add columns for nodes labels
                TreeObject['nodes_labels']=GraphObject.Nodes_dict['nodes_labels']
                #Add column for nodesImpurities
                TreeObject['nodes_impurities']=list(GraphObject.Nodes_dict['nodes_impurities'].values())
                #Add colums for nodes thresholds
                TreeObject['nodes_thresholds']=GraphObject.Nodes_dict['nodes_thresholds']
                #Add columns for nodes values
                TreeObject['nodes_values']=list(GraphObject.Nodes_dict['nodes_values'].values())
                #Add columns for nodes classes
                TreeObject['nodes_classes']=list(GraphObject.Nodes_dict['nodes_classes'].values())
                #Add columns with x & y nodes coordinates:
                TreeObject['x_coord']=GraphObject.graph_str.Xn
                TreeObject['y_coord']=GraphObject.graph_str.Yn

                #Construct the Edges x and y pairs coords
                for i_d in TreeObject['Id']:
                    if i_d==0:
                        TreeObject.loc[i_d,'edges_x_coord'] = 'Root'
                        TreeObject.loc[i_d,'edges_y_coord'] = 'Root'
                        Links=GraphObject.Edges_dict['links'][i_d]
                        TreeObject['edges_x_coord'][Links[0]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                        TreeObject['edges_y_coord'][Links[0]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                        TreeObject['edges_x_coord'][Links[1]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                        TreeObject['edges_y_coord'][Links[1]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']] 
                    if i_d>0 and i_d in GraphObject.Edges_dict['links'].keys():
                        Links=GraphObject.Edges_dict['links'][i_d]
                        TreeObject['edges_x_coord'][Links[0]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                        TreeObject['edges_y_coord'][Links[0]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                        TreeObject['edges_x_coord'][Links[1]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                        TreeObject['edges_y_coord'][Links[1]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']] 

                denominator=root_node_samples/10
                for i_d_width in TreeObject['Id']:
                    if i_d_width==0:
                        TreeObject.loc[i_d_width, 'edges_width'] = 'Root'
                    elif i_d_width >0:
                        TreeObject.loc[i_d_width, 'edges_width'] = TreeObject.loc[i_d_width, 'Node_Samples']/denominator
           
            elif x_y_coords_only==True:
                    #Add columns with x & y nodes coordinates:
                    TreeObject['x_coord']=GraphObject.Xn
                    TreeObject['y_coord']=GraphObject.Yn

                    #Construct the Edges x and y pairs coords
                    for i_d in TreeObject['Id']:
                        if i_d==0:
                            TreeObject.loc[i_d,'edges_x_coord'] = 'Root'
                            TreeObject.loc[i_d,'edges_y_coord'] = 'Root'
                            Links=tree_links[i_d]
                            TreeObject['edges_x_coord'][Links[0]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                            TreeObject['edges_y_coord'][Links[0]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                            TreeObject['edges_x_coord'][Links[1]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                            TreeObject['edges_y_coord'][Links[1]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']]
                        if i_d>0 and i_d in tree_links.keys():
                            Links=tree_links[i_d]
                            TreeObject['edges_x_coord'][Links[0]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                            TreeObject['edges_y_coord'][Links[0]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                            TreeObject['edges_x_coord'][Links[1]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                            TreeObject['edges_y_coord'][Links[1]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']] 

                    denominator=root_node_samples/10
                    for i_d_width in TreeObject['Id']:
                        if i_d_width==0:
                            TreeObject.loc[i_d_width, 'edges_width'] = 'Root'
                        elif i_d_width >0:
                            TreeObject.loc[i_d_width, 'edges_width'] = TreeObject.loc[i_d_width, 'Node_Samples']/denominator
                    
                    
    elif edges_only==True:
        if isinstance(TreeObject, TreeStructure):
            #Construct the Edges x and y pairs coords
            for i_d in TreeObject.IDs_Depths['Id']:
                if i_d==0:
                    TreeObject.IDs_Depths.loc[i_d,'edges_x_coord'] = 'Root'
                    TreeObject.IDs_Depths.loc[i_d,'edges_y_coord'] = 'Root'
                    Links=TreeObject.Links[i_d]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[0],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[0],'y_coord']]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[1],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[1],'y_coord']] 
                if i_d>0 and i_d in TreeObject.parents:
                    Links=TreeObject.Links[i_d]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[0],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[0]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[0],'y_coord']]
                    TreeObject.IDs_Depths['edges_x_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'x_coord'], TreeObject.IDs_Depths.loc[Links[1],'x_coord']] 
                    TreeObject.IDs_Depths['edges_y_coord'][Links[1]] = [TreeObject.IDs_Depths.loc[Links[2],'y_coord'], TreeObject.IDs_Depths.loc[Links[1],'y_coord']] 
            denominator=root_node_samples/10
            for i_d_width in TreeObject.IDs_Depths['Id']:
                if i_d_width==0:
                    TreeObject.IDs_Depths.loc[i_d_width, 'edges_width'] = 'Root'
                elif i_d_width >0:
                    TreeObject.IDs_Depths.loc[i_d_width, 'edges_width'] = TreeObject.IDs_Depths.loc[i_d_width, 'Node_Samples']/denominator

        elif isinstance(TreeObject, pd.DataFrame):
            #Construct the Edges x and y pairs coords
            for i_d in TreeObject['Id']:
                if i_d==0:
                    Links=GraphObject.Edges_dict['links'][i_d]
                    TreeObject['edges_x_coord'][Links[0]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                    TreeObject['edges_y_coord'][Links[0]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                    TreeObject['edges_x_coord'][Links[1]] = [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                    TreeObject['edges_y_coord'][Links[1]] = [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']] 
                if i_d>0 and i_d in GraphObject.Edges_dict['links'].keys():
                    Links=GraphObject.Edges_dict['links'][i_d]
                    TreeObject['edges_x_coord'][Links[0]]= [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[0],'x_coord']] 
                    TreeObject['edges_y_coord'][Links[0]]= [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[0],'y_coord']]
                    TreeObject['edges_x_coord'][Links[1]]= [TreeObject.loc[Links[2],'x_coord'], TreeObject.loc[Links[1],'x_coord']] 
                    TreeObject['edges_y_coord'][Links[1]]= [TreeObject.loc[Links[2],'y_coord'], TreeObject.loc[Links[1],'y_coord']]
            denominator=root_node_samples/10
            for i_d_width in TreeObject['Id']:
                if i_d_width==0:
                    TreeObject.loc[i_d_width, 'edges_width'] = 'Root'
                elif i_d_width >0:
                    TreeObject.loc[i_d_width, 'edges_width'] = TreeObject.loc[i_d_width, 'Node_Samples']/denominator
                    
        
                    
                    

                    
                    
                    
                    
                       
def format_graph(Nodes_dict, Edges_dict, criterion, classes_dict, nodes_coloring='Impurity', edges_shape='Lines', User_features_color_groups=None, Best_first_Tree_Builder=True):
    """
    Description: This function creates variables useful for the formatting of the DT plot. 
    
    Inputs:
    Nodes_dict:                  A dictionary conatining the following key-value pairs:
                                 'nr_vertices': An integer denoting the number of nodes in the DT
                                 'nodes_ids': List with the nodes_ids 
                                 'nodes_depths': List with the nodes depths
                                 'nodes_labels': List with the feature labels
                                 'nodes_impurities': Dictionary which has the nodes ids as keys and the corresponding nodes Impurities
                                 'nodes_thresholds': A numpy array with the nodes thresholds
                                 'nodes_samples': Dictionary which has the nodes ids as keys and the corresponding nodes samples
                                 'nodes_values': Dictionary which has the nodes ids as keys and the corresponding nodes values
                                 'nodes_classes': Dictionary which has the nodes ids as keys and the corresponding nodes classes
    Edges_dict:                  A dictionary containing the following key-value pairs:
                                 'links': A list containing sublists with the links among the various nodes
                                 'edge_seq': A list containing sublists with the edges sequences based on the links
    criterion:                   A string denoting the criterion used to make splits. It can be either 'gini' or 'entropy'
    classes_dict:                Dictionary containing the following key-value pairs:
                                 'Classes Labels': A list containing strings (or integers) corresponding to each class
                                 'Classes Colors': A list contining the colour code of each class
    nodes_coloring:              A string denoting which nodes colouring strategy to be used. It can be 'Impurity' for colouring nodes based on their Impurity, 'Classes' for colouring nodes based on their class and                                            'Features_color_groups' based on user defined features colour groups. Default to Impurity.
    edges_shape:                 A string denoting the shape of the edges. It can be either 'Lines' for using lines or 'Lines-steps' for lines with steps
    User_features_color_groups:  A dictionary with the following key-value pairs:
                                 'Groups & parameters': A dictionary which has the names of the groups as keys and list with the name of the inputs features that belong to that groups as values.
                                 'Colors of groups':    A dictionary which has the names of the groups as keys and the colour code for that groups as values.
    Best_first_Tree_Builder:     Whether the tree was built in Best first fashion. It can be either True or False. Default to True.
    
    Ouputs:
    Thres_dict:                  A dictionary containing the nodes ids as keys and strings as values. The strings will be of the form: "X<=Y" for non terminal nodes or a string denoting the class of the node if it is a                                        terminal node (leaf node).
    Hovertext:                   A dictionary containing the nodes ids as keys and strings as values. The strings will be of the form: ID: X, Impurity: Y, Node Samples: Z. These are the information that will appear when the                                  expert will hover over a node of the DT
    shape_dict:                  A dictionary with the nodes ids as keys and strings as values. The strings will be either 'circle' if the node is a leaf node or 'square' if the node is non leaf node.
    shape_lines_dict:            A list with a string. The string can be either 'Lines' or 'Lines-steps'
    color_impurities_dict:       A dictionary with keys from 1 to 4 and a color code as value for each key.
    color_classes_dict:          A dictionary with the names of the classes as keys and a color code as value for each key (class)
    color_visual_style:          It is a dictionary contining the information appearing in the legend. It can be either a dictionary for impurities or classes or for user defined groups. This is relevant for the legend of                                    the plot.
    subranges:                   A list containing the the values of impurities for the 4 different keys of color_impurities_dict.
    upper_limit:                 The upper_limit for the impurity if using the entropy criterion.
    """
    #Formatting the graph

    #Threshold dictionary
    thres_text="{}<={}"
    thres_text.format(Nodes_dict['nodes_labels'],Nodes_dict['nodes_thresholds'])

    Thres_dict={}
    for thres in range(0,len(Nodes_dict['nodes_thresholds'])):  
        if Nodes_dict['nodes_thresholds'][thres]!=-2:
            Thres_dict[thres] = [thres_text.format(Nodes_dict['nodes_labels'][thres],
                                                   np.round(Nodes_dict['nodes_thresholds'][thres], decimals=3))]
        else:
            Thres_dict[thres]= Nodes_dict['nodes_classes'][thres]

    #Hover information    
    hover_template="ID: {} <br>{}: {} <br>Node Samples: {}"
    Hovertext=[]
    for nd in range(0, Nodes_dict['nr_vertices']):
        Hovertext.append(hover_template.format(Nodes_dict['nodes_ids'][nd], criterion, np.round(Nodes_dict['nodes_impurities'][nd],decimals=3), Nodes_dict['nodes_samples'][nd]))


    #Formating the graph    
    shape_dict = {}
    for feat in range(0,Nodes_dict['nr_vertices']):     
        if Nodes_dict['nodes_labels'][feat]=='leaf_node' or Nodes_dict['nodes_labels'][feat]=='Make_leaf': #The Make_leaf condition is useful to enable the manual pruning
            shape_dict[feat] = "circle"   
        else:
            shape_dict[feat] = "square"   
            
    shape_lines_dict = []
    if edges_shape=='Lines':
        shape_lines_dict='linear'
    elif edges_shape=='Lines-steps':
        shape_lines_dict='hv'



    #Color dictionary based on Impurities
    color_impurities_dict={0:'rgb(247,247,247)',1:'rgb(204,204,204)', 2:'rgb(150,150,150)', 
                           3:'rgb(99,99,99)', 4: 'rgb(37,37,37)'}

    color_imp_visual_style={}
    if criterion == 'gini':
        if len(classes_dict['Classes Labels'])==0:
            upper_limit=max(Nodes_dict['nodes_impurities'].values()) 
        elif len(classes_dict['Classes Labels'])!=0:
            #For a different number of classes there is a different upper limit for Gini index: upper_limit=1-(1/no_of_classes)
            upper_limit=1-(1/len(classes_dict['Classes Labels']))
    elif criterion == 'entropy':
        if len(classes_dict['Classes Labels'])==0:
            upper_limit=max(Nodes_dict['nodes_impurities'].values())
        elif len(classes_dict['Classes Labels'])!=0:
            #For a different number of classes there is a different upper limit for entropy index: upper_limit=log2(no of classes)
            upper_limit=np.log2(len(classes_dict['Classes Labels']))

    #Create a list of evenly spaced numbers in the range 0 to upper limit
    subranges=list(np.around(np.linspace(0,upper_limit,6), 3))
    for cli in range(Nodes_dict['nr_vertices']):   
        if Nodes_dict['nodes_impurities'][cli] >=subranges[0] and Nodes_dict['nodes_impurities'][cli]<subranges[1]:
            color_imp_visual_style[cli] = color_impurities_dict[0]
        elif Nodes_dict['nodes_impurities'][cli] >=subranges[1] and Nodes_dict['nodes_impurities'][cli]<subranges[2]:
            color_imp_visual_style[cli] = color_impurities_dict[1]
        elif Nodes_dict['nodes_impurities'][cli] >=subranges[2] and Nodes_dict['nodes_impurities'][cli]<subranges[3]:
            color_imp_visual_style[cli] = color_impurities_dict[2]
        elif Nodes_dict['nodes_impurities'][cli] >=subranges[3] and Nodes_dict['nodes_impurities'][cli]<subranges[4]:
            color_imp_visual_style[cli] = color_impurities_dict[3]
        elif Nodes_dict['nodes_impurities'][cli] >=subranges[4] and Nodes_dict['nodes_impurities'][cli]<=subranges[5]:
            color_imp_visual_style[cli] = color_impurities_dict[4]                    


    #Color dictionary based on classes
    if len(classes_dict['Classes Labels']) == 0:
        color_classes_visual_style=color_imp_visual_style
        color_classes_dict={}
    elif len(classes_dict['Classes Labels']) != 0:
        color_classes_dict={}
        for clind in range(0, len(classes_dict['Classes Labels'])):
            color_classes_dict[classes_dict['Classes Labels'][clind]] = classes_dict['Classes Colors'][clind] 
        color_classes_visual_style={}
        for cl in range(Nodes_dict['nr_vertices']):   
            color_classes_visual_style[cl] = color_classes_dict[Nodes_dict['nodes_classes'][cl]]


    #Final nodes color dictionary based on user preference
    if nodes_coloring=='Impurity':
        color_visual_style=color_imp_visual_style
    elif nodes_coloring=='Classes':
        color_visual_style=color_classes_visual_style
    elif nodes_coloring=='Features_color_groups':
        usercolorgroups_visual_style={}
        for clu in range(Nodes_dict['nr_vertices']):    
            for gr in User_features_color_groups['Groups & parameters'].keys():
                if Nodes_dict['nodes_labels'][clu] in User_features_color_groups['Groups & parameters'][gr]:
                    group=gr
                    usercolorgroups_visual_style[clu] = User_features_color_groups['Colors of groups'][group]
                elif Nodes_dict['nodes_labels'][clu]=='leaf_node':
                    usercolorgroups_visual_style[clu] = 'white'
                    color_visual_style=usercolorgroups_visual_style 
    
    denominator=Nodes_dict['nodes_samples'][0]/10
    if Best_first_Tree_Builder==True:
        edges_widths=[Nodes_dict['nodes_samples'][i[1]]/denominator for i in Edges_dict['edge_seq']]    #nodes_samples[i]/100 for i in range(1, len(nodes_samples))
    elif Best_first_Tree_Builder==False:
        edges_widths=[Nodes_dict['nodes_samples'][i[1]]/denominator for i in Edges_dict['edge_seq']]              
                
    
    return Thres_dict, Hovertext, shape_dict, shape_lines_dict, color_impurities_dict, color_classes_dict, color_visual_style, subranges, upper_limit








                    
def add_legend(figure, Nodes_object, graph_structure, color_impurities_dict, subranges, color_classes_dict, shape_dict, nodes_coloring='Impurity', User_features_color_groups=None, mrk_size=10, txt_size=10):
    '''
    This function adds legent to plotly graph.
    
    Inputs:
    figure:                 An object of class plotly.FigureWidget
    Nodes_object            It can be: 1) An object of class Nodes (which is a dictionary storing the necessary information for the nodes) 2) pd.DataFrame containing information for the nodes
    graph_structure:        An object of class graph structure in case Nodes_object is of class Nodes. If Nodes_object is of class pd.DataFrame then set its value to None. 
    nodes_coloring:         A string that specifies the user preference for nodes coloring ('Impurity', 'Classes', 'Features_color_groups')
    color_impurities_dict:  A dictionary containing the necessary colour codes for the various impurity subranges 
    subranges:              A list containing numbers that denote the limits for each subrange (e.g. [0, 0.2, 0.4 ...] =>  subrange 1 will be 0-0.2, subrange 2 will be 0.2-0.4 etc)
    color_classes_dict:     Dictionary containing: 1) A dictionary with key Classes Labels where the names of the groups are stored in a list (the names are in string format) 
                                                   2) A dictionary with key Classes Colors where the names of the colors of each group are stored in a list (the names of the colors are in string format) 
    '''
    #Create additional traces in order to enable plot the appropriate legend
    if isinstance(Nodes_object, Nodes):
        if nodes_coloring=='Impurity':
            Impurity_ranges = {0: np.arange(subranges[0],subranges[1],0.001), 1: np.arange(subranges[1],subranges[2],0.001), 2: np.arange(subranges[2],subranges[3],0.001), 
                               3: np.arange(subranges[3],subranges[4],0.001), 4: np.arange(subranges[4],subranges[5],0.001)}
            impurity_text_template= 'Impurity: {} - {}'
            Impurity_ranges_labels = {0: impurity_text_template.format(subranges[0], subranges[1]), 1: impurity_text_template.format(subranges[1], subranges[2]), 
                                      2: impurity_text_template.format(subranges[2], subranges[3]), 3: impurity_text_template.format(subranges[3], subranges[4]), 
                                      4: impurity_text_template.format(subranges[4], subranges[5])}

            Impurity_groups = {'Ranges': Impurity_ranges, 'Ranges_labels': Impurity_ranges_labels, 'Colors of ranges': color_impurities_dict}
            for key_im_group in Impurity_groups['Ranges'].keys():
                for nd_im in range(0, len(Nodes_object.Nodes['nodes_impurities'])):
                    if Nodes_object.Nodes['nodes_impurities'][nd_im] >= min( Impurity_groups['Ranges'][key_im_group]) and Nodes_object.Nodes['nodes_impurities'][nd_im] <= max(Impurity_groups['Ranges'][key_im_group]):
                        #Check for manual pruning
                        if Nodes_object.Nodes['nodes_labels'][nd_im]=='Pruned':
                            nd_im+=1
                        else:
                            figure.add_trace(go.Scatter(x=[graph_structure.Xn[nd_im]],
                                                        y=[graph_structure.Yn[nd_im]],
                                                        name=Impurity_groups['Ranges_labels'][key_im_group],
                                                        mode='markers',
                                                        marker=dict(symbol='square',
                                                                    size=mrk_size,
                                                                    color=Impurity_groups['Colors of ranges'][key_im_group]),
                                                        showlegend=True))
                            break

        elif nodes_coloring=='Classes':
            for key_class_group in color_classes_dict.keys():
                for nd_class in Nodes_object.Nodes['nodes_classes'].keys():
                    if Nodes_object.Nodes['nodes_classes'][nd_class] == key_class_group:
                        #Check for manual pruning
                        if Nodes_object.Nodes['nodes_labels'][nd_class] == 'Pruned':
                            nd_class+=1
                        else:
                            figure.add_trace(go.Scatter(x=[graph_structure.Xn[nd_class]],
                                                        y=[graph_structure.Yn[nd_class]],
                                                        name=Nodes_object.Nodes['nodes_classes'][nd_class],
                                                        mode='markers',
                                                        marker=dict(symbol=shape_dict[nd_class],
                                                        size=mrk_size,
                                                        color=color_classes_dict[key_class_group]),
                                                        showlegend=True))
                            break


        elif nodes_coloring=='Features_color_groups':
            for key_group in User_features_color_groups['Colors of groups'].keys():
                for lab in range(0, len(Nodes_object.Nodes['nodes_labels'])):
                    if Nodes_object.Nodes['nodes_labels'][lab] in User_features_color_groups['Groups & parameters'][key_group]:
                        if Nodes_object.Nodes['nodes_labels'][lab] == 'Pruned':
                            lab+=1
                        else:
                            figure.add_trace(go.Scatter(x=[graph_structure.Xn[lab]],
                                                        y=[graph_structure.Yn[lab]],
                                                        name=key_group,
                                                        mode='markers',
                                                        marker=dict(symbol='square',
                                                                    size=mrk_size,
                                                                    color=User_features_color_groups['Colors of groups'][key_group]),
                                                        showlegend=True))
                            break 

        
        figure.update_layout(legend = dict(font={'size': txt_size}))
        
    elif isinstance(Nodes_object, pd.DataFrame):
        if nodes_coloring=='Impurity':
            Impurity_ranges = {0: np.arange(subranges[0],subranges[1],0.001), 1: np.arange(subranges[1],subranges[2],0.001), 2: np.arange(subranges[2],subranges[3],0.001), 
                               3: np.arange(subranges[3],subranges[4],0.001), 4: np.arange(subranges[4],subranges[5],0.001)}
            impurity_text_template= 'Impurity: {} - {}'
            Impurity_ranges_labels = {0: impurity_text_template.format(subranges[0], subranges[1]), 1: impurity_text_template.format(subranges[1], subranges[2]), 
                                      2: impurity_text_template.format(subranges[2], subranges[3]), 3: impurity_text_template.format(subranges[3], subranges[4]), 
                                      4: impurity_text_template.format(subranges[4], subranges[5])}

            Impurity_groups = {'Ranges': Impurity_ranges, 'Ranges_labels': Impurity_ranges_labels, 'Colors of ranges': color_impurities_dict}
            for key_im_group in Impurity_groups['Ranges'].keys():
                for nd_im in range(0, len(Nodes_object.loc[:, 'nodes_impurities'])):
                    if Nodes_object.loc[nd_im, 'nodes_impurities'] >= min( Impurity_groups['Ranges'][key_im_group]) and Nodes_object.loc[nd_im, 'nodes_impurities'] <= max(Impurity_groups['Ranges'][key_im_group]):
                        #Check for manual pruning
                        if Nodes_object.loc[nd_im, 'nodes_labels']=='Pruned':
                            nd_im+=1
                        else:
                            figure.add_trace(go.Scatter(x=[Nodes_object.loc[nd_im, 'x_coord']],
                                                        y=[Nodes_object.loc[nd_im, 'y_coord']],
                                                        name=Impurity_groups['Ranges_labels'][key_im_group],
                                                        mode='markers',
                                                        marker=dict(symbol='square',
                                                                    size=mrk_size,
                                                                    color=Impurity_groups['Colors of ranges'][key_im_group]),
                                                        showlegend=True))
                            break

        elif nodes_coloring=='Classes':
            for key_class_group in color_classes_dict.keys():
                for nd_class in Nodes_object.loc[:, 'Id']:
                    if Nodes_object.loc[nd_class, 'nodes_classes']==key_class_group:
                        #Check for manual pruning
                        if Nodes_object.loc[nd_class, 'nodes_labels']=='Pruned':
                            nd_class+=1
                        else:
                            figure.add_trace(go.Scatter(x=[Nodes_object.loc[nd_class, 'x_coord']],
                                                        y=[Nodes_object.loc[nd_class, 'y_coord']],
                                                        name=Nodes_object.loc[nd_class, 'nodes_classes'],
                                                        mode='markers',
                                                        marker=dict(symbol=shape_dict[nd_class],
                                                                    size=mrk_size,
                                                                    color=color_classes_dict[key_class_group]),
                                                                    showlegend=True))
                            break


        elif nodes_coloring=='Features_color_groups':
            for key_group in User_features_color_groups['Colors of groups'].keys():
                for lab in range(0, len(Nodes_object['nodes_labels'])):
                    if Nodes_object.loc[lab, 'nodes_labels'] in User_features_color_groups['Groups & parameters'][key_group]:
                        if Nodes_object.loc[lab, 'nodes_labels'] == 'Pruned':
                            lab+=1
                        else:
                            figure.add_trace(go.Scatter(x=[Nodes_object.loc[lab, 'x_coord']],
                                                        y=[Nodes_object.loc[lab, 'y_coord']],
                                                        name=key_group,
                                                        mode='markers',
                                                        marker=dict(symbol='square',
                                                                    size=mrk_size,
                                                                    color=User_features_color_groups['Colors of groups'][key_group]),
                                                        showlegend=True))
                            break
        
        figure.update_layout(legend = dict(font={'size': txt_size}))

                            
                            
                            
                            
                            
                            
                            
                            
def check_labels(TreeObject):
    '''
    Description: This function looks for missing labels in the tree nodes of a manually modified tree and stores them in a list. 
                 This is a helper function used in class specify_feature_point_to_split
    
    Inputs:
    TreeObject:     It can be: 1) an object of class TreeStructure 2) an object of class pd.DataFrame
    
    Outputs:
    missing_labes:  A list containing the labels of classes that are not included in this tree
    '''
    missing_labels = []
    for lbl in TreeObject.classes_dict['Classes Labels']:
        if isinstance(lbl, str):
            if lbl not in list(TreeObject.OutputsNodesData[0]):    #It was: if int(lbl) not in list(TreeObject.OutputsNodesData[0]):
                missing_labels.append(lbl)
        if isinstance(lbl, int):
            if lbl not in list(TreeObject.OutputsNodesData[0]):    #It was: if int(lbl) not in list(TreeObject.OutputsNodesData[0]):
                missing_labels.append(lbl)
    
    return missing_labels









def fill_missing_classes_nodes_values(TreeObject, missing_labels, classes_dict):
    '''
    Description: This function will fill with zeros the list of nodes_values of nodes that contain reduced classes. For example, normally nodes_values should be [x, y, z, w] but in some nodes
    in modified tree we have [x,z,w]. In these nodes this function will turn them into: [x, 0, z, w] for consistency.
    
    Inputs: 
    TreeObject:       It can be: 1) an object of class TreeStructure 2) an object of class pd.DataFrame
    missing_labels:   A list containing the labels of classes that are not included in some nodes
    classes_dict:     Dictionary storing the classes_labels and the corresponding colors
    '''
    if isinstance(TreeObject, iDT.TreeStructure):
        if len(missing_labels)>0:
            for nd in list(TreeObject.IDs_Depths.loc[:,'Id']):
                if len(TreeObject.Node_Values[nd])<len(classes_dict['Classes Labels']):
                    for mis in range(0, len(missing_labels)):
                        init_nd_val=list(TreeObject.Node_Values[nd])
                        index=classes_dict['Classes Labels'].index(missing_labels[mis])
                        init_nd_val.insert(index, 0)
                        TreeObject.Node_Values[nd]=np.array(init_nd_val)
    
    if isinstance(TreeObject, pd.DataFrame):
        if len(missing_labels)>0:
            for nd in list(TreeObject.loc[:,'Id']):
                if len(TreeObject.loc[nd,'nodes_values'])<len(classes_dict['Classes Labels']):
                    for mis in range(0, len(missing_labels)):
                        init_nd_val=list(TreeObject.Node_Values[nd])
                        index=classes_dict['Classes Labels'].index(missing_labels[mis])
                        init_nd_val.insert(index, 0)
                        TreeObject.loc[nd,'nodes_values']=np.array(init_nd_val)

                    
                    
                    
                    
                    
                    
                    
        
def change_feature_or_threshold_to_split(NodesData, OutputsNodesData, node_id, feature, new_threshold, criterion='gini',
                                         splitter='best', max_depth_left=None, max_depth_right=None, 
                                         min_samples_split_left=2, min_samples_split_right=2, min_samples_leaf_left=1,
                                         min_samples_leaf_right=1, min_weight_fraction_leaf_left=0.0,
                                         min_weight_fraction_leaf_right=0.0, max_features=None, random_state=0, 
                                         max_leaf_nodes_left=None, max_leaf_nodes_right=None, 
                                         min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                         ccp_alpha=0.0):    
    """
    Description:
    This function enables the user to manually make changes in the threshold/split points of nodes. The function
    will create two subdatasets one for the left branch (data<=new_threshold) and one for the right branch
    (data>new_threshold). Then, it will classify and fit a new tree (according to the tree parameters specified by
    the user) for each branch.

    Inputs:
    NodesData:        A dictionary which stores the input data for each node (of the classified tree). This can be 
                      retrieved using the function tree_nodes_data.
    OutputsNodesData: A dictionary which stores the output data for each node (of the classified tree). This can be 
                      retrieved using the function tree_nodes_data_outputs.
    node_id:          An integer number specifying the node the user wants to change the threshold/split point.
    feature:          A string indicating the new feature to be used for splitting. 
    new_threshold:    A number indicating the new threshold/split point.

    parameters:       The following parameters: criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                      min_weight_fraction_leaf, max_features, random_state,max_leaf_nodes, min_impurity_decrease,
                      min_impurity_split, class_weight, ccp_alpha Are parameters that are used in 
                      tree.DecisionTreeClassifier and the default values are the same as tree.DecisionTreeClassifier.
                      For more information see:
                      https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

                      However, the user is able to use different values for the left and right subtrees for the
                      following parameters: max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                      max_leaf_nodes.

    Outputs:
    left_subtree:         The classified tree for the left branch based on the new threshold
    right_subtree:        The classified tree for the left branch based on the new threshold
    left_branch:          A list containing the data for the left branch.
    left_branch_outputs:  A list containing the output data for the left branch
    right_branch:         A list containing the data for the right branch.
    right_branch_outputs: A list containing the output data for the right branch outputs.
    """


    #Retrieve the data for the node we wanna change threshold
    sub_data=NodesData[node_id]
    sub_outputs=OutputsNodesData[node_id]

    #Create the data for the left branch based on the user specified threshold
    left_branch=sub_data[sub_data[feature]<=new_threshold]
    left_branch_outputs=sub_outputs[sub_data[feature]<=new_threshold]
    #Create the data for the right branch based on the user specified threshold
    right_branch=sub_data[sub_data[feature]>new_threshold]
    right_branch_outputs=sub_outputs[sub_data[feature]>new_threshold]

    #Create the left subtree
    left_subtree=tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, 
                                             max_depth=max_depth_left, min_samples_split=min_samples_split_left,
                                             min_samples_leaf=min_samples_leaf_left, 
                                             min_weight_fraction_leaf=min_weight_fraction_leaf_left, 
                                             max_features=max_features, random_state=random_state,
                                             max_leaf_nodes=max_leaf_nodes_left, 
                                             min_impurity_decrease=min_impurity_decrease,
                                             min_impurity_split=min_impurity_split, class_weight=class_weight,
                                             ccp_alpha=ccp_alpha)
    #Fit the left subtree
    left_subtree.fit(left_branch,left_branch_outputs)

    #Create the right subtree
    right_subtree=tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, 
                                              max_depth=max_depth_right, min_samples_split=min_samples_split_right,
                                              min_samples_leaf=min_samples_leaf_right, 
                                              min_weight_fraction_leaf=min_weight_fraction_leaf_right, 
                                              max_features=max_features, random_state=random_state,
                                              max_leaf_nodes=max_leaf_nodes_right, 
                                              min_impurity_decrease=min_impurity_decrease,
                                              min_impurity_split=min_impurity_split, class_weight=class_weight,
                                              ccp_alpha=ccp_alpha)
    #Fit the right subtree
    right_subtree.fit(right_branch,right_branch_outputs)
    return left_subtree, right_subtree, left_branch, left_branch_outputs, right_branch, right_branch_outputs









def change_class(node_id, TreeObject, new_class):
    '''
    Description:
    This function enables the user to manually change the class (label) in a leaf node
    
    Inputs:
    node_id:      The id indicating leaf node the expert wants to change its class
    TreeObject:   It can be: 1) an object of class TreeStructure 2) an object of class pd.DataFrame
    new_class:    The new class to be assigned to the leaf_node. The new class should be in the same format as the outputs object of the dataset.
    
    Outputs:
    The TreeObject with the user changes incorporated
    '''
    
    if isinstance(TreeObject, TreeStructure):
        #Check for invalid user inputs:
        #Check if the node_id given by the user is a leaf_node
        if TreeObject.feature_labels[node_id] != 'leaf_node':
            raise ValueError('The node id must be a leaf node. A test node id is given') from None
        else:
            TreeObject.Node_classes[node_id] = new_class
    
    if isinstance(TreeObject, pd.DataFrame):
        #Check for invalid user inputs:
        #Check if the node_id given by the user is a leaf_node
        if TreeObject.loc[node_id,'nodes_labels'] != 'leaf_node':
            raise ValueError('The node id must be a leaf node. A test node id is given') from None  
        else:
            TreeObject.loc[node_id,'nodes_classes'] = new_class
        
            








class Data_Preprocessing:
    '''
    This class contains methods that enable user to prepare and precprocess the dataset for the Interactive construction and analysis of decision trees.
    '''
        
    def Data_Preparation(filename, header=0, index_col=0, sep =',', train_test_splitting=True, test_size=0.25, random_state=None):
        '''
        Description:    This function takes a csv file and prepares the dataset in order for the interactive functions to be applicable.
        
        Inputs:
        filename        The name of the file with the format ending (e.g. 'inputs.csv'). The file should be in the same working directory. 
        header          Which row to use as column labels. Default to 0.
        index_col       which column to use as counter of rows. Default to 0.
        test_size       What percentage of the dataset should be used as test set. Default to 0.25
        random_state    Random state. Default to None. Setting the random state to a number (integer) will force the algorithm to get the same results every time you run the code.
        
        Outputs: 
        x:                            The input dataset. This refers to the available features. It can contain the whole number of instances (all the rows of the dataset) or if train-test splitting is 
                                      true, then it is the training input dataset and it contains a part of the dataset.
        y:                            The output dataset. This refers to the target colum. It can contain the whole number of instances (all the rows of the dataset) or if train-test splitting is true,
                                      then it is the training output dataset and it contains a part of the dataset.  
        z:                            If train-test splitting is true, then this is the testing input dataset. Otherwise it will be an empty list.
        w:                            If train-test splitting is true, then this is the testing output dataset. Otherwise it will be an empty list.
        features:                     This is a list with the names of the availables features in the dataset.
        outputs_complete_dataset:     This is the complete output dataset
        inputs_complete_dataset:      This is the complete input dataset
        outputs_train:                This is the training output dataset ( I am not sure this is necessary)           
        '''
        
        if isinstance(filename, str):
            #Import the whole dataset
            data_table=pd.read_csv(filename, header=None if header==-1 else header, index_col=None if index_col==-1 else index_col, sep =',')
        elif isinstance(filename, pd.DataFrame):
            data_table=filename

        #Separate the dataset into features and targets
        outputs = data_table.iloc[:,-1] #The target (or else class or label), is what we want to predict. This line will keep the target which should be the last column of the dataset.
        inputs = data_table.iloc[:,:-1] #features are all the columns the model uses to make a prediction. This line will keep all columns except the last one
        parameters = list(data_table.columns[:-1]) #Keeps the names of the features except the last column which should be the classes
        
        #Split the dataset
        if train_test_splitting==True:
            #Split in training and test set
            X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=random_state)
            x=X_train
            y=y_train
            z=X_test
            w=y_test
            x_complete=inputs
            y_complete=outputs
        
        elif train_test_splitting==False:
            x=inputs
            y=outputs
            z=[]
            w=[]
            x_complete=inputs
            y_complete=outputs

        data_dict={'x': x, 'y': y, 'z': z, 'w': w, 'features': parameters, 
                   'outputs_complete_dataset': y_complete, 'inputs_complete_dataset': x_complete,
                   'outputs_train': y if train_test_splitting else None}
        return data_dict
        
    def features_color_groups():
        '''
        Description: 
        This function will ask the user to define a dictionary where the available features will be divided in groups and each group will be assigned with a specific colour. 
        While using this function the user will be asked to:
        1) Define the number of the groups the available features will be separated (max number of groups=12)
        2) Give the name of each group.
        1) Assign the available features to the appropriate group
        '''
        n_groups=int(input('Assign the number of the groups: '))
        if n_groups>12:
            raise Exception('Number of groups larger than the maximum allowed number of groups 12')
        groups_names=[]
        names_template="Name group {}: "
        for i in range(0,n_groups):
            print(names_template.format(i))
            groups_names.append(input())
        
        groups_colors_pallete=['rgb(166,206,227)','rgb(31,120,180)','rgb(178,223,138)','rgb(51,160,44)',
                               'rgb(251,154,153)','rgb(227,26,28)','rgb(253,191,111)','rgb(255,127,0)',
                               'rgb(202,178,214)','rgb(106,61,154)','rgb(255,255,153)','rgb(177,89,40)']
        groups_colors={}
        for cl in range(0, n_groups):
            groups_colors[groups_names[cl]]=groups_colors_pallete[cl]
        
        #Now the user needs to assign the variables to the corresponding groups
        groups_features={}
        features_groups_template="Assign features to group {} (space separated)"
        for vr in range(0, len(groups_names)):
            print(features_groups_template.format(groups_names[vr]))
            groups_features[groups_names[vr]]=input()  
            groups_features[groups_names[vr]]=groups_features[groups_names[vr]].split()
        
        features_color_groups_dict={'Groups & parameters': groups_features,
                                      'Colors of groups': groups_colors}
        return features_color_groups_dict
    
    def define_classes():
        '''
        Descriptions:
        This function asks the user to define the names of the classes of the case study.
        '''
        n_classes=int(input('Assign the number of the classes: '))
        classes_names=[]
        classes_template="Write the label of class {}: "
        for i in range(0,n_classes):
            print(classes_template.format(i))
            classes_names.append(input())
            
        return classes_names










class TreeStructure:
    '''
        Description:
        This class will calculate and store the basic attributes of the Tree:
        1) Number of nodes appearing in the classified Tree
        2) The right children of the tree
        3) The left children of the tree
        4) The thresholds (split points) of each node
        5) The ids of the features that were used for splitting in each node
        6) The ids of the nodes that are parents in the tree
        7) The ids of the nodes that are leaves in the tree
        8) The links (relationship: which are the children of each parent node) of the tree
    
        
        
        Inputs: 
        Tree:                          The classified tree
        features:                      The various features of the dataset
        X:                             The Input data that were used to classify the tree
        Y:                             The output data (classes)
        classes_labels:                A list containing the user defined names of classes [class_1_label, class_2_label]
        print_rules:                   If true the rules based on which the tree was constructed will be printed. 
                                       If false the rules wont be printed. Default to False.
        Best_first_Tree_Builder=True:  Whether the tree was built in a Best First or Depth First manne. It takes
                                       True or False values. Default to True.
                    
    '''
    def __init__(self, Tree, features, X, Y, outputs_train, classes_dict, print_rules=False, Best_first_Tree_Builder=True):
        
        self.Tree=Tree                         #The classified tree. Object of cass sklearn.tree.DecisionTreeClassifier, sklearn.tree.fit                                 
        self.features=features                 #A list containing the names of the various features as strings
        self.X=X                               #Pandas data frame object which contains the input data used to classify the tree (most of the times it will be the
                                               #training dataset)
        self.Y=Y                               #Pandas Dataframe (series) containing the whole output dataset
        self.classes_dict=classes_dict         #A dictionary containing the user defined classes labels and colors
        self.outputs_train=outputs_train       #Pandas Dataframe object (series) containing the output training dataset

        
#         This coding block will store the basic attributes of the Tree:
#         1) Number of nodes appearing in the classified Tree
#         2) The right children of the tree
#         3) The left children of the tree
#         4) The thresholds (split points) of each node
#         5) The ids of the features that were used for splitting in each node
#         6) The criterion (gini or entropy) used to evaluate the impurity of each node
#         7) The maximum depth of the tree

        self.n_nodes = Tree.tree_.node_count
        self.Children_right=Tree.tree_.children_right
        self.Children_left=Tree.tree_.children_left
        self.Thresholds=Tree.tree_.threshold
        self.feat_id = Tree.tree_.feature
        self.criterion = Tree.criterion
        self.max_depth = Tree.tree_.max_depth

#         In this coding block the appropriate parameter labels will be assigned to the corresponding id features 
#         for each node.
        
        self.feature_labels=[]
        for l in range(0,len(self.feat_id)):
            if self.feat_id[l]<0:
                self.feature_labels.append('leaf_node')
            else:
                self.feature_labels.append(features[self.feat_id[l]])
        
#         In this coding block the samples contained in each node of the tree will be stored in a dictionary.

        self.Node_samples={}
        for node in range(0,self.n_nodes):
            self.Node_samples[node]=Tree.tree_.n_node_samples[node]    

#         In this coding block the impurities of each node are stored in a dictionary.
        
        self.Impurities={}
        for Imp in range(0, self.n_nodes):
            self.Impurities[Imp]=round(Tree.tree_.impurity[Imp],3)

            
#         In this coding block the values of each node will be stored in a dictionary
#         *The values is a two element list. The first element of the list shown the number of instances
#         belonging to class 1 and the second element of the list shows the number of instances belonging
#         to class 2.

        self.Node_Values={}
        for Val in range(0, self.n_nodes):
            self.Node_Values[Val]=Tree.tree_.value[Val][0]
            
#         In this coding block the class of each node will be calculated and stored in a dictionary. This is for binary classes.
    
        self.Node_classes={}
        if len(self.classes_dict['Classes Labels'])==0:
            for cl in range(0,self.n_nodes):
                self.Node_classes[cl]='Unknown'
        elif len(self.classes_dict['Classes Labels'])!=0:
            for cl in range(0,self.n_nodes):
                self.Node_classes[cl]=self.classes_dict['Classes Labels'][np.argmax(self.Node_Values[cl])]

#         This coding block will print the rules based on which the tree was built
#         Moreover, the nodes ids and their corresponding depths will be stored.

        node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            
            # If we have a test node
            if (self.Children_left[node_id] != self.Children_right[node_id]):
                stack.append((self.Children_left[node_id], parent_depth + 1))
                stack.append((self.Children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
    
        if print_rules==True:
            print("The binary tree structure has %s nodes and has "
                  "the following tree structure:"
                  % self.n_nodes)
            for i in range(self.n_nodes):
                if is_leaves[i]:
                    print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
                else:
                    print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                          "node %s."
                          % (node_depth[i] * "\t",
                          i,
                          self.Children_left[i],
                          self.feat_id[i],
                          self.Thresholds[i],
                          self.children_right[i],
                          ))
                    #Create a dataframe with the nodes ids and their corresponding depth
                    ids=np.array(range(0,self.n_nodes)).T
                    depth=np.array(node_depth).T
                    self.IDs_Depths = pd.DataFrame({'Id': ids, 'Depth': depth})
                    print()
            #return df
        else:
            ids=np.array(range(0,self.n_nodes)).T
            depth=np.array(node_depth).T
            self.IDs_Depths = pd.DataFrame({'Id': ids, 'Depth': depth, 'Node_Samples': np.array(list(self.Node_samples.values()))})  

#         In this coding block the parent and leaves nodes (ids) are stored in a list
        
        #Import necessary libs to help us identify leaves
        self.parents=np.zeros(len(Tree.tree_.threshold))      #Pre-allocate necessary variables
        self.leaves=np.zeros(len(Tree.tree_.threshold))       #Pre-allocate necessary variables
        
        for thres in range(0,len(Tree.tree_.threshold)):
            #Look for parents first
            if Tree.tree_.threshold[thres]!=TREE_UNDEFINED:
                self.parents[thres]=thres
            elif Tree.tree_.threshold[thres] == TREE_UNDEFINED:
                self.leaves[thres]=thres
        root=[0] #Keep the root
        self.parents=self.parents[self.parents>0] #Keep only the parents (omit zeros)
        self.parents=np.concatenate((root,self.parents)) 
        self.leaves=self.leaves[self.leaves>0] #Keep only the leaves (ommit zeros)

        
        

#         In this coding block we will calculate what is the relationship among the various nodes of the tree and
#         store them in a dictionary               
        
        Links_inter={}   #Create an empty dictionary where we will assign the links for intermediate nodes with 
                         #their parents
        self.Links={}    #Create an empty dictionary where we will assign the links for all nodes
        
        #Create a dictionary to store for each depth the nodes ids contained in it
        same_depth={}
        for depth in range(0,self.max_depth+1):    
            same_depth[depth]=self.IDs_Depths[self.IDs_Depths['Depth']==depth]
        #Create a list containing the depths of the tree
        depths=list(same_depth.keys())
        
        if Best_first_Tree_Builder==True:
            #Start establishing the links for each node of the tree
            #Look at the ids of parent nodes
            for par in range(0, len(depths)-1):
                #Look at at the nodes included in depth par
                chil_list_one=list(same_depth[par+1].loc[:,'Id'])
                for ndi in chil_list_one:
                    #Look at the nodes with id higher than the previously selected one
                    chil_list_two=list(same_depth[par+1].loc[ndi+1:,'Id'])
                    for ndd in chil_list_two: 
                        if (#Make sure that the children have subsequent ids:
                            self.IDs_Depths.loc[ndd,'Id']-self.IDs_Depths.loc[ndi,'Id']==1):
                            ids_list=list(same_depth[par].loc[:,'Id'])
                            #Look for parents
                            for nds in ids_list: 
                                if (#The sum of the samples of children nodes need to be equal to the parent one
                                    self.Node_samples[self.IDs_Depths.loc[ndi,'Id']]+self.Node_samples[self.IDs_Depths.loc[ndd,'Id']]==self.Node_samples[self.IDs_Depths.loc[nds,'Id']]
                                    #Make sure that the children cannot be at a higher depth than the parent (remember depth=1 higher than depth=2)
                                    and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndi,'Depth'] and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndd,'Depth']
                                    #Make sure that the depth differennce of the parent with the children nodes is equal to 1:
                                    and self.IDs_Depths.loc[ndi,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1 and self.IDs_Depths.loc[ndd,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1
                                    #Make sure that the sum of the node_values of each class of the children is equal to the node_values of the corresponding class of the parent
                                    and self.Node_Values[ndi][0]+self.Node_Values[ndd][0]==self.Node_Values[nds][0]
                                    and self.Node_Values[ndi][1]+self.Node_Values[ndd][1]==self.Node_Values[nds][1]):                     
                                    Links_inter[nds]=[ndi,ndd,nds]

                #Concatenate the links dictionaries into one
                self.Links=Links_inter


                #Check if there are leaves wrongly placed as parents. If so delete them
                keys= []
                for l in self.Links.keys():
                    keys.append(l)

                for k in keys:
                    if self.Links[k][2] in self.leaves:
                       del self.Links[k]
            
                            
        
        elif Best_first_Tree_Builder==False:
            for par in self.parents: 
                #Look at the nodes with id higher than the selected parent
                for ndi in self.IDs_Depths.loc[par:len(self.IDs_Depths),'Id']:
                    #Look at the nodes with id higher than the previously selected one
                    for ndd in self.IDs_Depths.loc[ndi+1:len(self.IDs_Depths),'Id']: 
                       if (#Children nodes need to be at the same depth
                           self.IDs_Depths.loc[ndi,'Depth']==self.IDs_Depths.loc[ndd,'Depth']):                  
                           #Look for parents
                           for nds in self.IDs_Depths.loc[0:len(self.IDs_Depths),'Id']:
                               if (#In case the two children are leaves they need to have subsequent ids
                                    self.IDs_Depths.loc[ndi,'Depth']==Tree.tree_.max_depth and self.IDs_Depths.loc[ndd,'Depth']==Tree.tree_.max_depth
                                    #The sum of the samples of children nodes need to be equal to the parent one
                                    and self.Node_samples[self.IDs_Depths.loc[ndi,'Id']]+self.Node_samples[self.IDs_Depths.loc[ndd,'Id']]==self.Node_samples[self.IDs_Depths.loc[nds,'Id']]
                                    #Make sure that the children cannot be at a higher depth than the parent (remember depth=1 higher than depth=2)
                                    and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndi,'Depth'] and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndd,'Depth']
                                    #Make sure that the depth differennce of the parent with the children nodes is equal to 1:
                                    and self.IDs_Depths.loc[ndi,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1 and self.IDs_Depths.loc[ndd,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1):
                                    Links_inter[nds]=[nds+1,nds+2,nds]  
                               elif (#The sum of the samples of children nodes need to be equal to the parent one
                                    self.Node_samples[self.IDs_Depths.loc[ndi,'Id']]+self.Node_samples[self.IDs_Depths.loc[ndd,'Id']]==self.Node_samples[self.IDs_Depths.loc[nds,'Id']]
                                    #Make sure that the children cannot be at a higher depth than the parent (remember depth=1 higher than depth=2)
                                    and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndi,'Depth'] and self.IDs_Depths.loc[nds,'Depth']<self.IDs_Depths.loc[ndd,'Depth']
                                    #Make sure that the depth differennce of the parent with the children nodes is equal to 1:
                                    and self.IDs_Depths.loc[ndi,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1 and self.IDs_Depths.loc[ndd,'Depth']-self.IDs_Depths.loc[nds,'Depth']==1):                            
                                    #Store the links into a dictionary 
                                    Links_inter[nds]=[ndi,ndd,nds]  
                               
                                    
            #Concatenate the links dictionaries into one
            self.Links=Links_inter
            
            #Check if there are leaves wrongly placed as parents. If so delete them
            keys= []
            for l in self.Links.keys():
                keys.append(l)
            
            for k in keys:
                if self.Links[k][2] in self.leaves:
                   del self.Links[k]    


#         In this coding block the data for each node of the tree will be calculated and stored in a dictionary

        self.TreeNodesData=[]
        #Look at the id of every node in the tree
        for nd in range(self.n_nodes): 
            if nd==0: 
                self.TreeNodesData=[self.X] #Assign the whole dataset to the root node       
            
            elif nd>0:
                #Look at the parents nodes ids
                for ndd in set(self.Links.keys()): 
                    for ndi in range(0,2):
                        #Check is nd id is equal to the right child of the parent. If so store the data that correspond
                        #to the values that are lower than the threshold of the parent node
                        if nd==self.Links[ndd][0]: 
                            self.TreeNodesData.insert(nd, self.TreeNodesData[self.Links[ndd][2]][self.TreeNodesData[self.Links[ndd][2]].loc[:,self.feature_labels[self.Links[ndd][2]]]<=self.Thresholds[self.Links[ndd][2]]])
                        #Same as the above condition but for the left child. The values of this node will be larger than
                        #the threshold of the parent node
                        elif nd==self.Links[ndd][1]: 
                            self.TreeNodesData.insert(nd, self.TreeNodesData[self.Links[ndd][2]][self.TreeNodesData[self.Links[ndd][2]].loc[:,self.feature_labels[self.Links[ndd][2]]]>self.Thresholds[self.Links[ndd][2]]])    
        del(self.TreeNodesData[len(self.Thresholds):])

#         In this coding block we calculate the output of each instance for each node of the tree

        self.OutputsNodesData=[]
        self.classes_train=np.array(self.outputs_train)                   
        for nd in range(self.n_nodes): #Look at the id of every node in the tree
            if nd==0:
                self.OutputsNodesData=[self.classes_train]  #Assign to the root node the classes that correspond to the whole training outputs dataset
            else:
                self.OutputsNodesData.insert(nd, self.Y[self.TreeNodesData[nd].index]) #Assign the appropriate classes looking at the indexes of the original outputs object

                
                
                
                
                
        
    
                   
                
class Nodes:
    '''
    Description:
    This class contains a function that creates an object of type dictionary. The dictionary contains the following key value pairs:
    'nr_vertices': An integer denoting the number of nodes in the DT
    'nodes_ids': List with the nodes_ids 
    'nodes_depths': List with the nodes depths
    'nodes_labels': List with the feature labels
    'nodes_impurities': Dictionary which has the nodes ids as keys and the corresponding nodes Impurities
    'nodes_thresholds': A numpy array with the nodes thresholds
    'nodes_samples': Dictionary which has the nodes ids as keys and the corresponding nodes samples
    'nodes_values': Dictionary which has the nodes ids as keys and the corresponding nodes values
    'nodes_classes': Dictionary which has the nodes ids as keys and the corresponding nodes classes
    
    Inputs:
    TreeObject:  It can be: 1) An object of class: TreeStructure 2) An object of class pandas dataframe
    '''
    def __init__(self, TreeObject):
        if isinstance(TreeObject, TreeStructure):
            #Nodes properties:
            self.Nodes={'nr_vertices': TreeObject.n_nodes, #Assign number of nodes
                        'nodes_ids': TreeObject.IDs_Depths.iloc[:,0], #Assign nodes_ids 
                        'nodes_depths': TreeObject.IDs_Depths.iloc[:,1], #Assign nodes depths
                        'nodes_labels': TreeObject.feature_labels, #Assign feature labels
                        'nodes_impurities': TreeObject.Impurities, #Assign impurities
                        'nodes_thresholds': TreeObject.Thresholds, #Assign thresholds
                        'nodes_samples': TreeObject.Node_samples, #Assign node samples
                        'nodes_values': TreeObject.Node_Values, #Assign node values
                        'nodes_classes': TreeObject.Node_classes, #Assign node classes
                       }
            
        elif isinstance(TreeObject, pd.DataFrame):
            #Nodes properties:
            self.Nodes={}
            self.Nodes['nr_vertices'] = len(TreeObject.loc[:,'Id'])
            self.Nodes['nodes_ids'] = TreeObject['Id']
            self.Nodes['nodes_depths'] = TreeObject['Depth']
            self.Nodes['nodes_labels'] = list(TreeObject['nodes_labels'])
            self.Nodes['nodes_impurities'] = dict(TreeObject['nodes_impurities'])
            self.Nodes['nodes_thresholds'] = np.array(TreeObject['nodes_thresholds'])
            self.Nodes['nodes_samples'] = dict(TreeObject['Node_Samples'])
            self.Nodes['nodes_values'] = dict(TreeObject['nodes_values'])
            self.Nodes['nodes_classes'] = dict(TreeObject['nodes_classes'])
            
        
        
        
        
        

        
        
        
class Edges:
    '''
    Description:
    This class contains a function that creates an object of type dictionary. The dictionary contains the following key value pairs:
    'links':      A list containing sublists with the links among the various nodes
    'edge_seq':   A list containing sublists with the edges sequences based on the links
    
    Inputs:
    TreeObject:  It can be: 1) An object of class: TreeStructure 2) An object of class pandas dataframe
    '''

    def __init__(self, TreeObject):
        if isinstance(TreeObject, TreeStructure):
            #Edges
            links=TreeObject.Links
            edge_seq=[]
            for i in links.keys():
                edge_seq.append([i, links[i][0]])
                edge_seq.append([i, links[i][1]])
            #Create a method to store edges properties:
            self.Edges={'links': links, #Assign the links to each node
                        'edge_seq': edge_seq #Assign the edges sequence
                       } 
        elif isinstance(TreeObject, pd.DataFrame):
            #Edges    
            links=get_links(TreeObject)
            edges_seq=get_edge_seq(links)

            self.Edges = {}
            self.Edges['links'] = links
            self.Edges['edge_seq'] = edges_seq
                
                

                
                
             
                
                    

class ManualPruning:
    """
    Description:
    This function enables manual pruning of the tree.

    Inputs:
    node_id:                         The node id which defines the branch to be pruned
    TreeObject:                      It can be: 1) An object of class: TreeStructure 2) An object of class pandas dataframe
    classes_dict:                    Dictionary containing: 1) A dictionary with key Classes Labels where the names of the groups are stored in a list (the names are in string format) 
                                                            2) A dictionary with key Classes Colors where the names of the colors of each group are stored in a list (the names of the colors are in string format)
    criterion:                       A string denoting the criterion used to make splits. It can be either 'gini' or 'entropy'. Default to 'gini'
    nodes_coloring:                  A string denoting which nodes colouring strategy to be used. It can be 'Impurity' for colouring nodes based on their Impurity, 'Classes' for colouring nodes based on their class and                                            'Features_color_groups' based on user defined features colour groups. Default to Impurity.
    edges_shape:
    User_features_color_groups:      A dictionary with the following key-value pairs:
                                    'Groups & parameters': A dictionary which has the names of the groups as keys and list with the name of the inputs features that belong to that groups as values.
                                    'Colors of groups':    A dictionary which has the names of the groups as keys and the colour code for that groups as values.
    Best_first_Tree_Builder=True:    Whether the tree was built in a Best First or Depth First manne. It takes
                                     True or False values. Default to True. 
    txt_size:                        An integer denoting the size of the text appearing in the plot and legend. Default to 15
    mrk_size:                        An integer denoting the size of the markers appearing in the plot and legend. Default to 15
    plot_width:                      An integer denoting the width of the plot. Default to 1400
    plot_height:                     An integer denoting the height of the plot. Default to 800
    """
    
    def __init__(self, node_id, TreeObject, classes_dict, criterion='gini', nodes_coloring='Impurity', edges_shape='Lines', User_features_color_groups=None, Best_first_Tree_Builder=True, txt_size=15, mrk_size=15, plot_width = 1400, plot_height=800): 
        self.node_id=node_id
        self.classes_dict=classes_dict
        self.nodes_coloring=nodes_coloring
        self.edges_shape=edges_shape
        self.User_features_color_groups=User_features_color_groups
        self.Best_first_Tree_Builder=Best_first_Tree_Builder
        self.txt_size=txt_size
        self.mrk_size=mrk_size
        self.plot_width=plot_width
        self.plot_height=plot_height
        
        
        if isinstance(TreeObject, TreeStructure):
            #Retrieve the criterion the tree was built (gini or entropy)
            self.criterion=TreeObject.criterion
            #Store the original parents and leaves nodes list in variables because during the pruning processs there some nodes will become from parents leaves
            parents_list=list(TreeObject.parents)
            leaves_list=list(TreeObject.leaves)
            
            Tree_edges=Edges(TreeObject)
            Tree_gr_str=graph_structure(len(TreeObject.IDs_Depths.loc[:,'Id']), Tree_edges.Edges['edge_seq'], TreeObject.IDs_Depths.loc[:,'Node_Samples'], TreeObject.IDs_Depths.loc[:,'Id'], Best_first_Tree_Builder=True)
            add_coordinates_columns(TreeObject, Tree_gr_str, TreeObject.IDs_Depths['Node_Samples'][0], x_y_coords_only=False, tree_links=None)
                
            self.Pruned_tree=TreeObject.IDs_Depths.copy()

            #Check the input node id: Valid node ID
            if self.node_id==0 or self.node_id<0:
                print('Node ID should be greater than 0')
            elif self.node_id>0:
                #Check if the input node id is a leaf. If yes, raise an Error:
                if self.node_id in TreeObject.leaves:
                    print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given")
                else:
                    if TreeObject.Links[self.node_id][0] in TreeObject.leaves and TreeObject.Links[self.node_id][1] in TreeObject.leaves:
                        nodes_pr=smallest_branch(self.node_id, TreeObject)
                        nodes_pr.append(self.node_id)
                    else:
                        nodes_pr=nodes_to_prune(self.node_id, TreeObject)

        
        if isinstance(TreeObject, pd.DataFrame):
            #Retrieve the criterion the tree was built (gini or entropy)
            self.criterion=criterion
            
            #Retrieve the parents and leaves nodes
            parents_list=list(TreeObject.loc[TreeObject.loc[:,'nodes_thresholds'] != -2, 'Id'])
            leaves_list=list(TreeObject.loc[TreeObject.loc[:,'nodes_thresholds'] == -2, 'Id'])
            
            Tree_edges=Edges(TreeObject)
            self.Pruned_tree=TreeObject.copy()

            #Check the input node id: Valid node ID
            if self.node_id==0 or self.node_id<0:
                print('Node ID should be greater than 0')
            elif self.node_id>0:
                #Check if the input node id is a leaf. If yes, raise an Error:
                if self.node_id in leaves_list:
                    print("The id of the node to be pruned must not be a leaf. An id of a leaf node is given")
                else:
                    if Tree_edges.Edges['links'][self.node_id][0] in leaves_list and Tree_edges.Edges['links'][self.node_id][1] in leaves_list:
                        nodes_pr=smallest_branch(self.node_id, TreeObject)
                        nodes_pr.append(self.node_id)
                    else:
                        nodes_pr=nodes_to_prune(self.node_id, TreeObject)
        
        #PRUNE THE TREE:

        #Define the necessary conditions to prune the tree
        #Change the label of the node to leaf:
        self.Pruned_tree['nodes_labels'][self.node_id]='leaf_node'
        #Change the threshold of the node to -2
        self.Pruned_tree['nodes_thresholds'][self.node_id]=-2
        #Store the indexes to remove in variables
        subset = self.Pruned_tree[(self.Pruned_tree['Id'].isin(nodes_pr)) & (self.Pruned_tree['Id'] > self.node_id)]
        #Remove the redundant nodes and edges
        self.Pruned_tree.drop(index=list(subset.index), inplace=True)

        #Re-index the IDs so that we have a normal sequence of IDs
        self.Pruned_tree.index = range(len(self.Pruned_tree))
        self.Pruned_tree['Id']=self.Pruned_tree.index


        #RETRIEVE INFO TO PLOT THE TREE:
        #Retrieve the Edges info of the tree
        self.Pruned_tree_Edges=Edges(self.Pruned_tree)

        #Get the graph structure of the tree
        self.Pruned_tree_gr_str=graph_structure(len(self.Pruned_tree.loc[:,'Id']), self.Pruned_tree_Edges.Edges['edge_seq'], self.Pruned_tree.loc[:,'Node_Samples'], self.Pruned_tree.loc[:,'Id'], Best_first_Tree_Builder=True)
        add_coordinates_columns(self.Pruned_tree, self.Pruned_tree_gr_str, self.Pruned_tree['Node_Samples'][0], x_y_coords_only=True, tree_links=self.Pruned_tree_Edges.Edges['links'])

        #Retrieve the Nodes info of the tree
        self.Pruned_tree_Nodes=Nodes(self.Pruned_tree)
   
        
        #PLOT THE TREE:

        #Create a figure
        self.Img=go.FigureWidget()

        #Retrieve layout info
        Thres_dict, Hovertext, shape_dict, shape_lines_dict, color_impurities_dict, color_classes_dict, color_visual_style, subranges, upper_limit= format_graph(self.Pruned_tree_Nodes.Nodes, self.Pruned_tree_Edges.Edges, self.criterion, self.classes_dict, nodes_coloring=self.nodes_coloring, edges_shape='Lines', User_features_color_groups=self.User_features_color_groups, Best_first_Tree_Builder=True)

        #Add the necessary traces
        for ed in self.Pruned_tree.loc[1:, 'Id']:
            self.Img.add_trace(go.Scatter(x=[self.Pruned_tree['edges_x_coord'][ed][0], self.Pruned_tree['edges_x_coord'][ed][1]],    #self.TreePlot.fig
                                          y=[self.Pruned_tree['edges_y_coord'][ed][0], self.Pruned_tree['edges_y_coord'][ed][1]],
                                          mode='lines+text',
                                          line=dict(color='rgb(90,80,70)',
                                          width=self.Pruned_tree.loc[ed, 'edges_width'], # edges_widths[ed], 
                                          shape=shape_lines_dict),
                                          showlegend=False,
                                          hoverinfo='none',
                                          opacity=0.7))

        for nd in self.Pruned_tree.loc[:, 'Id']:
            self.Img.add_trace(go.Scatter(x=[self.Pruned_tree.loc[nd, 'x_coord']],
                                          y=[self.Pruned_tree.loc[nd, 'y_coord']],
                                          mode='markers+text',
                                          marker=dict(symbol=shape_dict[nd],  size=self.mrk_size, color=color_visual_style[nd], line=dict(color='rgb(50,50,50)', width=1)),
                                          showlegend=False,
                                          text=Thres_dict[nd],
                                          textposition='top center',
                                          textfont={'size':self.txt_size},
                                          hovertext=Hovertext[nd],
                                          hoverlabel=dict(namelength=0),
                                          hoverinfo='text',
                                          opacity=1))

        layout=go.Layout(autosize=False, width= self.plot_width, height= self.plot_height,
                         xaxis= go.layout.XAxis(linecolor = 'black',
                                                linewidth = 1,
                                                mirror = True),
                         yaxis= go.layout.YAxis(linecolor = 'black',
                                                linewidth = 1,
                                                mirror = True),
                         margin=go.layout.Margin(l=20,r=20,b=50,t=50,pad = 4),
                         paper_bgcolor='white',
                         plot_bgcolor='white')

        self.Img.update_layout(layout)
        add_legend(self.Img, self.Pruned_tree, None, color_impurities_dict, subranges, color_classes_dict, shape_dict, nodes_coloring=self.nodes_coloring, User_features_color_groups=self.User_features_color_groups, mrk_size=self.mrk_size, txt_size=self.txt_size,)
        self.Img.show()          
    
  
            
            







class graph_structure:
    '''
    Description:
    This class creates objects regarding the graph structure. The DT will be plotted on a scatterplot. So each node and edge need to be assigned with the appropriate coordinates.
    That's what this class do.
    '''
    def __init__(self, nr_vertices, edge_seq, nodes_samples, nodes_ids, Best_first_Tree_Builder=True):
        #Assign X,Y coordinates to the nodes and the edges
        #Creating the layout
        self.g=Graph()
        self.g.add_vertices(nr_vertices)
        self.g.add_edges(edge_seq)
        self.lay=self.g.layout(layout='rt', root=nodes_ids[0])


        self.position = {k: self.lay[k] for k in range(nr_vertices)}
        self.Y = [self.lay[k][1] for k in range(nr_vertices)]
        self.M = max(self.Y)


        self.L = len(self.position)
        self.Xn = [self.position[k][0] for k in range(self.L)]
        self.Yn = [2*self.M-self.position[k][1] for k in range(self.L)]
        self.Xe = []
        self.Ye = []


        for edge in edge_seq:
            self.Xe+=[self.position[edge[0]][0],self.position[edge[1]][0], None]
            self.Ye+=[2*self.M-self.position[edge[0]][1],2*self.M-self.position[edge[1]][1], None]



        #Assign attributes to the edges
        #Widths
        denominator=nodes_samples[0]/10
        if Best_first_Tree_Builder==True:
            self.edges_widths=[nodes_samples[i[1]]/denominator for i in edge_seq]    #nodes_samples[i]/100 for i in range(1, len(nodes_samples))
        elif Best_first_Tree_Builder==False:
            self.edges_widths=[nodes_samples[i[1]]/denominator for i in edge_seq]              
                

                
                
                
                
                
                
                
                                
class Plot_Tree:
    '''
    This class creates a plot of the fitted tree in plotly.
    
    Inputs: 
    TreeObject:                   It can be: 1) An object of class: TreeStructure 2) An object of class pandas dataframe
    classes_dict:                 Dictionary containing: 1) A dictionary with key Classes Labels where the names of the groups are stored in a list (the names are in string format) 
                                                         2) A dictionary with key Classes Colors where the names of the colors of each group are stored in a list (the names of the colors are in string format)
    criterion:                    A string denoting the criterion used to make splits. It can be either 'gini' or 'entropy'. Default to 'gini'   
    Best_first_Tree_Builder:      
    nodes_coloring:               A string denoting which nodes colouring strategy to be used. It can be 'Impurity' for colouring nodes based on their Impurity, 'Classes' for colouring nodes based on their class and                                           'Features_color_groups' based on user defined features colour groups. Default to Impurity.
    edges_shape:
    User_features_color_groups:   A dictionary with the following key-value pairs:
                                  'Groups & parameters': A dictionary which has the names of the groups as keys and list with the name of the inputs features that belong to that groups as values.
                                  'Colors of groups':    A dictionary which has the names of the groups as keys and the colour code for that groups as values.
     plot_width:                  An integer denoting the width of the plot. Default to 1400
    plot_height:                  An integer denoting the height of the plot. Default to 800
    mrk_size:                     An integer denoting the size of the markers appearing in the plot and legend. Default to 15
    txt_size:                     An integer denoting the size of the text appearing in the plot and legend. Default to 15
    opacity_edges:                A float indicating the opacity of the edges in the plot 
    opacity_nodes:                A float indicating the opacity of the nodes in the plot
    show_figure:                  Whether to show the figure or not. It can be either True or False. Default to True.
    '''
    def __init__(self, TreeObject, classes_dict, criterion='gini', Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines',
                 User_features_color_groups=None, plot_width= 1400, plot_height=800, mrk_size=15, txt_size =15, opacity_edges=0.7, opacity_nodes=1, show_figure=True):
        self.TreeObject = TreeObject
        self.classes_dict = classes_dict
        self.criterion = criterion
        self.Best_first_Tree_Builder = Best_first_Tree_Builder
        self.nodes_coloring = nodes_coloring
        self.edges_shape = edges_shape
        self.User_features_color_groups = User_features_color_groups
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.mrk_size = mrk_size
        self.txt_size = txt_size
        self.opacity_edges = opacity_edges
        self.opacity_nodes = opacity_nodes
        
        
        #Construct Nodes_dict and Edges_dict
        self.nodes = Nodes(self.TreeObject)
        self.edges = Edges(self.TreeObject)
        
        #Create the graph structure
        self.gr_str = graph_structure(nr_vertices=self.nodes.Nodes['nr_vertices'], edge_seq=self.edges.Edges['edge_seq'], nodes_samples=self.nodes.Nodes['nodes_samples'], nodes_ids=self.nodes.Nodes['nodes_ids'], Best_first_Tree_Builder=Best_first_Tree_Builder)
        
        #Extract necessary variables for formatting the graph
        self.Thres_dict, self.Hovertext, self.shape_dict, self.shape_lines_dict, self.color_impurities_dict, self.classes_dict, self.color_visual_style, subranges, upper_limit = format_graph(self.nodes.Nodes, self.edges.Edges, self.criterion, self.classes_dict, nodes_coloring=self.nodes_coloring, edges_shape='Lines', User_features_color_groups=self.User_features_color_groups, Best_first_Tree_Builder=True)
        
        #Construct the image
        self.fig=go.FigureWidget()
        #Layout of the image
        self.fig_layout=go.Layout(autosize=False, width=self.plot_width, height=self.plot_height,
                                      xaxis = go.layout.XAxis(linecolor = 'black',
                                                             linewidth = 1,
                                                             mirror = True),
                                      yaxis = go.layout.YAxis(linecolor = 'black',
                                                                linewidth = 1,
                                                                mirror = True),
                                      margin = go.layout.Margin(l=20,r=20,b=50,t=50,pad = 4),
                                      paper_bgcolor = 'white',
                                      plot_bgcolor = 'white')
        self.fig.update_layout(self.fig_layout)
        
        #Add traces to the image:
        
        if isinstance(self.TreeObject, TreeStructure):
            add_coordinates_columns(self.TreeObject, self.gr_str, self.TreeObject.IDs_Depths['Node_Samples'][0])
            
            for ed in self.TreeObject.IDs_Depths.loc[1:, 'Id']:
                self.fig.add_trace(go.Scatter(x=[self.TreeObject.IDs_Depths['edges_x_coord'][ed][0], self.TreeObject.IDs_Depths['edges_x_coord'][ed][1]],    #self.TreePlot.fig
                                              y=[self.TreeObject.IDs_Depths['edges_y_coord'][ed][0], self.TreeObject.IDs_Depths['edges_y_coord'][ed][1]],
                                              mode='lines+text',
                                              line=dict(color='rgb(90,80,70)',
                                              width=self.TreeObject.IDs_Depths.loc[ed, 'edges_width'], # edges_widths[ed], 
                                              shape=self.shape_lines_dict),
                                              showlegend=False,
                                              hoverinfo='none',
                                              opacity=self.opacity_edges))

            for nd in self.TreeObject.IDs_Depths.loc[:, 'Id']:
                self.fig.add_trace(go.Scatter(x=[self.TreeObject.IDs_Depths.loc[nd, 'x_coord']],
                                              y=[self.TreeObject.IDs_Depths.loc[nd, 'y_coord']],
                                              mode='markers+text',
                                              marker=dict(symbol=self.shape_dict[nd],  size=self.mrk_size, color=self.color_visual_style[nd], line=dict(color='rgb(50,50,50)', width=1)),
                                              showlegend=False,
                                              text=self.Thres_dict[nd],
                                              textposition='top center',
                                              textfont={'size':self.txt_size},
                                              hovertext=self.Hovertext[nd],
                                              hoverlabel=dict(namelength=0),
                                              hoverinfo='text',
                                              opacity=self.opacity_nodes))

            self.fig.update_layout(self.fig_layout)
            add_legend(self.fig, self.TreeObject.IDs_Depths, None, self.color_impurities_dict, subranges, self.classes_dict, self.shape_dict, nodes_coloring=self.nodes_coloring, User_features_color_groups=self.User_features_color_groups, mrk_size=self.mrk_size, txt_size=self.txt_size)
        
        elif isinstance(self.TreeObject, pd.DataFrame):
            for ed in self.TreeObject.loc[1:, 'Id']:
                self.fig.add_trace(go.Scatter(x=[self.TreeObject['edges_x_coord'][ed][0], self.TreeObject['edges_x_coord'][ed][1]],    #self.TreePlot.fig
                                              y=[self.TreeObject['edges_y_coord'][ed][0], self.TreeObject['edges_y_coord'][ed][1]],
                                              mode='lines+text',
                                              line=dict(color='rgb(90,80,70)',
                                              width=self.TreeObject.loc[ed, 'edges_width'], # edges_widths[ed], 
                                              shape=self.shape_lines_dict),
                                              showlegend=False,
                                              hoverinfo='none',
                                              opacity=self.opacity_edges))

            for nd in self.TreeObject.loc[:, 'Id']:
                self.fig.add_trace(go.Scatter(x=[self.TreeObject.loc[nd, 'x_coord']],
                                              y=[self.TreeObject.loc[nd, 'y_coord']],
                                              mode='markers+text',
                                              marker=dict(symbol=self.shape_dict[nd],  size=self.mrk_size, color=self.color_visual_style[nd], line=dict(color='rgb(50,50,50)', width=1)),
                                              showlegend=False,
                                              text=self.Thres_dict[nd],
                                              textposition='top center',
                                              textfont={'size':self.txt_size},
                                              hovertext=self.Hovertext[nd],
                                              hoverlabel=dict(namelength=0),
                                              hoverinfo='text',
                                              opacity=self.opacity_nodes))

            add_legend(self.fig, self.TreeObject, None, self.color_impurities_dict, subranges, self.classes_dict, self.shape_dict, nodes_coloring=self.nodes_coloring, User_features_color_groups=self.User_features_color_groups, mrk_size=self.mrk_size, txt_size=self.txt_size)
            
        if show_figure == True:
            self.fig.show()
            


        
        
        
        
  

        
class specify_feature_split_point:
    '''
    This class contains methods that enable user to manually change feature to split or split point.
    
    Inputs:
    Tree:             The tree in which we want to mannualy change feature to split or split point
    features:         A list containing the names of the various features as strings
    X:                Pandas data frame object which contains the input data used to classify the tree (most of the times it will be the
                      training dataset)
    Y:                Pandas Dataframe (series) containing the whole output dataset
    outputs_train:    Pandas Dataframe object (series) containing the output training dataset
    classes_dict:     Dictionary containing: 1) A dictionary with key Classes Labels where the names of the groups are stored in a list (the names are in string format) 
                                             2) A dictionary with key Classes Colors where the names of the colors of each group are stored in a list (the names of the colors are in string format)
    parameters:       The following parameters: criterion, splitter, max_depth_left, max_depth_right, min_samples_split_left, min_samples_split_right,
                      min_samples_leaf_left, min_samples_leaf_right, min_weight_fraction_leaf_left, min_weight_fraction_leaf_left, max_features, 
                      random_state, max_leaf_nodes_left, max_leaf_nodes_right, min_impurity_decrease,
                      min_impurity_split, class_weight, ccp_alpha
                      Are parameters that are used in tree.DecisionTreeClassifier and the default values are the same as tree.DecisionTreeClassifier.
                      For more information see:
                      https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

                      However, the user is able to use different values for the left and right subtrees for the
                      following parameters: max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                      max_leaf_nodes.
    '''
    def __init__(self, Tree, features, X, Y, outputs_train, classes_dict, node_id, feature, new_threshold, print_rules=False, Best_first_Tree_Builder=True,
                 criterion='gini', splitter='best', max_depth_left=None, max_depth_right=None, min_samples_split_left=2, min_samples_split_right=2, min_samples_leaf_left=1,
                 min_samples_leaf_right=1, min_weight_fraction_leaf_left=0.0, min_weight_fraction_leaf_right=0.0, max_features=None, random_state=0, max_leaf_nodes_left=None, 
                 max_leaf_nodes_right=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0,
                 nodes_coloring='Impurity', edges_shape='Lines', User_features_color_groups=None, opacity_edges=0.7, opacity_nodes=1,
                 plot_width=1400, plot_height=800, mrk_size=15, txt_size=15, show_figure=True):
        
        self.Tree=Tree                         #The classified tree. Object of cass sklearn.tree.DecisionTreeClassifier, sklearn.tree.fit                                 
        self.features=features                 #A list containing the names of the various features as strings
        self.X=X                               #Pandas data frame object which contains the input data used to classify the tree (most of the times it will be the
                                               #training dataset)
        self.Y=Y                               #Pandas Dataframe (series) containing the whole output dataset
        self.outputs_train=outputs_train       #Pandas Dataframe object (series) containing the output training dataset
        self.classes_dict=classes_dict         #A dictionary containing the user defined classes labels and colors
        self.node_id=node_id                   #Integer denoting the node id that feature or the threshold will be changed 
        self.feature=feature                   #String denoting the new feature label to split in the node
        self.new_threshold=new_threshold       #Integer or float denoting the new split point at the node
        self.nodes_coloring=nodes_coloring
        self.User_features_color_groups=User_features_color_groups
        self.criterion=criterion
        self.opacity_edges=opacity_edges
        self.opacity_nodes=opacity_nodes
        self.plot_width=plot_width
        self.plot_height=plot_height
        self.mrk_size=mrk_size
        self.txt_size=txt_size
        
        if isinstance(self.Tree, sklearn.tree._classes.DecisionTreeClassifier):
            self.TreeStructure=TreeStructure(self.Tree, self.features, self.X, self.Y, self.outputs_train, self.classes_dict, print_rules=False, Best_first_Tree_Builder=True)
            self.TreePlot=Plot_Tree(self.TreeStructure, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines', User_features_color_groups=None, plot_width=self.plot_width, plot_height=self.plot_height, mrk_size=self.mrk_size, txt_size=self.txt_size, show_figure=False)
            self.nodes=Nodes(self.TreeStructure)
            self.edges=Edges(self.TreeStructure)
            self.gr_str=graph_structure(self.nodes.Nodes['nr_vertices'], self.edges.Edges['edge_seq'], self.nodes.Nodes['nodes_samples'], self.nodes.Nodes['nodes_ids'], Best_first_Tree_Builder=True)
            self.left_subtree, self.right_subtree, self.left_branch, self.left_branch_outputs, self.right_branch, self.right_branch_outputs=change_feature_or_threshold_to_split(self.TreeStructure.TreeNodesData, self.TreeStructure.OutputsNodesData, node_id, feature, new_threshold, 
                                                                                       criterion=criterion, splitter=splitter, max_depth_left=max_depth_left, max_depth_right=max_depth_right, 
                                                                                       min_samples_split_left=min_samples_split_left, min_samples_split_right=min_samples_split_right, 
                                                                                       min_samples_leaf_left=min_samples_leaf_left, min_samples_leaf_right=min_samples_leaf_right, 
                                                                                       min_weight_fraction_leaf_left=min_weight_fraction_leaf_left,
                                                                                       min_weight_fraction_leaf_right=min_weight_fraction_leaf_right, max_features=max_features, 
                                                                                       random_state=random_state, max_leaf_nodes_left=max_leaf_nodes_left, max_leaf_nodes_right=max_leaf_nodes_right, 
                                                                                       min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, 
                                                                                       class_weight=class_weight, ccp_alpha=ccp_alpha)
        elif isinstance(self.Tree, pd.DataFrame):
            self.TreePlot=Plot_Tree(self.Tree, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines', User_features_color_groups=None, plot_width=self.plot_width, plot_height=self.plot_height, mrk_size=self.mrk_size, txt_size=self.txt_size, show_figure=False)
            self.nodes=Nodes(self.Tree)
            self.edges=Edges(self.Tree)
            self.gr_str=graph_structure(self.nodes.Nodes['nr_vertices'], self.edges.Edges['edge_seq'], self.nodes.Nodes['nodes_samples'], self.nodes.Nodes['nodes_ids'], Best_first_Tree_Builder=True)
            self.TreeNodesData=get_nodes_data(self.Tree, self.X)
            self.OutputsNodesData=get_OutputsNodesData(self.Tree, self.Y, self.outputs_train, self.TreeNodesData)
            self.left_subtree, self.right_subtree, self.left_branch, self.left_branch_outputs, self.right_branch, self.right_branch_outputs=change_feature_or_threshold_to_split(self.TreeNodesData, self.OutputsNodesData, node_id, feature, new_threshold, 
                                                                                       criterion=criterion, splitter=splitter, max_depth_left=max_depth_left, max_depth_right=max_depth_right, 
                                                                                       min_samples_split_left=min_samples_split_left, min_samples_split_right=min_samples_split_right, 
                                                                                       min_samples_leaf_left=min_samples_leaf_left, min_samples_leaf_right=min_samples_leaf_right, 
                                                                                       min_weight_fraction_leaf_left=min_weight_fraction_leaf_left,
                                                                                       min_weight_fraction_leaf_right=min_weight_fraction_leaf_right, max_features=max_features, 
                                                                                       random_state=random_state, max_leaf_nodes_left=max_leaf_nodes_left, max_leaf_nodes_right=max_leaf_nodes_right, 
                                                                                       min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, 
                                                                                       class_weight=class_weight, ccp_alpha=ccp_alpha)
                
    def merge_subtrees(self):
        '''
        Description:
        This method merges the two  subtree into one unified tree. It will store the necessary nodes and edges information in dictionaries
        '''
        #Retrieve the TreeStructure of left tree
        if self.node_id==0:
            self.left_subtree_TreeStructure=TreeStructure(self.left_subtree, self.features, self.left_branch, self.Y, self.left_branch_outputs, self.classes_dict, Best_first_Tree_Builder=True)
            self.left_subtree_plot=Plot_Tree(self.left_subtree_TreeStructure, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines',  
                                             User_features_color_groups=None, show_figure=False);
        else:
            self.left_subtree_TreeStructure=TreeStructure(self.left_subtree, self.features, self.left_branch, self.left_branch_outputs, self.left_branch_outputs, self.classes_dict, Best_first_Tree_Builder=True)
            self.left_subtree_plot=Plot_Tree(self.left_subtree_TreeStructure, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines',  
                                             User_features_color_groups=None, show_figure=False);
        
        #Check for missing class labels in nodes_values in the nodes
        missing_lbl_left=check_labels(self.left_subtree_TreeStructure)
        #Fill with zeros the nodes_values of nodes with classes less than the max available classes
        fill_missing_classes_nodes_values(self.left_subtree_TreeStructure, missing_lbl_left, self.classes_dict)

        #Right tree
        if self.node_id==0:
            self.right_subtree_TreeStructure=TreeStructure(self.right_subtree, self.features, self.right_branch, self.Y, self.right_branch_outputs, self.classes_dict, Best_first_Tree_Builder=True)
            self.right_subtree_plot=Plot_Tree(self.right_subtree_TreeStructure, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines',  
                                              User_features_color_groups=None, show_figure=False);
        else:
            self.right_subtree_TreeStructure=TreeStructure(self.right_subtree, self.features, self.right_branch, self.right_branch_outputs, self.right_branch_outputs, self.classes_dict, Best_first_Tree_Builder=True)
            self.right_subtree_plot=Plot_Tree(self.right_subtree_TreeStructure, self.classes_dict, self.criterion, Best_first_Tree_Builder=True, nodes_coloring='Impurity', edges_shape='Lines',  
                                              User_features_color_groups=None, show_figure=False);
        
        #Check for missing class labels in nodes_values in the nodes
        missing_lbl_right=check_labels(self.right_subtree_TreeStructure)
        #Fill with zeros the nodes_values of nodes with classes less than the max available classes
        fill_missing_classes_nodes_values(self.right_subtree_TreeStructure, missing_lbl_right, self.classes_dict)
        
        
        #Assign information for root node
        if isinstance(self.Tree, sklearn.tree._classes.DecisionTreeClassifier):
            root_depth=self.TreeStructure.IDs_Depths.loc[0,'Depth']
            root_impurity=self.TreeStructure.Impurities[self.node_id]
            root_samples=self.TreeStructure.Node_samples[self.node_id]
            root_values=self.TreeStructure.Node_Values[self.node_id]
            root_classes=self.TreeStructure.Node_classes[self.node_id]
        elif isinstance(self.Tree, pd.DataFrame):
            root_depth=self.Tree.loc[0,'Depth']
            root_impurity=self.Tree.loc[self.node_id, 'nodes_impurities']
            root_samples=self.Tree.loc[self.node_id, 'Node_Samples']
            root_values=self.Tree.loc[self.node_id, 'nodes_values']
            root_classes=self.Tree.loc[self.node_id, 'nodes_classes']
            
        
        #Now we have to merge the two subtrees into one so that we can plot them as one
        #We will now mmake the correspondecne of initial ids of the separate subtrees to the new ids of the new unified tree
        
        #Store in a list the original info of the left subtree
        orig_left_impurities=list(self.left_subtree_TreeStructure.Impurities.values())
        orig_left_node_samples=list(self.left_subtree_TreeStructure.Node_samples.values())
        orig_left_node_values=list(self.left_subtree_TreeStructure.Node_Values.values())
        orig_left_node_classes=list(self.left_subtree_TreeStructure.Node_classes.values())
        
        for i in self.left_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            #Change ids: New_ids=original id + 1 e.g. new_root_node_id= 0+1 (this is because the root node is the node of the big tree that we changed the feature/split point)
            self.left_subtree_TreeStructure.IDs_Depths.iloc[i,0]=self.left_subtree_TreeStructure.IDs_Depths.iloc[i,0]+1 
            #Change depths: New_depth=original depth + 1
            self.left_subtree_TreeStructure.IDs_Depths.iloc[i,1]=self.left_subtree_TreeStructure.IDs_Depths.iloc[i,1]+1 
            #Change impurities
        for new_i in self.left_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            #Change Impurities: We have an indexer looking at the new ids. So the Impurity with the updated node id= Impurity at the node with id (new-1)
            self.left_subtree_TreeStructure.Impurities[new_i]=orig_left_impurities[new_i-1]
            #Change nodes_samples: Same logic as for impurities
            self.left_subtree_TreeStructure.Node_samples[new_i]=orig_left_node_samples[new_i-1]
            #Change nodes_values Same logic as for impurities
            self.left_subtree_TreeStructure.Node_Values[new_i]=orig_left_node_values[new_i-1]
            #Change nodes_classes Same logic as for impurities
            self.left_subtree_TreeStructure.Node_classes[new_i]=orig_left_node_classes[new_i-1]

        del self.left_subtree_TreeStructure.Impurities[0]
        del self.left_subtree_TreeStructure.Node_samples[0]
        del self.left_subtree_TreeStructure.Node_Values[0]
        del self.left_subtree_TreeStructure.Node_classes[0]

        #Now we need to change the ids in the node links
        self.left_subtree_links={}
        self.left_subtree_keys_list=list(self.left_subtree_TreeStructure.Links.keys()) #This holds the ids of the initial left subtree. We need to find the ids for link nodes of the updated subtree
        for left_key in self.left_subtree_keys_list:   #Therefore the rule is: The indexer holds the initial id. So if the initial links were [1,2,0] in this case the indexer left_key==0 so
                                # we should add 1 [1+1, 2+1, 0+1]. In order to achieve this we need to apply updated_links[left_key+1]=[init_subtree_links[left_key][0]+1, init_subtree_links[left_key][1]+1
            self.left_subtree_links[left_key+1]=[self.left_subtree_TreeStructure.Links[left_key][0]+1,self.left_subtree_TreeStructure.Links[left_key][1]+1, left_key+1]

        #Retrieve the edge sequence for the updated subtree
        self.left_subtree_plot.edges.Edges['edge_seq']=[]
        for ln in self.left_subtree_links.keys():
            self.left_subtree_plot.edges.Edges['edge_seq'].append([ln, self.left_subtree_links[ln][0]])
            self.left_subtree_plot.edges.Edges['edge_seq'].append([ln, self.left_subtree_links[ln][1]])   
        
        #Retrieve the parents and leaves nodes for the updated subtree
        self.left_subtree_TreeStructure.parents=[]
        self.left_subtree_TreeStructure.leaves=[]
        for nd_id in self.left_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            if nd_id in self.left_subtree_links.keys():
                self.left_subtree_TreeStructure.parents.append(nd_id)
            else:
                self.left_subtree_TreeStructure.leaves.append(nd_id)
                


        #Store in a list the original ids for right subtree
        orig_right_impurities=list(self.right_subtree_TreeStructure.Impurities.values())
        orig_right_node_samples=list(self.right_subtree_TreeStructure.Node_samples.values())
        orig_right_node_values=list(self.right_subtree_TreeStructure.Node_Values.values())
        orig_right_node_classes=list(self.right_subtree_TreeStructure.Node_classes.values())
        for i_r in self.right_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            #Change Ids: New_ids=original id + last_node_from_left_subtree_+1 
            self.right_subtree_TreeStructure.IDs_Depths.iloc[i_r,0]=self.right_subtree_TreeStructure.IDs_Depths.iloc[i_r,0]+self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1
            #Change Depths: New_depth=original depth + root_node_depth_of_left_subtree Why? Because original right root node depth=0. But 0 should be the depth of the root of the new unified tree. So we add the depth of the root node of left subtree which should be 1. 
            self.right_subtree_TreeStructure.IDs_Depths.iloc[i_r,1]=self.right_subtree_TreeStructure.IDs_Depths.iloc[i_r,1]+self.left_subtree_TreeStructure.IDs_Depths.iloc[0,1]
        for new_i_r in self.right_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            #Change Impurities:  We have an indexer looking at the new ids. So the Impurity with the updated node id= Impurity at the node with id (new-last node id from left subtree-1)
            self.right_subtree_TreeStructure.Impurities[new_i_r]=orig_right_impurities[new_i_r-self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]-1]
            #Change nodes_samples: Same logic as Impurities
            self.right_subtree_TreeStructure.Node_samples[new_i_r]=orig_right_node_samples[new_i_r-self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]-1]
            #Change nodes_values: Same logic as Impurities
            self.right_subtree_TreeStructure.Node_Values[new_i_r]=orig_right_node_values[new_i_r-self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]-1]
            #Change nodes_classes: Same Logic as Impurities
            self.right_subtree_TreeStructure.Node_classes[new_i_r]=orig_right_node_classes[new_i_r-self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]-1]

        #Delete unnecessary keys from the dictionary. The above block will also keep keys of the original right subtree (0, 1, 2..). So we need to delete them
        for dl in range(0, self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1): #Why left? Because we need to delete everything before the last node id of the left subtree
            if dl in self.right_subtree_TreeStructure.Impurities.keys():
                del self.right_subtree_TreeStructure.Impurities[dl]
                del self.right_subtree_TreeStructure.Node_samples[dl]
                del self.right_subtree_TreeStructure.Node_Values[dl]
                del self.right_subtree_TreeStructure.Node_classes[dl]


        self.right_subtree_links={}
        self.right_subtree_keys_list=list(self.right_subtree_TreeStructure.Links.keys())
        for right_key in self.right_subtree_keys_list: #Same logic as the left subtree
            self.right_subtree_links[right_key+self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1]=[self.right_subtree_TreeStructure.Links[right_key][0]+self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1, 
                                                                                               self.right_subtree_TreeStructure.Links[right_key][1]+self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1,
                                                                                               right_key+self.left_subtree_TreeStructure.IDs_Depths.iloc[-1,0]+1]


        #Retrieve the edges sequence
        self.right_subtree_plot.edges.Edges['edge_seq']=[]
        for lnr in self.right_subtree_links.keys():
            self.right_subtree_plot.edges.Edges['edge_seq'].append([lnr, self.right_subtree_links[lnr][0]])
            self.right_subtree_plot.edges.Edges['edge_seq'].append([lnr, self.right_subtree_links[lnr][1]])

        #Retrieve the parent and leaves nodes
        self.right_subtree_TreeStructure.parents=[]
        self.right_subtree_TreeStructure.leaves=[]
        for nd_id_r in self.right_subtree_TreeStructure.IDs_Depths.iloc[:,0]:
            if nd_id_r in self.right_subtree_links.keys():
                self.right_subtree_TreeStructure.parents.append(nd_id_r)
            else:
                self.right_subtree_TreeStructure.leaves.append(nd_id_r)
        
        
        #Assemble all the info together
        #Concatenate the info for Nodes into one dictionary:
        nr_vertices=1+self.left_subtree_TreeStructure.n_nodes+self.right_subtree_TreeStructure.n_nodes
        nodes_ids=np.concatenate((np.array([0]), self.left_subtree_TreeStructure.IDs_Depths.iloc[:,0], self.right_subtree_TreeStructure.IDs_Depths.iloc[:,0]))
        nodes_depths=np.concatenate((np.array([0]), self.left_subtree_TreeStructure.IDs_Depths.iloc[:,1], self.right_subtree_TreeStructure.IDs_Depths.iloc[:,1]))
        nodes_labels=[self.feature]+self.left_subtree_plot.nodes.Nodes['nodes_labels']+self.right_subtree_plot.nodes.Nodes['nodes_labels']

        root_impurity_dict={nodes_ids[0]: root_impurity}
        nodes_impurities= {**root_impurity_dict, **self.left_subtree_plot.nodes.Nodes['nodes_impurities'], **self.right_subtree_plot.nodes.Nodes['nodes_impurities']}

        root_thres=np.array([self.new_threshold])
        nodes_thresholds=np.concatenate((root_thres, self.left_subtree_plot.nodes.Nodes['nodes_thresholds'], self.right_subtree_plot.nodes.Nodes['nodes_thresholds']))

        root_samples_dict={nodes_ids[0]: root_samples}
        nodes_samples= {**root_samples_dict, **self.left_subtree_plot.nodes.Nodes['nodes_samples'], **self.right_subtree_plot.nodes.Nodes['nodes_samples']}

        root_values_dict={nodes_ids[0]: root_values}
        nodes_values= {**root_values_dict, **self.left_subtree_plot.nodes.Nodes['nodes_values'], **self.right_subtree_plot.nodes.Nodes['nodes_values']}

        root_classes_dict={nodes_ids[0]: root_classes}
        nodes_classes= {**root_classes_dict, **self.left_subtree_plot.nodes.Nodes['nodes_classes'], **self.right_subtree_plot.nodes.Nodes['nodes_classes']}


        self.Nodes_dict={'nr_vertices': nr_vertices, 'nodes_ids': nodes_ids, 'nodes_depths': nodes_depths, 'nodes_labels': nodes_labels, 'nodes_impurities': nodes_impurities,
                    'nodes_thresholds': nodes_thresholds, 'nodes_samples': nodes_samples, 'nodes_values': nodes_values, 'nodes_classes': nodes_classes}



        #Concatenate the info for the edges into one dict
        root_links= {nodes_ids[0]: [self.left_subtree_plot.nodes.Nodes['nodes_ids'][0], self.right_subtree_plot.nodes.Nodes['nodes_ids'][0], nodes_ids[0]]}
        root_edge_seq= [[root_links[nodes_ids[0]][2], root_links[nodes_ids[0]][0]], [root_links[nodes_ids[0]][2], root_links[nodes_ids[0]][1]]]
        links={**root_links,**self.left_subtree_links, **self.right_subtree_links}
        edge_seq=root_edge_seq + self.left_subtree_plot.edges.Edges['edge_seq'] + self.right_subtree_plot.edges.Edges['edge_seq']
        self.Edges_dict={'links': links, 'edge_seq': edge_seq}
                
    def merged_tree_graph(self):
        '''
        This method stores and creates the graph for the merged tree.
        '''
        
        #Create the graph Structure       
        self.graph_str=graph_structure(nr_vertices=self.Nodes_dict['nr_vertices'], edge_seq=self.Edges_dict['edge_seq'], nodes_samples=self.Nodes_dict['nodes_samples'], 
                            nodes_ids=self.Nodes_dict['nodes_ids'], Best_first_Tree_Builder=True)
        
        Thres_dict, Hovertext, shape_dict, shape_lines_dict, color_impurities_dict, color_classes_dict, color_visual_style, subranges, upper_limit = format_graph(self.Nodes_dict, self.Edges_dict, self.criterion, self.classes_dict, nodes_coloring=self.nodes_coloring, edges_shape='Lines', User_features_color_groups=self.User_features_color_groups, Best_first_Tree_Builder=True)

        #Create the figure of the merged tree
        #Add edges       
        counter=list(range(0, len(self.graph_str.Xe),3)) 

        self.merged_tree_plot=go.FigureWidget()
        for count in range(0,len(counter)):
            self.merged_tree_plot.add_trace(go.Scatter(x=[self.graph_str.Xe[counter[count]], self.graph_str.Xe[counter[count]+1]],
                                                       y=[self.graph_str.Ye[counter[count]], self.graph_str.Ye[counter[count]+1]], 
                                                       mode='lines+text', line=dict(color='rgb(90,80,70)', width=self.graph_str.edges_widths[count]), showlegend=False, opacity=0.7))       

        #Add the new nodes    
        for i in range(0, len(self.graph_str.Xn)):
            self.merged_tree_plot.add_trace(go.Scatter(x=[self.graph_str.Xn[i]], 
                                                       y=[self.graph_str.Yn[i]], mode='markers+text',
                                                     marker=dict(symbol=shape_dict[i], size=15, color=color_visual_style[i], line=dict(color='rgb(50,50,50)', width=1)),
                                                     showlegend=False, text=Thres_dict[i], hovertext=Hovertext[i], hoverlabel=dict(namelength=0), hoverinfo='text', 
                                                         textposition='top center', opacity=1 ))
            
        self.merged_tree_plot['layout'].update(autosize=False, width=1400,  height=800, 
                                           xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
                                           yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 1, mirror = True),
                                           margin=go.layout.Margin(l=20,r=20,b=50,t=50,pad = 4), paper_bgcolor='white', plot_bgcolor='white')
        
    def unified_plot(self):  
        if isinstance(self.Tree, sklearn.tree._classes.DecisionTreeClassifier):
            #1st step: Define the left, right branches, find the nodes that have equal depth and add the necessary columns to IDs_Depths pandas dataframe for the initial tree
            Left_branch_nds, Right_branch_nds=get_left_and_right_banches_nodes(self.TreeStructure)
            Same_Depth, Left_Same_Depth, Right_Same_Depth=get_nodes_with_equal_depth(self.TreeStructure, Left_branch_nds, Right_branch_nds)
            add_coordinates_columns(self.TreeStructure, self.gr_str, self.TreeStructure.IDs_Depths['Node_Samples'][0])

            #2nd step: Prune the initial tree at the node where the changes (feature or split point) will take place:
            if self.TreeStructure.Links[self.node_id][0] in self.TreeStructure.leaves and self.TreeStructure.Links[self.node_id][1] in self.TreeStructure.leaves:
                nodes_pr=smallest_branch(self.node_id, self.TreeStructure)
                nodes_pr.append(self.node_id)
            else:
                nodes_pr=nodes_to_prune(self.node_id, self.TreeStructure)

            #3rd Step: Make a copy of IDs_Depths of the original TreeStructure
            IDs_Depths_copy=self.TreeStructure.IDs_Depths.copy()

            #Define the necessary conditions to prune the tree
            #Store the indexes to remove in a variables
            subset = IDs_Depths_copy[(IDs_Depths_copy['Id'].isin(nodes_pr)) & (IDs_Depths_copy['Id'] >= self.node_id)]
            #Remove the redundant nodes and edges
            IDs_Depths_copy.drop(index=list(subset.index), inplace=True)

            #Re-index the IDs so that we have a normal sequence of IDs
            IDs_Depths_copy.index = range(len(IDs_Depths_copy))
            IDs_Depths_copy['Id']=IDs_Depths_copy.index


            #3rd Step: Construct the appropriate IDs_depths dataframe for the merged tree:
            merged_IDs_depths=pd.DataFrame(data={'Id': self.Nodes_dict['nodes_ids'], 'Depth':  self.Nodes_dict['nodes_depths'], 'Node_Samples': list(self.Nodes_dict['nodes_samples'].values())})

            #Add the additional columns required (nodes_labels, nodes_threshold etc.):
            Merged_Left_branch_nds=list(self.left_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
            Merged_Right_branch_nds=list(self.right_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
            Merged_Same_Depth, Merged_Left_Same_Depth, Merged_Right_Same_Depth=get_nodes_with_equal_depth(merged_IDs_depths, Merged_Left_branch_nds, Merged_Right_branch_nds)
            add_coordinates_columns(merged_IDs_depths, self, self.TreeStructure.IDs_Depths['Node_Samples'][0])


            #6th Step: #Transform the coordinates of the merged tree in appropriate coordinates in order to be able to map it on the plot we have original, pruned tree

            #Make a copy of the merged IDs_depths pandas dataframe object
            merged_copy=merged_IDs_depths.copy()

            #Nodes coordinates and Ids trabsformation
            if self.node_id != 0:
                #Coordinates
                merged_copy.loc[:,'x_coord'] = merged_copy.loc[:, 'x_coord'] + IDs_Depths_copy['x_coord'][IDs_Depths_copy.index[-1]]
                merged_copy.loc[:,'y_coord'] = merged_copy.loc[:, 'y_coord'] + (IDs_Depths_copy['y_coord'][IDs_Depths_copy.index[-1]] - merged_copy.loc[0, 'y_coord'])
                #Edges coordinates transformation
                add_coordinates_columns(merged_copy, self, self.TreeStructure.IDs_Depths['Node_Samples'][0], edges_only=True)
                #Ids transformation
                for nd_add in merged_copy.iloc[:, 0]:
                    merged_copy.iloc[nd_add, 0] = merged_copy.iloc[nd_add, 0] + IDs_Depths_copy.iloc[-1, 0] + 1
                    if nd_add == 0:
                        merged_copy.iloc[nd_add, 1] = self.TreeStructure.IDs_Depths.iloc[self.node_id, 1]
                    else:
                        merged_copy.iloc[nd_add, 1] = merged_copy.iloc[nd_add, 1] + merged_copy.iloc[0, 1]


                #Create the denominator in order for the width of the merged tree to be relative to the whole tree
                denominator=IDs_Depths_copy.loc[0, 'Node_Samples']/10
                merged_copy.loc[0, 'edges_width']= merged_copy.loc[0, 'Node_Samples']/denominator

            #Concatenate the two Dataframes
            
            frames = [IDs_Depths_copy, merged_copy]
            self.unified_tree_df= pd.concat(frames)
            self.unified_tree_df.reset_index(inplace=True)
            self.unified_tree_df.drop(columns='index', inplace=True)
            self.unified_tree_df.drop(columns='x_coord', inplace=True)
            self.unified_tree_df.drop(columns='y_coord', inplace=True)
            self.unified_tree_df.drop(columns='edges_x_coord', inplace=True)
            self.unified_tree_df.drop(columns='edges_y_coord', inplace=True)




            #Get the links of the unified tree        
            self.unified_Edges_dict=Edges(self.unified_tree_df)

            #Nodes info
            self.unified_Nodes_dict=Nodes(self.unified_tree_df)

            #Now we need to check whether the node_id input by the user belongs to a left branch. Why? In that case the nodes of the left branch in the final tree will be placed in the end of the dataframe.
            #The algorithm that add traces to the Image places the nodes according to their id and their place in the dataframe. We have two trees that are unified by concatenating them in a single dataframe.
            #In this dataframe the nodes that correspond to the branch that the user made changes are placed in the end. Therefore, the algorithm will plot first the big tree withou the left branch which 
            #was modified. But in this way the branch that was not modified and it was initially right it will be plotted as a left branch and the initially left branch (the one modified) will appear in 
            #the final tree as right. This is wrong. And in this case we need to put the nodes of the initially right branch in the end of the dataframe.

            #Check if the node_id input by the use is a left node
            if self.node_id in list(self.TreeStructure.Children_left):   #For the moment is good. But i need to see how it works also for dataframes
                #The node_id input by the user has changed id
                new_node_id=merged_copy.iloc[0, 0]
                #Look at the parent nodes of the tree
                for nds in list(self.unified_Edges_dict.Edges['links'].keys()):
                    #Identify its parent
                    if new_node_id in self.unified_Edges_dict.Edges['links'][nds] and self.unified_Edges_dict.Edges['links'][nds].index(new_node_id) == 1: 
                        #Assign the node id of the right sibling node to a variable
                        nd_id=self.unified_Edges_dict.Edges['links'][nds][0]
                        #Assign the nodes of this right branch to a variable
                        if self.unified_tree_df.loc[nd_id, 'nodes_labels'] != 'leaf_node':
                            chil_l=list(self.unified_tree_df.loc[[self.unified_Edges_dict.Edges['links'][nd_id][0]], 'nodes_labels'])
                            chil_r=list(self.unified_tree_df.loc[[self.unified_Edges_dict.Edges['links'][nd_id][1]], 'nodes_labels'])
                            if chil_l[0] == 'leaf_node' and chil_r[0] == 'leaf_node':
                                nodes_to_change=smallest_branch(nd_id, self.unified_tree_df)
                                nodes_to_change.sort()
                                count=1
                                for index in nodes_to_change: 
                                    #Change the id of the nodes of the right branch (add the id of the last node and then +1)
                                    self.unified_tree_df.loc[index, 'Id'] = self.unified_tree_df.iloc[-1, 0] + count
                                    self.unified_Nodes_dict.Nodes['nodes_ids'][index] = self.unified_tree_df.iloc[-1, 0] + count
                                    count+=1
                            else:
                                nodes_to_change=nodes_to_prune(nd_id, self.unified_tree_df)
                                nodes_to_change.sort()
                                count=1
                                for index in nodes_to_change: 
                                    #Change the id of the nodes of the right branch (add the id of the last node and then +1)
                                    self.unified_tree_df.loc[index, 'Id'] = self.unified_tree_df.iloc[-1, 0] + count
                                    self.unified_Nodes_dict.Nodes['nodes_ids'][index] = self.unified_tree_df.iloc[-1, 0] + count
                                    count+=1
                        elif self.unified_tree_df.loc[nd_id, 'nodes_labels'] == 'leaf_node':
                            count=1
                            self.unified_tree_df.loc[nd_id, 'Id'] = self.unified_tree_df.iloc[-1, 0] + count
                            self.unified_Nodes_dict.Nodes['nodes_ids'][nd_id] = self.unified_tree_df.iloc[-1, 0] + count


            #Sort the values according to id column (in this way the branch which is normally right and appear higher in the dataframe will be placed in the end)
            self.unified_tree_df=self.unified_tree_df.sort_values(['Id'])
            #Reset the index (this will create a continuous sequence of numbers that will be used for the indexing of a dataframe keeping the nodes ids of the right branch in the end.
            self.unified_tree_df.reset_index(inplace=True)
            #Assign a list which contains the sequence of numbers consistent with the indexing of the dataframe 
            self.unified_tree_df['Id']=list(range(0,len(self.unified_tree_df.loc[:,'Depth'])))
            
            #Drop the index column
            self.unified_tree_df.drop(columns='index', inplace=True)
            
            #Make sure that the right classes will be assigned to each node
            for cl in self.unified_tree_df.loc[:, 'Id']:
                self.unified_tree_df.loc[cl, 'nodes_classes']=self.classes_dict['Classes Labels'][np.argmax(self.unified_tree_df.loc[cl, 'nodes_values'])]

            #Get the links of the unified tree to account for the updates nodes above 
            self.unified_Nodes_dict=Nodes(self.unified_tree_df)
            self.unified_Edges_dict=Edges(self.unified_tree_df)

            #Get the structure of the unified tree
            self.unified_gr_str=graph_structure(len(self.unified_tree_df.loc[:,'Id']), self.unified_Edges_dict.Edges['edge_seq'], self.unified_tree_df.loc[:,'Node_Samples'], self.unified_tree_df.loc[:,'Id'], Best_first_Tree_Builder=True)
            add_coordinates_columns(self.unified_tree_df, self.unified_gr_str, self.unified_tree_df['Node_Samples'][0], x_y_coords_only=True, tree_links=self.unified_Edges_dict.Edges['links'])

            #Create the Figure
            self.Img=go.FigureWidget()


            Thres_dict, Hovertext, shape_dict, shape_lines_dict, color_impurities_dict, color_classes_dict, color_visual_style, subranges, upper_limit= format_graph(self.unified_Nodes_dict.Nodes, self.unified_Edges_dict.Edges, self.criterion, self.classes_dict, nodes_coloring=self.nodes_coloring, edges_shape='Lines', User_features_color_groups=self.User_features_color_groups, Best_first_Tree_Builder=True)

            #Add the necessary traces
            for ed in self.unified_tree_df.loc[1:, 'Id']:
                self.Img.add_trace(go.Scatter(x=[self.unified_tree_df['edges_x_coord'][ed][0], self.unified_tree_df['edges_x_coord'][ed][1]],    #self.TreePlot.fig
                                              y=[self.unified_tree_df['edges_y_coord'][ed][0], self.unified_tree_df['edges_y_coord'][ed][1]],
                                              mode='lines+text',
                                              line=dict(color='rgb(90,80,70)',
                                              width=self.unified_tree_df.loc[ed, 'edges_width'], # edges_widths[ed], 
                                              shape=shape_lines_dict),
                                              showlegend=False,
                                              hoverinfo='none',
                                              opacity=self.opacity_edges))

            for nd in self.unified_tree_df.loc[:, 'Id']:
                self.Img.add_trace(go.Scatter(x=[self.unified_tree_df.loc[nd, 'x_coord']],
                                              y=[self.unified_tree_df.loc[nd, 'y_coord']],
                                              mode='markers+text',
                                              marker=dict(symbol=shape_dict[nd],  size=self.mrk_size, color=color_visual_style[nd], line=dict(color='rgb(50,50,50)', width=1)),
                                              showlegend=False,
                                              text=Thres_dict[nd],
                                              textposition='top center',
                                              textfont={'size':self.txt_size}, 
                                              hovertext=Hovertext[nd],
                                              hoverlabel=dict(namelength=0),
                                              hoverinfo='text',
                                              opacity=self.opacity_nodes))

            Img_layout=self.TreePlot.fig.layout

            self.Img.update_layout(Img_layout)
            add_legend(self.Img, self.unified_tree_df, None, color_impurities_dict, subranges, color_classes_dict, shape_dict, nodes_coloring=self.nodes_coloring, User_features_color_groups=self.User_features_color_groups, txt_size=self.txt_size, mrk_size=self.mrk_size)
            self.Img.show()
        
        
        elif isinstance(self.Tree, pd.DataFrame):
            #1st step: Define the left, right branches, find the nodes that have equal depth and add the necessary columns to IDs_Depths pandas dataframe for the initial tree
            
            #Retrieve the links of the tree the parent nodes and the leaves nodes
            self.tree_links_init=get_links(self.Tree)
            self.parents_init, self.leaves_init=get_parents_and_leaves_nodes(self.Tree)
            self.left_chil_init, self.right_chil_init=get_left_and_right_children(self.Tree)

            #2nd step: Prune the initial tree at the node where the changes (feature or split point) will take place:
            if self.tree_links_init[self.node_id][0] in self.leaves_init and self.tree_links_init[self.node_id][1] in self.leaves_init:
                nodes_pr=smallest_branch(self.node_id, self.Tree)
                nodes_pr.append(self.node_id)
            else:
                nodes_pr=nodes_to_prune(self.node_id, self.Tree)

            #3rd Step: Make a copy of IDs_Depths of the original TreeStructure
            IDs_Depths_copy=self.Tree.copy()

            #Define the necessary conditions to prune the tree
            #Store the indexes to remove in a variables
            subset = IDs_Depths_copy[(IDs_Depths_copy['Id'].isin(nodes_pr)) & (IDs_Depths_copy['Id'] >= self.node_id)]  
            #Remove the redundant nodes and edges
            IDs_Depths_copy.drop(index=list(subset.index), inplace=True)

            #Re-index the IDs so that we have a normal sequence of IDs
            IDs_Depths_copy.index = range(len(IDs_Depths_copy))
            IDs_Depths_copy['Id']=IDs_Depths_copy.index


            #3rd Step: Construct the appropriate IDs_depths dataframe for the merged tree:
            merged_IDs_depths=pd.DataFrame(data={'Id': self.Nodes_dict['nodes_ids'], 'Depth':  self.Nodes_dict['nodes_depths'], 'Node_Samples': list(self.Nodes_dict['nodes_samples'].values())})

            #Add the additional columns required (nodes_labels, nodes_threshold etc.):
            Merged_Left_branch_nds=list(self.left_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
            Merged_Right_branch_nds=list(self.right_subtree_TreeStructure.IDs_Depths.loc[:,'Id'])
            Merged_Same_Depth, Merged_Left_Same_Depth, Merged_Right_Same_Depth=get_nodes_with_equal_depth(merged_IDs_depths, Merged_Left_branch_nds, Merged_Right_branch_nds)
            add_coordinates_columns(merged_IDs_depths, self, self.Tree['Node_Samples'][0])


            #6th Step: #Transform the coordinates of the merged tree in appropriate coordinates in order to be able to map it on the plot we have original, pruned tree

            #Make a copy of the merged IDs_depths pandas dataframe object
            merged_copy=merged_IDs_depths.copy()

            #Nodes coordinates and Ids trabsformation
            if self.node_id != 0:
                #Coordinates
                merged_copy.loc[:,'x_coord'] = merged_copy.loc[:, 'x_coord'] + IDs_Depths_copy['x_coord'][IDs_Depths_copy.index[-1]]
                merged_copy.loc[:,'y_coord'] = merged_copy.loc[:, 'y_coord'] + (IDs_Depths_copy['y_coord'][IDs_Depths_copy.index[-1]] - merged_copy.loc[0, 'y_coord'])
                #Edges coordinates transformation
                add_coordinates_columns(merged_copy, self, self.Tree['Node_Samples'][0], edges_only=True)
                #Ids transformation
                for nd_add in merged_copy.loc[:, 'Id']:
                    merged_copy.loc[nd_add, 'Id'] = merged_copy.loc[nd_add, 'Id'] + IDs_Depths_copy['Id'].index[-1] + 1
                    if nd_add == 0:
                        merged_copy.loc[nd_add, 'Depth'] = self.Tree.loc[self.node_id, 'Depth']
                    else:
                        merged_copy.loc[nd_add, 'Depth'] = merged_copy.loc[nd_add, 'Depth'] + merged_copy.loc[0, 'Depth']


                #Create the denominator in order for the width of the merged tree to be relative to the whole tree
                denominator=IDs_Depths_copy.loc[0, 'Node_Samples']/10
                merged_copy.loc[0, 'edges_width']= merged_copy.loc[0, 'Node_Samples']/denominator

            #Concatenate the two Dataframes
            
            frames = [IDs_Depths_copy, merged_copy]
            self.unified_tree_df= pd.concat(frames)
            self.unified_tree_df.reset_index(inplace=True)
            self.unified_tree_df.drop(columns='index', inplace=True)
            self.unified_tree_df.drop(columns='x_coord', inplace=True)
            self.unified_tree_df.drop(columns='y_coord', inplace=True)
            self.unified_tree_df.drop(columns='edges_x_coord', inplace=True)
            self.unified_tree_df.drop(columns='edges_y_coord', inplace=True)


            #Get the links of the unified tree        
            self.unified_Edges_dict=Edges(self.unified_tree_df)

            #Nodes info
            self.unified_Nodes_dict=Nodes(self.unified_tree_df)

            #Now we need to check whether the node_id input by the user belongs to a left branch. Why? In that case the nodes of the left branch in the final tree will be placed in the end of the dataframe.
            #The algorithm that add traces to the Image places the nodes according to their id and their place in the dataframe. We have two trees that are unified by concatenating them in a single dataframe.
            #In this dataframe the nodes that correspond to the branch that the user made changes are placed in the end. Therefore, the algorithm will plot first the big tree withou the left branch which 
            #was modified. But in this way the branch that was not modified and it was initially right it will be plotted as a left branch and the initially left branch (the one modified) will appear in 
            #the final tree as right. This is wrong. And in this case we need to put the nodes of the initially right branch in the end of the dataframe.

            #Check if the node_id input by the use is a left node
            if self.node_id in list(self.left_chil_init):   
                #The node_id input by the user has changed id
                new_node_id=merged_copy.iloc[0, 0]
                #Look at the parent nodes of the tree
                for nds in list(self.unified_Edges_dict.Edges['links'].keys()):
                    #Identify its parent
                    if new_node_id in self.unified_Edges_dict.Edges['links'][nds] and self.unified_Edges_dict.Edges['links'][nds].index(new_node_id) == 1: 
                        #Assign the node id of the right sibling node to a variable
                        nd_id=self.unified_Edges_dict.Edges['links'][nds][0]
                        #Assign the nodes of this right branch to a variable
                        if self.unified_tree_df.loc[nd_id, 'nodes_labels'] != 'leaf_node':
                            chil_l=list(self.unified_tree_df.loc[[self.unified_Edges_dict.Edges['links'][nd_id][0]], 'nodes_labels'])
                            chil_r=list(self.unified_tree_df.loc[[self.unified_Edges_dict.Edges['links'][nd_id][1]], 'nodes_labels'])
                            if chil_l[0] == 'leaf_node' and chil_r[0] == 'leaf_node':
                                nodes_to_change=smallest_branch(nd_id, self.unified_tree_df)
                                nodes_to_change.append(nd_id)
                                nodes_to_change.sort()
                                count=1
                                for index in nodes_to_change: 
                                    #Change the id of the nodes of the right branch (add the id of the last node and then +1)
                                    self.unified_tree_df.loc[index, 'Id'] = self.unified_tree_df['Id'].index[-1] + count
                                    self.unified_Nodes_dict.Nodes['nodes_ids'][index] = self.unified_tree_df['Id'].index[-1] + count
                                    count+=1
                            else:
                                nodes_to_change=nodes_to_prune(nd_id, self.unified_tree_df)
                                nodes_to_change.sort()
                                count=1
                                for index in nodes_to_change: 
                                    #Change the id of the nodes of the right branch (add the id of the last node and then +1)
                                    self.unified_tree_df.loc[index, 'Id'] = self.unified_tree_df['Id'].index[-1] + count 
                                    self.unified_Nodes_dict.Nodes['nodes_ids'][index] = self.unified_tree_df['Id'].index[-1] + count
                                    count+=1
                        elif self.unified_tree_df.loc[nd_id, 'nodes_labels'] == 'leaf_node':
                            count=1
                            self.unified_tree_df.loc[nd_id, 'Id'] = self.unified_tree_df['Id'].index[-1] + count
                            self.unified_Nodes_dict.Nodes['nodes_ids'][nd_id] = self.unified_tree_df['Id'].index[-1] + count


            #Sort the values according to id column (in this way the branch which is normally right and appear higher in the dataframe will be placed in the end)
            self.unified_tree_df=self.unified_tree_df.sort_values(['Id'])
            #Reset the index (this will create a continuous sequence of numbers that will be used for the indexing of a dataframe keeping the nodes ids of the right branch in the end.
            self.unified_tree_df.reset_index(inplace=True)
            #Assign a list which contains the sequence of numbers consistent with the indexing of the dataframe 
            self.unified_tree_df['Id']=list(range(0,len(self.unified_tree_df.loc[:,'Depth'])))
            
            #Drop the index column
            self.unified_tree_df.drop(columns='index', inplace=True)
            
            #Make sure that the right classes will be assigned to each node
            for cl in self.unified_tree_df.loc[:, 'Id']:
                self.unified_tree_df.loc[cl, 'nodes_classes']=self.classes_dict['Classes Labels'][np.argmax(self.unified_tree_df.loc[cl, 'nodes_values'])]

            #Get the links of the unified tree to account for the updates nodes above 
            self.unified_Nodes_dict=Nodes(self.unified_tree_df)
            self.unified_Edges_dict=Edges(self.unified_tree_df)

            #Get the structure of the unified tree
            self.unified_gr_str=graph_structure(len(self.unified_tree_df.loc[:,'Id']), self.unified_Edges_dict.Edges['edge_seq'], self.unified_tree_df.loc[:,'Node_Samples'], self.unified_tree_df.loc[:,'Id'], Best_first_Tree_Builder=True)
            add_coordinates_columns(self.unified_tree_df, self.unified_gr_str, self.unified_tree_df['Node_Samples'][0], x_y_coords_only=True, tree_links=self.unified_Edges_dict.Edges['links'])

            #Create the Figure
            self.Img=go.FigureWidget()


            Thres_dict, Hovertext, shape_dict, shape_lines_dict, color_impurities_dict, color_classes_dict, color_visual_style, subranges, upper_limit= format_graph(self.unified_Nodes_dict.Nodes, self.unified_Edges_dict.Edges, self.criterion, self.classes_dict, nodes_coloring=self.nodes_coloring, edges_shape='Lines', User_features_color_groups=self.User_features_color_groups, Best_first_Tree_Builder=True)

            #Add the necessary traces
            for ed in self.unified_tree_df.loc[1:, 'Id']:
                self.Img.add_trace(go.Scatter(x=[self.unified_tree_df['edges_x_coord'][ed][0], self.unified_tree_df['edges_x_coord'][ed][1]],    #self.TreePlot.fig
                                              y=[self.unified_tree_df['edges_y_coord'][ed][0], self.unified_tree_df['edges_y_coord'][ed][1]],
                                              mode='lines+text',
                                              line=dict(color='rgb(90,80,70)',
                                              width=self.unified_tree_df.loc[ed, 'edges_width'], # edges_widths[ed], 
                                              shape=shape_lines_dict),
                                              showlegend=False,
                                              hoverinfo='none',
                                              opacity=self.opacity_edges))

            for nd in self.unified_tree_df.loc[:, 'Id']:
                self.Img.add_trace(go.Scatter(x=[self.unified_tree_df.loc[nd, 'x_coord']],
                                              y=[self.unified_tree_df.loc[nd, 'y_coord']],
                                              mode='markers+text',
                                              marker=dict(symbol=shape_dict[nd],  size=self.mrk_size, color=color_visual_style[nd], line=dict(color='rgb(50,50,50)', width=1)),
                                              showlegend=False,
                                              text=Thres_dict[nd],
                                              textposition='top center',
                                              textfont={'size':self.txt_size},
                                              hovertext=Hovertext[nd],
                                              hoverlabel=dict(namelength=0),
                                              hoverinfo='text',
                                              opacity=self.opacity_nodes))

            Img_layout=self.TreePlot.fig.layout

            self.Img.update_layout(Img_layout)
            add_legend(self.Img, self.unified_tree_df, None, color_impurities_dict, subranges, color_classes_dict, shape_dict, nodes_coloring=self.nodes_coloring, User_features_color_groups=self.User_features_color_groups, txt_size=self.txt_size, mrk_size=self.mrk_size)
            self.Img.show()
            
            
    

    
    
    
    
    
class classify:
    '''
    This class contains functions/methods to classify samples based on a fitted decision tree.
    
    Inputs:
    TreeObject:           The classified tree. It can be: 1) An object of class sklearn.tree._classes.DecisionTreeClassifier 2) An object of class pandas.Dataframe
    samples_to_predict:   The samples to predict their classes.
    '''
    def __init__(self, TreeObject, samples_to_predict):
        self.TreeObject = TreeObject
        self.samples_to_predict = samples_to_predict
        
    def check_if_in_left_branch(leaf_node, left_branch, tree_links, nd_seed=0):
        '''
        Description:
        This function check if the leaf node is in the left branch which originates from node nd_seed. This function is used to get the classification rules of a leaf node.
        
        Inputs:
        leaf_node:    The leaf node for which we want to check whether it belongs to left branch
        left_branch:  List of nodes ids contained in left branch
        tree_links:   A dictionary containing the links of the tree
        nd_seed:      This is the root of the left branch
        
        Outputs:
        nd_seed:      If the leaf node is contained in the left branch the function will give the left children of the the initial nd_seed or if equal to the leaf node it will be the leaf node.
                      If it is not contained in the left branch it will be an empty list.
        boolean:      A boolean. It will be True is leaf_node is contained in the left branch. Otherwise it will be False.
        '''
        left_chil=tree_links[nd_seed][0]
        if isinstance(left_branch, list):
            if leaf_node in left_branch:
                nd_seed=left_chil
                boolean= True
            else:
                nd_seed=[]
                boolean=False
        elif isinstance(left_branch, int):
            if leaf_node == left_branch:
                nd_seed=left_branch
                boolean= True
            else:
                nd_seed=[]
                boolean=False

        return nd_seed, boolean

    def check_if_in_right_branch(leaf_node, right_branch, tree_links, nd_seed=0):
        '''
        Description:
        This function check if the leaf node is in the right branch which originates from node nd_seed. This function is used to get the classification rules of a leaf node.
        
        Inputs:
        leaf_node:     The leaf node for which we want to check whether it belongs to right branch
        right_branch:  List of nodes ids contained in right branch
        tree_links:    A dictionary containing the links of the tree
        nd_seed:       This is the root of the right branch
        
        Ouptuts:
        nd_seed:       If the leaf node is contained in the right branch the function will give the right children of the the initial nd_seed or if equal to the leaf node it will be the leaf node.
                       If it is not contained in the right branch it will be an empty list.
        boolean:       A boolean. It will be True if leaf_node is contained in the right branch. Otherwise it will be False.
        '''
        right_chil=tree_links[nd_seed][1]
        if isinstance(right_branch, list):
            if leaf_node in right_branch:
                nd_seed=right_chil
                boolean=True
            else:
                nd_seed=[]
                boolean=False
        elif isinstance(right_branch, int):
            if leaf_node == right_branch:
                nd_seed=right_chil
                boolean=True
            else:
                nd_seed=[]
                boolean=False
        return nd_seed, boolean

    def get_classification_rules(TreeObject, leaf_node):
        '''
        Description: 
        This function extracts and stores the classification rules for a leaf node in a dictionary
        
        Inputs:
        TreeObject:    An object of class pandas.DataFrame
        leaf_node:     Leaf node id for which the classification rules will be extracted
        
        Outputs:
        A dictionary that stores the classification rules for the specified leaf node.
        '''
        test_string=''
        test_string_left= '{} <= {}'
        test_string_right= '{} > {}'
        Rules={}
        nd_seed=0
        left_branch, right_branch=get_left_and_right_banches_nodes(TreeObject, 0)
        tree_links=get_links(TreeObject)
        while len(Rules)==0:
            if leaf_node == nd_seed:
                test_string=test_string 
                Rules[leaf_node]=test_string
                break
            elif leaf_node != nd_seed:
                nd_seed_left, boolean_left=classify.check_if_in_left_branch(leaf_node, left_branch, tree_links, nd_seed) 
                nd_seed_right, boolean_right=classify.check_if_in_right_branch(leaf_node, right_branch, tree_links, nd_seed)
                if boolean_left is True:
                    if nd_seed==0:
                        test_string=test_string + test_string_left.format(TreeObject.loc[nd_seed, 'nodes_labels'], TreeObject.loc[nd_seed, 'nodes_thresholds'])
                        nd_seed=nd_seed_left
                    else:
                        test_string=test_string + ' and ' + test_string_left.format(TreeObject.loc[nd_seed, 'nodes_labels'], TreeObject.loc[nd_seed, 'nodes_thresholds'])
                        nd_seed=nd_seed_left
                    if leaf_node == nd_seed:
                        test_string=test_string 
                        Rules[leaf_node]=test_string
                        break
                    else:
                        left_branch, right_branch=get_left_and_right_banches_nodes(TreeObject, node_id=nd_seed)
                elif boolean_right is True:
                    if nd_seed == 0:
                        test_string=test_string + test_string_right.format(TreeObject.loc[nd_seed, 'nodes_labels'], TreeObject.loc[nd_seed, 'nodes_thresholds'])
                        nd_seed=nd_seed_right
                    else:
                        test_string=test_string + ' and ' + test_string_right.format(TreeObject.loc[nd_seed, 'nodes_labels'], TreeObject.loc[nd_seed, 'nodes_thresholds'])
                        nd_seed=nd_seed_right
                    if leaf_node == nd_seed:
                        test_string=test_string 
                        Rules[leaf_node]=test_string
                        break
                    else:
                        left_branch, right_branch=get_left_and_right_banches_nodes(TreeObject, node_id=nd_seed)
        return Rules 


    def classify(self):
        '''
        This function predicts the classes of the samples input by the user. If TreeObject is sklearn.tree._classes.DecisionTreeClassifier it will give an numpy array with the 
        class predicted for each sample. If TreeObject is pandas dataframe it will give a pandas dataframe with the class predicted for each sample.
        '''
        if isinstance(self.TreeObject, sklearn.tree._classes.DecisionTreeClassifier):
            self.Predicted_classes=TreeObject.predict(self.samples_to_predict)
       
        elif isinstance(self.TreeObject, pd.DataFrame):
            parents, leaves = get_parents_and_leaves_nodes(self.TreeObject)
            self.rules={}
            indexes=list(self.samples_to_predict.index)
            for leaf in leaves:
                leaf_rule=list(classify.get_classification_rules(self.TreeObject, leaf).values())
                data_class_leaf=self.samples_to_predict.query(leaf_rule[0])
                indexes_class_leaf=list(data_class_leaf.index)
                class_list= [self.TreeObject.loc[leaf, 'nodes_classes']] * len(indexes_class_leaf)
                self.df=pd.DataFrame(data=class_list, index=indexes_class_leaf)
                if len(self.df) == 0:
                    self.df['Predicted_class'] = 'Empty'
                else:
                    self.df.columns=['Predicted_Class']
                    self.rules[leaf]=self.df

            self.predicted_classes=pd.concat([self.rules[leaf] for leaf in self.rules.keys()])
            self.predicted_classes=self.predicted_classes.reindex(self.samples_to_predict.index)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            