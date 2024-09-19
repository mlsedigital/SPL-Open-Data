import os

import json
import pandas as pd
import numpy as np

## I'll add some multiprocessing once I know it works so I can save the dataset really really really quickly

def json_to_tab(file_path):
    """
    Func for converting the Json data into csv file for easy 
    data and sta analysis

    Args: 
        file_path (str): The location of the json file
    
    Returns:
        tab (list): a list of the data
        shot_id (int): the shot number from 1 to 125 
    """

    data_list = []
    with open(file_path, 'r') as file:
        data = json.load(file)

        ## data is a dictionary cotaining 
    keys = data.keys()
    for i in keys:
        if i != "tracking":
            data_list.append(data[i])

    for i in data['tracking']:
        #frame_id = str(i['frame']) # I don't need this now. I need to return this later. Although it could be more effective to save
        ## Not including time because there is a easy 1 to 1 correspondance so we can always get it back

        data_list.extend(i['data']['ball'])
        for j in i['data']['player'].keys():
            data_list.extend(i['data']['player'][j])

    ## list should have a size of 6 + 240*3*27 + 240*3
    return data_list

def convert_to_tab(json_dir):
    """
    This func does 2 main things. Call json_to_tab with the respective json paths. 
    Then along with column names returns a dataframe that can be saved as a .csv file

    Args:
        json_dir: (str) the folder directory containing all the data that needs to be converted (may be adapted to include additional players)
    
    Returns:
        dataset: (pd.dataset) This is the dataset with 125 rows 201-ish rows.
    """ 
    data_index = {}
    columns = []

    for index, filename in enumerate(os.listdir(json_dir)):
        f = os.path.join(json_dir, filename)

        data_index[index] = json_to_tab(f)

    with open(f, 'r') as file:
        data = json.load(file)
    
    columns.extend(data.keys())
    columns.pop() ## scary

    for i in data['tracking']:
        frame_id = str(i['frame'])
        for index_j, j in enumerate(i['data']['ball']):
            if index_j == 0:
                columns.append(f'{frame_id}_ball_x')
            elif index_j == 1:
                columns.append(f'{frame_id}_ball_y')
            elif index_j == 2:
                columns.append(f'{frame_id}_ball_z')
        
        for k in i['data']['player'].keys():
            for index_l, l in enumerate(i['data']['player'][k]):
                if index_l == 0:
                    columns.append(f'{frame_id}_{k}_x')
                elif index_l == 1:
                    columns.append(f'{frame_id}_{k}_y')
                elif index_l == 2:
                    columns.append(f'{frame_id}_{k}_z')

    data_csv = pd.DataFrame.from_dict(data_index, orient = 'index', columns = columns)
    
    data_csv.to_csv("P0001_shooting.csv")

    return None

convert_to_tab("C:\\Users\\User\\OneDrive\\Desktop\\DS Practice\\Sport\\SPL-Open-Data-Rahul\\basketball\\freethrow\\data\\P0001")






    
