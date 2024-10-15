import os
import json
import pandas as pd

from utils import *

def json_to_tab(file_path, start_dict = None):
    """
    func takes the json data and then returns a dict with 90 keys with each key having exactly 240 values

    Args:
        file_path (str): Pretty obvious
        start_dict (dict): This is the dict that we append more data to. This could be empty #TODO: Add functionality for start_dict empty
    Returns:
        start_dict (dict): start_dict and then for each key append info to it.  (Index is participant id-trial number id-frame id) 
    """

    if start_dict == None:
        start_dict = {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    participant_id = data["participant_id"]
    trial_id = data["trial_id"]
    result = data["result"]
    landing_x = data["landing_x"]
    landing_y = data["landing_y"]
    entry_angle = data["entry_angle"]

    for i in data["tracking"]:
        dict_list = []
        dict_key = f"{participant_id}-{trial_id}-{i["frame"]}"
        dict_list.extend([participant_id, trial_id, result, landing_x, landing_y, entry_angle])
        dict_list.extend(i["data"]["ball"])
        
        for _, j in i["data"]['player'].items():
            dict_list.extend(j) 
        start_dict[dict_key] = dict_list

    return start_dict  

def convert_to_tab(json_dir):

    dict_output = {}
    columns = []
    
    for _, filename in enumerate(os.listdir(json_dir)):
        file_path = os.path.join(json_dir, filename)
        if not filename.endswith('.json'):
            continue
        dict_output = json_to_tab(file_path, dict_output)

    first_file_path = os.path.join(json_dir, os.listdir(json_dir)[0])    
    with open(first_file_path, 'r') as file:
        first_file_data = json.load(file)
    columns.extend([key for key in first_file_data.keys() if key != 'tracking'])
    
    for frame in first_file_data['tracking']:
        columns.extend(['ball_x', 'ball_y', 'ball_z'])
        for player_id in frame['data']['player'].keys():
            columns.extend([f'{player_id}_x', f'{player_id}_y', f'{player_id}_z'])
        break
    data_csv = pd.DataFrame.from_dict(dict_output, orient='index', columns=columns)
    data_csv.to_csv("P0001_shooting_alt.csv", index=False)
    return None

convert_to_tab(os.path.join(ROOT, "basketball", "freethrow", "data", "P0001"))
