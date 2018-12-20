import json
from deepcopy import deepcopy
from pom_discrete_A3C_main import generate_NN_input
import utils
from torch.nn import unsqueeze

def generate_saliency(observation, net, game_tracker, record_json_dir, new_val=True):

    actual_state = generate_NN_input(10, observation, observation['step_count'], game_tracker)
    m_actual_state = v_wrap(actual_state).unsqueeze(0)
    #get the unmodified result from the network to compare modifications against.
    #Don't need hx or cx (still don't know wwhat they do)
    #Don't need the buffers as only this steps value is important.
    actual_action, _, _, actual_probs, actual_terminal_predicton = net.choose_action(m_actual_state)

    data = {}
    data['step'] = observation['step_count']
    data['actual_probs'] = actual_probs
    data['actual_action'] = actual_action
    data['actual_terminal_prediction'] = actual_terminal_predicton
    data['mods'] = []
    data['actions'] = []
    data['predictions'] = []

    board_size = len(observation['board'][0]) #assuming board is square
    for i in range(board_size):

        mod_list = []
        action_list = []
        prediction_list = []

        for j in range(board_size):
            mod_observation = deepcopy(observation)
            
            if new_val:
                mod_observation['board'][i][j] = 14 #first number that doesn't have a definition already
            else:
                #if the tile here is a passage, make it a wall. Otherwise, make this tile a passage.
                #prevents no change from occuring in passages
                mod_observation['board'][i][j] = 1 if mod_observation['board'][i][j] == 0 else 0

            #won't ever be a bomb in this tile. possible values are 0, 1, and 14
            mod_observation['bomb_blast_strength'][i][j] = 0
            mod_observation['bomb_life'][i][j] = 0

            this_state = generate_NN_input(10, mod_observation, mod_observation['step_count'], game_tracker)
            m_this_state = v_wrap(filtered_state).unsqueeze(0)
            this_action, _, _, this_probs, this_terminal_prediction = net.choose_action(m_this_state)
            
            diffs = [this_probs[k] - actual_probs[k] for k in this_action]

            mod_list.append(deepcopy(diffs))
            action_list.append(this_action)
            prediction_list.append(this_terminal_prediction)

        data['mods'].append(deepcopy(mod_list))
        data['actions'].append(deepcopy(action_list))
        data['predictions'].append(deepcopy(prediction_list))

    with open(record_json_dir + "/" + str(observation['step_count']) + ".json", 'w') as f:
        f.write(json.dumps(data))