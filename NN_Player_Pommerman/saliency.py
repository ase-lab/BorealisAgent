import json
import numpy
from copy import deepcopy
from pom_discrete_A3C_main import generate_NN_input
from utils import v_wrap
# from torch import unsqueeze, detach, numpy
import torch

def generate_saliency(observation, game_step, net, game_tracker, record_json_dir, actual_action, actual_probs, actual_pred, opponent_action, opponent_id, new_val=True):
    """Generate saliency data for a given observation

    observation     -- the observation to be used
    game_step       -- the current timestep of the game, needed because of an off-by-one in the observation's counter
    net             -- the network to generate saliency from
    game_tracker    -- Needed for generate_NN_input
    record_json_dir -- output directory for saliency data files. Should be kept with the data from env.render
    actual_action   -- action decided from actor distribution by the unmodified observation
    actual_probs    -- action distribution produced by the unmodified observation
    actual_pred     -- terminal prediction from the unmodified observation
    opponent_action -- the action the opponent took at this time step
    opponent_id     -- the id of the opponent
    new_val         -- if false uses passage/wall modifications, otherwise uses previously unknown value modifications
    """

    data = {}
    data['step'] = game_step
    data['actual_action'] = actual_action #have to pass actual action in from outside the func because the agent selects an action from the prob distribution in the main code
    data['actual_probs'] = actual_probs
    data['actual_pred'] = actual_pred
 
    data['opponent_action'] = opponent_action
    data['opponent_id'] = opponent_id
    data['state_value'] = net.critic_value([observation], game_step, game_tracker) #calculate the V func via a forward pass of the network.

    #below gets populated by the algorithm
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

            #remove the player from the 'alive' channel as well just to truly remove the data.
            if mod_observation['board'][i][j] >= 10:
                mod_observation['alive'].remove(mod_observation['board'][i][j])
            
            #replace this tile's value with a new value.
            if new_val:
                mod_observation['board'][i][j] = 14 #first number that doesn't have a definition already
            else:
                #if the tile here is a passage, make it a wall. Otherwise, make this tile a passage.
                #prevents no change from occuring in passages
                mod_observation['board'][i][j] = 1 if mod_observation['board'][i][j] == 0 else 0

            #won't ever be a bomb in this tile. possible values are 0, 1, and 14
            mod_observation['bomb_blast_strength'][i][j] = 0
            mod_observation['bomb_life'][i][j] = 0

            this_state = generate_NN_input(10, mod_observation, game_step, game_tracker)
            m_this_state = v_wrap(this_state).unsqueeze(0)
            this_action, _, _, this_probs, this_terminal_prediction = net.choose_action(m_this_state)

            diffs = [this_probs[k] - actual_probs[k] for k in range(len(this_probs))]

            mod_list.append(deepcopy(diffs)) # Have to deppcopy to avoid pointer hell.
            action_list.append(this_action) 
            prediction_list.append(this_terminal_prediction.detach().numpy()[0][0]) #detach removes gradient descent metadata, numpy transforms the value from a torch-specific format (tensor) to a float array

            #end tile

        data['mods'].append(deepcopy(mod_list))
        data['actions'].append(deepcopy(action_list))
        data['predictions'].append(deepcopy(prediction_list))

        #end row

    with open(f"{record_json_dir}/d{game_step:03d}.json", 'w') as f:
        #Uses MyEncoder to properly clean numpy data to types which json will serialize.
        #json.dumps won't serialize numpy data by default
        f.write(json.dumps(data, cls=MyEncoder))

def write_info(record_json_dir, game_step):
    data = {}
    data['totalSteps'] = game_step
    with open(f"{record_json_dir}/info.json", 'w') as f:
        f.write(json.dumps(data))

#https://fangyh09.github.io/TypeError-Object-of-type-float32-is-not-JSON-serializable/
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)