"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
Adapted from (https://morvanzhou.github.io/).
"""
# TODO MAKE IT SO THAT IT CAN WORK WITH BOTH RECURRENT OR LINEAR LAYER WITH A FLAG
import sys
#sys.path.insert(1, '/home/bilal/anaconda3/envs/playground') # this should be enabled on the server run - check for memory issue

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record, save_rew_len, save_game_value_policy
import torch.nn.functional as F
import torch.multiprocessing as mp  # to use many CPU not threads - threads for individual players for simple agent
from torch.autograd import Variable
import multiprocessing
from copy import deepcopy
from shared_adam import SharedAdam, SharedRmsprop
from pommerman import make
from pommerman import agents
from pommerman import constants
from pommerman import utility
from pommerman import board_generator
import numpy as np
from random import randint, randrange, random
from Search import Search_Oracle
import queue
import inspect

from collections import defaultdict

import saliency

FILTER_RULES = False
RUN_ON_SERVER = False
DISPLAY_FIGURES = (not RUN_ON_SERVER)
USE_DENSE_REWARD = False
randomize_tile_size = False
SOFT_SWITCH = False
LSTM_ENABLED = False  # make it so that a Linear layer can be added easily
SAVE_VALUE_AND_POLICY_TO_TEXT = False # TODO this is for visualization purposes

SEARCH_AD_HOC_BUDGET = 20 # MAX OF 10 ACTIONS CAN BE ASKED FROM THE EXPERT DURING DRL GAME PLAY

SEARCH_EXPERT_ENABLED = False # This will keep MCTS game-tracker up & running - anythime ready for searching
IMITATE_MCTS = False * SEARCH_EXPERT_ENABLED
ADHOC_SEARCH = False * SEARCH_EXPERT_ENABLED

SEARCH_AS_SEPERATE_WORKER = False * SEARCH_EXPERT_ENABLED
NUMBER_OF_PLANNER = 2 # THIS CAN BE THE NUMBER OF PLANNERS OVERALL - must be less than the number of cores ...
# Code Merge - Pablo
UPDATE_GLOBAL_ITER = 1000  # TODO THIS CAN BE SET TO 5 - as no intermediate reward only update when game ends
GAMMA = 0.999
IMITATION_LEARNING_DECAY = 0.9999


MAX_EP = 3 # how many games
CPU_COUNT = 1
PURE_LEARNING = 1 # 50 % OF THE EPISODES WILL BE IMITATION-FREE

LR = 0.0001

OPPONENT_SIZE = 1
OPPONENT_CURRICULUM = False
CUSTOM_GENERATE_BOARD = False
CUSTOM_BOARD_SHUFFLE = True


from random import choice
R_KILL_BOMBER = 0.1
R_POWER_KICK = 0.03
R_POWER_BLAST = 0.02
R_POWER_BOMB = 0.01
R_KICK_BOMB = 0.001
R_BLAST_WOOD = 0.001
R_PLACED_BOMB = 0.0001
R_BEING_ALIVE = 0.00000001
movements = [constants.Action.Left, constants.Action.Right, constants.Action.Up,
             constants.Action.Down]

CustomMap = None
CustomMapShuffle = None
CUSTOM_BOARD_SIZE = 8 # SET THIS DEFAULT TO 11 FOR THE GAME
IMITATION_CONSTANT = 0.1 # DURING IMITATION, DECREASE ACTOR-CRITIC LOSS CONTRIBUTION

CRITIC_LOSS_W = 1
ACTOR_LOSS_W = 1
SUP_LOSS_W = 1
TERMINAL_LOSS_W = 1
ENT_LOSS_W = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.a_dim = 6
        self.cnn_last_dim = 32

        board_size = CUSTOM_BOARD_SIZE

        self.lin_input = self.cnn_last_dim * board_size * board_size  # TODO change this to

        self.conv1 = nn.Conv2d(28, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, self.cnn_last_dim, kernel_size=3, stride=1, padding=1)

        # Add LSTMCell here - see how the hidden state will be reset each turn
        self.post_cnn_layer_dim = 128

        if LSTM_ENABLED:
            self.lstm = nn.LSTMCell(self.lin_input, self.post_cnn_layer_dim)
        else:
            self.post_cnn_layer = nn.Linear(self.lin_input, self.post_cnn_layer_dim )
            set_init([self.post_cnn_layer])

        #self.ln2 = nn.Linear(self.post_cnn_layer_dim, 128)
        self.head_policy_actor = nn.Linear(self.post_cnn_layer_dim, self.a_dim)
        self.head_value_critic = nn.Linear(self.post_cnn_layer_dim, 1)

        #self.ln3 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim) # for game end prediction - TODO this can be an LSTM indeed
        self.head_terminal_predictor = nn.Linear(self.post_cnn_layer_dim,1) # single number - between 0 and 1 - to predict how close we are to the end of the game

        set_init([self.conv1, self.conv2, self.conv3, self.conv4, self.head_policy_actor, self.head_value_critic, self.head_terminal_predictor])  # disable this for now - random init is used

        if LSTM_ENABLED is False:
            set_init([self.post_cnn_layer])

        self.distribution = torch.distributions.Categorical

    def forward(self, x, hx=None, cx=None):
        #print(f" size of hx is {hx.shape}")

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, self.lin_input)  # to fix the shape before fully connected layer

        if LSTM_ENABLED:
            hx, cx = self.lstm(x, (hx,cx))
            x = hx
        else:
            x = F.elu(self.post_cnn_layer(x))

        #x = F.elu(self.ln2(x))

        logits = self.head_policy_actor(x)
        values = self.head_value_critic(x)

        #y = F.elu(self.ln3(x))
        end_predict = F.sigmoid(self.head_terminal_predictor(x)) # scalar from 0 to 1

        return logits, values, hx, cx, end_predict

    def choose_action(self, s, hx=None, cx=None, value_viz_buffer=None, policy_viz_buffer=None):
        #print(s)
        # print(f"set to eval {s.shape}")
        self.eval()  # to freeze weights
        logits, value, hx, cx, end_predict = self.forward(s, hx, cx)

        prob = F.softmax(logits, dim=1).data
        #prob_np = prob.data.numpy()
        #print(f"probs are {prob_np}")
        #print(f"probs are {prob_np[0][0]}")


        if value_viz_buffer is not None: # TODO these parts have been added to log game data for visualization
            value_viz_buffer.append((value.data.numpy().flatten()))
        if policy_viz_buffer is not None:
            policy_viz_buffer.append((prob.data.numpy().flatten()))

        m = self.distribution(prob)

        #print(f"state value is {value}")

        return m.sample().numpy()[0], hx, cx, prob.data.numpy()[0], end_predict # also return probs

    def loss_func(self, s, a, v_t, imitation_continue, is_worker_pure_planner, hx=None, cx=None):
        self.eval()
        logits, values, _, _, end_predict = self.forward(s, hx, cx )
        # CRITIC LOSS
        td = 0.5*(v_t - values)
        c_loss = td.pow(2)  # critic loss

        #ACTOR LOSS
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()  # why detach() here
        a_loss = -exp_v  # actor loss

        #ENTROPY LOSS
        ent_loss = -(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1)).sum(1) # to enforce exploration


        #GAME END PREDICTION LOSS (TYPE OF AUX LOSS)
        end_predict_activated = F.sigmoid(end_predict) # to get between 0 to 1
        #print(f"predictions are for {end_predict_activated.shape} is {end_predict_activated}")
        # assume true labels are linearly increase as we get close to end of the game
        end_np = np.arange(1,(len(a)+1))/(len(a)+1)
        end_predict_labels = v_wrap(end_np, np.float32).unsqueeze(1) # get torch version.
        #print(f"labels are for {end_predict_labels.shape} is {end_predict_labels}")
        # compute loss for end-game prediction
        terminal_pred_loss = F.mse_loss(end_predict_activated, end_predict_labels)
        #print(f" term loss is {terminal_pred_loss}")

        if imitation_continue or is_worker_pure_planner is True: # IMITATION LOSS generated by either imitating a rule-based agent or based on mcts agent
            # compute supervised loss for cross-entrophy during imitation - or ad-hoc imitation phase
            one_hot_expert_move = torch.zeros([len(a), 6], dtype=torch.float32)
            indexer = list(range(0, len(a)))
            one_hot_expert_move[indexer, a[indexer]] = 1.0
            #sup_loss = -(F.log_softmax(logits, dim=1) * F.softmax(one_hot_expert_move, dim=1)).sum(1)
            sup_loss = -(F.log_softmax(logits, dim=1) * one_hot_expert_move).sum(1)
            #print(f" sup loss is {sup_loss}")
            total_loss = ( IMITATION_CONSTANT*CRITIC_LOSS_W * c_loss + IMITATION_CONSTANT*ACTOR_LOSS_W * a_loss - ENT_LOSS_W * ent_loss + SUP_LOSS_W * sup_loss + IMITATION_CONSTANT*TERMINAL_LOSS_W*terminal_pred_loss ).mean()  # total loss for the shared parameters
        else:
            total_loss = ( CRITIC_LOSS_W * c_loss + ACTOR_LOSS_W * a_loss - ENT_LOSS_W * ent_loss + TERMINAL_LOSS_W*terminal_pred_loss ).mean()  # total loss for the shared parameters

        return total_loss

    def save_trained_model(self, episode_number):
        folder_name = 'models'
        if os.path.isdir(folder_name) is False:
            print("creating the folder")
            os.mkdir(folder_name)

        here = os.path.dirname(os.path.realpath(__file__))

        if LSTM_ENABLED:
            model_filename = 'A3C_global_NN' + str(episode_number) + '.pt'
        else:
            model_filename = 'A3C_global_NN_linear' + str(episode_number) + '.pt'

        filepath = os.path.join(here, folder_name, model_filename)

        torch.save(self.state_dict(), filepath)




    def load_trained_model(self, filename):  # TODO FOR INFERENCE
        self.load_state_dict(torch.load(filename))

    def load_trained_model(self, moreTraining):  # for further training
        pass





# Pommerman Domain Information to Simulate the Game

all_actions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down, constants.Action.Bomb]
directions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
_directionsAsVector = {constants.Action.Up: (0, 1),
                       constants.Action.Down: (0, -1),
                       constants.Action.Left: (1, 0),
                       constants.Action.Right: (-1, 0),
                       constants.Action.Stop: (0, 0)}
_directionsAsList = _directionsAsVector.items()

def _filter_legal_actions(state):
    my_position = tuple(state['position'])
    board = np.array(state['board'])
    enemies = [constants.Item(e) for e in state['enemies']]
    ret = [constants.Action.Bomb]
    for direction in directions:
        position = utility.get_next_position(my_position, direction)
        if utility.position_on_board(board, position) and utility.position_is_passable(board, position, enemies):
            ret.append(direction)
    return ret

def _walls(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Rigid.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _wood_positions(board):
    ret = []
    locations = np.where(board == constants.Item.Wood.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append((r,c))
    return ret


def _flame(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _passage(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Passage.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _flame_counter(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 2 # TODO hardcode 2 is flame life if change - then this must be changed as well
    return ret

def _wood(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Wood.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _bombs(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Bomb.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_ExtraBomb(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.ExtraBomb)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_IncRange(board):
    ret = np.zeros(board.shape)
    locations = np.where( board == constants.Item.IncrRange )
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_Kick(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Kick )
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _enemies_positions(board, enemies):
    ret = []
    for e in enemies:
        locations = np.where(board == e.value)
        for r, c in zip(locations[0], locations[1]):
            ret.append((r, c))
    return ret

def _agents_positions(board):
    ret = {}
    agent_ids = [10,11,12,13]
    for e in agent_ids:
        locations = np.where(board == e)
        for r, c in zip(locations[0], locations[1]):
            ret[e-10] = (r,c)
    return ret

def convert_bombs(bomb_map):
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({'position': (r, c), 'blast_strength': int(bomb_map[(r, c)])})
    return ret

def convert_flames(flame_map):
    ret = []
    locations = np.where(flame_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({'position': (r, c)})
    return ret


    # agent alive or not?

    # for flames
    # binary - flame locations
    # integer - flame time remaining

def diagonalAgentId(ourAgentId):
    list = [constants.Item.Agent0.value ,constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value] # agent IDs from the list
    return list[(list.index(ourAgentId)+2)%len(list)]


def nonDiagonalAgents(ourAgentId):
    # returns two adjacent (non-diagonal) enemy ids
    list = [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]  # agent IDs from the list
    # get diagonal agent - remove diagonal and itself , return the other remaining two.
    list.remove(diagonalAgentId(ourAgentId))
    list.remove(ourAgentId)
    return list

def generate_NN_input(my_game_id ,observation, time_step, game_tracker):
    print()
    print(observation)

    numberOfChannels = 28
    current_board = np.array(observation['board'])  # done

    board_size = current_board.shape[0] # TODO ASSUMING THE BOARD IS SQUARE

    ret = np.zeros((numberOfChannels, board_size, board_size))  # inspired from backplay pommerman paper

    index_bomb_strength = 0  # integer
    index_bomb_t_explode = 1  # integer

    index_agent_loc = 2
    index_agent_bomb_count = 3
    index_agent_blast = 4
    index_agent_can_kick = 5
    index_agent_has_teammate = 6

    index_agent_mate_or_enemy1_loc = 7
    index_agent_enemy2_loc = 8
    index_agent_enemy3_loc = 9

    index_passage_loc = 10
    index_wall_loc = 11       # binary
    index_wood_loc = 12       # binary
    index_flame_loc = 13      # binary

    index_powerup_extrabomb = 14
    index_powerup_increaseblast = 15
    index_powerup_cankick = 16
    index_game_timestep = 17 # float % of game episode

    index_mate_or_enemy1_bomb_count = 18
    index_mate_or_enemy1_blast = 19
    index_mate_or_enemy1_cankick = 20

    index_enemy2_bomb_count = 21
    index_enemy2_blast = 22
    index_enemy2_cankick = 23

    index_enemy3_bomb_count = 24
    index_enemy3_blast = 25
    index_enemy3_cankick = 26

    index_flame_lifetime = 27

    #print("current board is \n")
    #print(current_board, "\n")

    alives = np.array(observation['alive'])

    our_agent_pos = np.array(observation['position'])
    #print("our position is ", our_agent_pos)
    our_agentid = my_game_id
    #print("our agent id is ", our_agentid)

    agent_positions = _agents_positions(current_board) # returns all the agent locations

    #print("agent positions are", agent_positions, " \n")

    enemies = np.array(observation['enemies'])

    #print("enemies are ", enemies)



    #print("alive agents are ", alives)

    ret[index_bomb_strength, :, :] = np.array(observation['bomb_blast_strength'])  # done

    #print( " bomb blast strength is ", ret[index_bomb_strength, :, :] )

    ret[index_bomb_t_explode, :, :] = np.array(observation['bomb_life'])  # done

    #print (" bomb time to explode is ", ret[index_bomb_t_explode,:,:])

    # print(f"agent locations {agent_positions} and our aget id {our_agentid}")

    if our_agentid in alives: # otherwise return our agent location empty or noT? TODO
        ret[index_agent_loc,agent_positions[our_agentid-10][0], agent_positions[our_agentid-10][1]] = 1 # binary map for our agent

    #print(" channel for our agent location ",ret[index_agent_loc,:,:])

    ret[index_agent_bomb_count,:,:] = np.ones((board_size,board_size)) * np.array(observation['ammo']) # integer full map

    #print(" channel for our agent ammo size ",ret[index_agent_bomb_count,:,:])

    ret[index_agent_blast,:,:] = np.ones((board_size, board_size)) * np.array(observation['blast_strength']) # integer map full

    #print(" channel for agent blast strength ",ret[index_agent_blast,:,:])

    ret[index_agent_can_kick,:,:] = np.ones((board_size, board_size)) * np.array(observation['can_kick']) # binary map full

    #print(" channel for agent kick ",ret[index_agent_can_kick,:,:])

    diagonalPlayer = diagonalAgentId(our_agentid) # TODO remove lines setting to zeros after testing

    if diagonalPlayer in alives:
        ret[index_agent_mate_or_enemy1_loc, agent_positions[diagonalPlayer - 10][0], agent_positions[diagonalPlayer - 10][1]] = 1  # set location to binary
        ret[index_mate_or_enemy1_bomb_count, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[diagonalPlayer - 10].ammo)  # binary map full
        ret[index_mate_or_enemy1_blast, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[diagonalPlayer - 10].blast_strength)  # binary map full
        ret[index_mate_or_enemy1_cankick, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[diagonalPlayer - 10].can_kick)  # binary map full

        if np.array(observation['teammate']) != constants.Item.AgentDummy: # team game
            ret[index_agent_has_teammate,:,:] = np.ones((board_size, board_size)) # full binary map

    #print("team mate channel is ", ret[index_agent_has_teammate,:,:])
    #print("team mate location channel is ", ret[index_agent_mate_or_enemy1_loc, :, :])

    otherTwoEnemyIds = nonDiagonalAgents(our_agentid) # returns an array with two elements

    #print("other agents are ", otherTwoEnemyIds)


    # first layer is the smallest id among two
    # second layer is the higher one

    if otherTwoEnemyIds[0] in alives:
        ret[index_agent_enemy2_loc,agent_positions[otherTwoEnemyIds[0]-10][0], agent_positions[otherTwoEnemyIds[0]-10][1]] = 1 # set location to binary
        ret[index_enemy2_bomb_count, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[0] - 10].ammo)  # binary map full
        ret[index_enemy2_blast, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[0] - 10].blast_strength)  # binary map full
        ret[index_enemy2_cankick, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[0] - 10].can_kick)  # binary map full

    if otherTwoEnemyIds[1] in alives:
        ret[index_agent_enemy3_loc, agent_positions[otherTwoEnemyIds[1]-10][0], agent_positions[otherTwoEnemyIds[1]-10][1]] = 1  # set location to binary
        ret[index_enemy3_bomb_count, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[1] - 10].ammo)  # binary map full
        ret[index_enemy3_blast, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[1] - 10].blast_strength)  # binary map full
        ret[index_enemy3_cankick, :, :] = np.ones((board_size, board_size)) * np.array(game_tracker.bombermen_list[otherTwoEnemyIds[1] - 10].can_kick)  # binary map full

    #print("enemy2 location channel is ", ret[index_agent_enemy2_loc, :, :])
    #print("enemy3 location channel is ", ret[index_agent_enemy3_loc, :, :])

    ret[index_passage_loc,:,:] = _passage(current_board) # done
    ret[index_wall_loc,:,:] = _walls(current_board) # done
    ret[index_wood_loc,:,:] = _wood(current_board)  # done
    ret[index_flame_loc,:,:] = _flame(current_board) # done


    #print("passage is ", ret[index_passage_loc,:,:])
    #print("wall is ", ret[index_wall_loc,:,:])
    #print("wood is ", ret[index_wood_loc,:,:])
    #print("flame is ", ret[index_flame_loc,:,:])

    ret[index_powerup_extrabomb,:,:] = _powerup_ExtraBomb(current_board) # done
    ret[index_powerup_increaseblast, :, :] = _powerup_IncRange(current_board) # done
    ret[index_powerup_cankick, :, :] = _powerup_Kick(current_board) # done
    ret[index_game_timestep,:,:] = np.ones((board_size,board_size))*(time_step/800.0) # done TODO 800 game length here

    ret[index_flame_lifetime,:,:] = game_tracker.global_flame_map # just access the global flame map


    #for i in range(numberOfChannels):
    #    print(f"generated channels {i} is {ret[i,:,:]} \n\n\n\n")

    # TODO OPPONENT CHANNELS READY NOW

    return ret

    #print(" >>>>>>>>>>>>>>>>>>>>>>>>> \n")

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, g_res_list, g_res_lock, g_res_length_list, g_res_length_lock,
                 g_loss_list, g_loss_lock, name, global_imitation_p):
        super(Worker, self).__init__()
        self.g_imitation_p = global_imitation_p

        # print(inspect.getmembers(sys.modules[__name__], inspect.isclass))
        self.agent_no_cur = [  # STANDART WAY
            agents.StaticAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
        ]

        self.curriculum_agent_list = [[
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ],
        [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ],
        [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]]

        self.agent_list = self.agent_no_cur

        self.generatedBoard = CUSTOM_GENERATE_BOARD
        self.shuffle = CUSTOM_BOARD_SHUFFLE
        self.config = 'PommeFFACompetition-v0'
        self.size = CUSTOM_BOARD_SIZE
        self.step_count = 0
        self.rigid = 14
        self.wood = 14
        self.items = 8
        self.n_opponents = OPPONENT_SIZE
        self.kick = False
        self.ammo = 1
        self.blast = 2

        ##### Pablo
        if self.generatedBoard:

            self.env = board_generator.randomizeWithAgents(self.config, self.agent_list, self.size, self.rigid,
                                                           self.wood, self.items, self.step_count, self.n_opponents,
                                                           self.kick, self.ammo, self.blast)
            with open("board-used" + str(self.shuffle) + ".txt", 'w') as file:
                info = self.env.get_json_info()
                file.write(json.dumps(info, sort_keys=True, indent=4))
        else:
            self.env = make('PommeFFACompetition-v0', self.agent_list).unwrapped

        self.training_agent_index = 0  # TODO GENERALIZE THIS TO ANY START POSITION
        self.worker_id = name
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r = global_ep, global_ep_r
        self.gnet, self.opt = gnet, opt
        self.lnet = Net()  # local network

        self.value_buffer = None # these are to log data for visualization purposes
        self.policy_buffer = None

        if SAVE_VALUE_AND_POLICY_TO_TEXT and self.worker_id == 0:
            self.value_buffer = [] # save them - at the end to a file
            self.policy_buffer = []

        self.hx = None
        self.cx = None

        self.expert_searcher = None

        self.pure_searcher = False  # set this based - and let it behave as a searcher

        if self.worker_id < NUMBER_OF_PLANNER:
            self.pure_searcher = True # set this based - and let it behave as a searcher

        self.expert_query_histogram = np.zeros((10,801)) # 10 bins during the course of training

        #print(f"hx is {self.hx}")

        self.lnet.load_state_dict(
            gnet.state_dict())  # I assume initially all worker and global networks must be the same

        # self.imitation_schedule = IMITATION_RATE #* (float(CPU_COUNT - self.worker_id) / CPU_COUNT) #
        self.imitation_continue = True
        # print(f"worker imit schedue is {self.imitation_schedule}")

        if self.name == 'w0':
            global CustomMap, CustomMapShuffle
            CustomMap = self.generatedBoard
            CustomMapShuffle = self.shuffle
        # print(f" worker name is {self.name} and worker network is {self.lnet}")

        self.g_res_list = g_res_list
        self.g_res_lock = g_res_lock

        self.g_res_length_list = g_res_length_list
        self.g_res_length_lock = g_res_length_lock

        self.g_loss_list = g_loss_list
        self.g_loss_lock = g_loss_lock

    def run(self):
        while self.g_ep.value < MAX_EP:

            total_step = 1
            game_step = 0

            if OPPONENT_CURRICULUM:
                self.agent_list = deepcopy(self.curriculum_agent_list[((3*self.g_ep.value)//MAX_EP)])
                #print(self.agent_list)
                for id, agent in enumerate(self.agent_list):
                    #assert isinstance(agent, agents.BaseAgent)
                    agent.init_agent(id, 'constants.GameType.FFA') # change this for other versions ...

                self.env.set_agents(self.agent_list)

            # print(f" total step begin is {total_step}")
            # print(f" worker {self.name}")

            #print(self.agent_list)

            if randomize_tile_size:
                if self.shuffle and self.generatedBoard:
                    x = randrange(2,self.rigid, 2)
                    y = randrange(4,self.wood, 2)
                    z = randrange(2,y, 2)

                    self.env = board_generator.shuffle(self.env, self.config, self.size, x, y, z, self.step_count, self.n_opponents)
            else:
                if self.shuffle:
                    self.env = board_generator.shuffle(self.env, self.config, self.size, self.rigid, self.wood, self.items, self.step_count, self.n_opponents)

            s = self.env.reset()
            self.expert_searcher = Search_Oracle(constants.GameType.FFA, self.env)  # initiate search

            # print(f"worker expert searcher is initialized within the worker")

            if LSTM_ENABLED:
                self.hx = torch.zeros(1, self.lnet.post_cnn_layer_dim) # check if we need to zero both for lstm cell for episode
                self.cx = torch.zeros(1, self.lnet.post_cnn_layer_dim)

            self.init_rewards()

            buffer_s, buffer_a, buffer_r = [], [], []
            #buffer_hx, buffer_cx = [], [] # for lstm hidden and cell states

            buffer_hx = self.hx
            buffer_cx = self.cx

            if SAVE_VALUE_AND_POLICY_TO_TEXT and self.worker_id == 0:
                pass
                #self.env.render(mode=False, record_json_dir=record_json_dir) # TODO THIS IS TO SAVE GAME STATES ---

            ep_r = 0.

            ####SOFT SWITCH GET PROBABILITIES
            if SOFT_SWITCH:
                if self.g_imitation_p.value > 0:
                    rnd = random()
                else:
                    rnd = 1


                if rnd < self.g_imitation_p.value:  # firstly just follow the simple agent footsteps
                    #              print(self.g_imitation_p.value, rnd, 'imitate')
                    self.imitation_continue = True
                    imitation_in_episode = True
                else:
                    #              print(self.g_imitation_p.value, rnd, 'learn')
                    self.imitation_continue = False
                    imitation_in_episode = False

                if self.g_imitation_p.value > 0:
                    self.g_imitation_p.value *= IMITATION_LEARNING_DECAY

                if self.g_imitation_p.value <= 0.001:
                    self.imitation_continue = False
                    self.g_imitation_p.value = 0
                ###

            while True:
                if self.name == 'w0' and RUN_ON_SERVER is False:  # TODO not displaying now
                    self.env.render()
                    self.env.render(mode=False, record_json_dir="out")
                    saliency.generate_saliency(s[0], self.lnet, self.expert_searcher.game_tracker, "out", False)
                    pass

                self.expert_searcher.keep_tracking_game(s[0])

                actions = self.env.act(s)
                filtered_state = generate_NN_input(10, s[0], game_step, self.expert_searcher.game_tracker)
                m_filtered_state = v_wrap(filtered_state).unsqueeze(0)

                NN_action_used = True
                nn_action, self.hx, self.cx, NN_probs, terminal_predicton = self.lnet.choose_action(m_filtered_state,
                                                                                                     self.hx, self.cx,
                                                                                                     self.value_buffer,
                                                                                                     self.policy_buffer)  # overload the first action
                FLAG_FOR_NATHAN = True

                if FLAG_FOR_NATHAN:
                    print(f'taking action {nn_action} propbs were {NN_probs}')

                #TODO Note for Nathan
                #Original state the agent gets from Pommerman API is s[0]
                #We generate a set of feature channels with the generate_NN_input method and obtain filtered_state, and this is fed to the NN
                #At this point, nn_action keeps the taken action, and NN_probs has the softmax probabilities for the policy network.
                #Indeed, choose_action method has everything you need, it makes a forward pass through the NN, and produces policy probabilities, from which agent takes a softmax action

                if SOFT_SWITCH is False:
                    if self.g_ep.value >= (1 - PURE_LEARNING) * MAX_EP:  # firstly just follow the simple agent footsteps
                        self.imitation_continue = False
                    else:
                        actions[0] = np.int64(actions[0])
                        NN_action_used = False
                else: ## SOFT Switch
                    if imitation_in_episode:
                        actions[0] = np.int64(actions[0])
                        NN_action_used = False


                if NN_action_used is True:
                    actions[0] = nn_action


                #if FILTER_RULES:
                #    actions_ok = self.filter_legal_actions(s[0])
                #    #     print('filter rules', actions[0], actions_ok)
                #    if actions[0] not in actions_ok:
                #        if NN_action_used: # then select from safe actions based on the NN policy actor probabilities
                #            #print(f" prob from NN are {NN_probs}")
                #            NN_probs[np.setdiff1d([0, 1, 2, 3, 4, 5], actions_ok)] = 0
                #            #print(f" clipped prob from NN are {NN_probs}")
                #           #print(f" normalized prob from NN are {NN_probs/sum(NN_probs)} \n\n\n")
                #
                #            actions[0] = np.random.choice(6, 1, replace=False, p=NN_probs/(sum(NN_probs)))
                #        else:
                #            actions[0] = np.int64(choice(actions_ok)) # if imitating, pick a uniform random safe action

                if SEARCH_EXPERT_ENABLED:
                    if (terminal_predicton > 0.9 and ADHOC_SEARCH and self.expert_searcher.usage_counter < SEARCH_AD_HOC_BUDGET) or (NN_action_used is False and IMITATE_MCTS) or self.pure_searcher is True: # tie this to value prediction or something else to trigger search during the game
                        mcts_action = self.expert_searcher.return_action(s[0])
                        actions[0] = np.int64(mcts_action)
                        self.expert_query_histogram[self.g_ep.value//MAX_EP][game_step] += 1 # keep histogram
                        #print(f"action by the expert search  is {mcts_action} at step {game_step}") # incorporate some sort of loss by the expert action

                #if LSTM_ENABLED:
                #    self.hx = Variable(self.hx.data)
                #    self.cx = Variable(self.cx.data)

                previous_s = deepcopy(s[0])
                s_, r, done, info = self.env.step(actions)

                # PABLO:adding rewards
                if done:
                    print(f" reward is {r[0]}")


                # print(f" game step {game_step} and agent status {self.agent_list[self.training_agent_index].is_alive}")

                if self.agent_list[self.training_agent_index].is_alive is False:
                    # print(s_[0])
                    done = True



                if USE_DENSE_REWARD and done is False:
                    r[0] = self.get_reward(previous_s, actions[0], s_[0])

                ep_r += r[0]  # TODO  for pommerman, this is the only non-zero time-step for reward

                buffer_a.append(actions[0])
                buffer_s.append(m_filtered_state)
                buffer_r.append(r[0])  # TODO are we sending step reward or total reward upto time step t

                if done is False and LSTM_ENABLED:
                    buffer_hx = torch.cat((buffer_hx, self.hx), 0)
                    buffer_cx = torch.cat((buffer_cx, self.cx), 0)

                game_step += 1

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    # print(f" time to push pull by {self.name}")
                    #print(f"reward is {r[0]}")
                    filtered_next_state = generate_NN_input(10, s_[0], game_step, self.expert_searcher.game_tracker)
                    m_filtered_next_state = v_wrap(filtered_next_state).unsqueeze(0)
                    #push_and_pull(self, self.opt, self.lnet, self.gnet, done, m_filtered_next_state, buffer_s, buffer_a, buffer_r, GAMMA, buffer_hx, buffer_cx, self.hx, self.cx) # training stopped


                    buffer_s, buffer_a, buffer_r = [], [], [] # should keep hidden and cell state too?

                    #buffer_hx, buffer_cx = [], []

                    if done:  # done and print information
                        self.env.close()

                        with self.g_res_lock:

                            if self.value_buffer is not None and self.policy_buffer is not None: # only for worker id 0, this is satisfied

                                save_game_value_policy(len(self.g_res_length_list), self.value_buffer, self.policy_buffer)
                                self.value_buffer = []
                                self.policy_buffer = []

                            # print(f"worker {self.name} adding a reward before {len(self.g_res_list)}") # TODO check shows processes obey locks so far.
                            self.g_res_list.append(ep_r)
                            # print(f"worker {self.name} adding a reward after  {len(g_res_list)} \n")
                        with self.g_res_length_lock:
                            self.g_res_length_list.append(game_step)

                        record(self)
                        game_step = 0
                        break
                s = s_
                total_step += 1

    def init_rewards(self):
        self.bombedSomething = False
        self.bombedSomeone = False
        self.kickedSomething = False
        self.maxAmmo = 1
        self.placedABomb = set()
        self.last_obs = None
        self.last_action = None
        self.newBomb = False
        self.canKick = False

    def filter_legal_actions(self, observ):
        my_position = tuple(observ['position'])
        board = np.array(observ['board'])
        bomb_life = observ['bomb_life']
        blast_st = observ['bomb_blast_strength']
        enemies = [constants.Item(e) for e in observ['enemies']]
        ammo = int(observ['ammo'])

        directions = [constants.Action.Left, constants.Action.Up, constants.Action.Right, constants.Action.Down]
        ret = [constants.Action.Stop.value]
        if ammo > 0:
            ret.append(constants.Action.Bomb.value)
        unsafe_positions = self.surely_unsafe_positions(observ)
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if not utility.position_on_board(board, position):
                continue
            if position in unsafe_positions:
                continue
            if self.position_is_passable(board, position, enemies):
                if not self.position_has_no_escape(observ, position):
                    ret.append(direction.value)
            elif utility.position_in_items(board, position, [constants.Item.Bomb]) and observ['can_kick']:
                life = int(bomb_life[position])
                pos = utility.get_next_position(position, direction)
                test = []
                for i in range(life):
                    if utility.position_on_board(board, pos) and self.position_is_passable(board, pos, enemies):
                        test.append(True)
                    else:
                        test.append(False)
                    pos = utility.get_next_position(position, direction)
                # can kick and kick direction is valid
                if all(test):
                    ret.append(direction.value)
        if my_position in self.surely_unsafe_positions(observ) and len(ret) > 1:
            ret.remove(constants.Action.Stop.value)  # if Stop is unsafe, dont stop
        return ret

    def position_is_passable(self, board, pos, enemies):
        # hard code the smallest agent id on board
        if board[pos] >= 10:
            return False
        return utility.position_is_passable(board, pos, enemies)

    def surely_unsafe_positions(self, observ):
        my_position = observ['position']
        board = observ['board']
        bomb_life = observ['bomb_life']
        blast_st = observ['bomb_blast_strength']
        enemies = [constants.Item(e) for e in observ['enemies']]
        all_other_agents = [e.value for e in observ['enemies']] + [observ['teammate'].value]

        going_to_explode_bomb_positions = list(zip(*np.where(bomb_life == 1)))
        directions = [constants.Action.Left, constants.Action.Up, constants.Action.Right, constants.Action.Down]
        may_be_kicked = []
        for pos in going_to_explode_bomb_positions:
            for direction in directions:
                pos2 = utility.get_next_position(pos, direction)
                if not utility.position_on_board(board, pos2):
                    continue
                if board[pos2] in all_other_agents:
                    # mark to kicked
                    # may_be_kicked.append(pos)
                    break
        surely_danger_bomb_positions = [pos for pos in going_to_explode_bomb_positions if pos not in may_be_kicked]
        danger_positions = set()
        covered_bomb_positions = set()
        for pos in surely_danger_bomb_positions:
            self.add_to_danger_positions(pos, danger_positions, observ, covered_bomb_positions)

        all_covered = set()
        while len(covered_bomb_positions) > 0:
            for pos in list(covered_bomb_positions):
                self.add_to_danger_positions(pos, danger_positions, observ, covered_bomb_positions)
                all_covered.add(pos)

            for pos in list(covered_bomb_positions):
                if pos in all_covered:
                    covered_bomb_positions.remove(pos)

        # print('agent pos:', my_position, 'danger:', danger_positions)
        return danger_positions

    def add_to_danger_positions(self, pos, danger_positions, observ, covered_bomb_positions):
        '''due to bombing chain, bombs with life>=2 would still blow up if they are in the danger positions '''
        blast_st = observ['bomb_blast_strength']
        bomb_life = observ['bomb_life']
        sz = int(blast_st[pos])
        x, y = pos
        danger_positions.add(pos)
        for i in range(1, sz):
            pos2 = (x + i, y)
            if utility.position_on_board(observ['board'], pos2):
                danger_positions.add(pos2)
                if bomb_life[pos2] > 1:
                    covered_bomb_positions.add(pos2)
            pos2 = (x - i, y)
            if utility.position_on_board(observ['board'], pos2):
                danger_positions.add(pos2)
                if bomb_life[pos2] > 1:
                    covered_bomb_positions.add(pos2)
            pos2 = (x, y + i)
            if utility.position_on_board(observ['board'], pos2):
                danger_positions.add(pos2)
                if bomb_life[pos2] > 1:
                    covered_bomb_positions.add(pos2)
            pos2 = (x, y - i)
            if utility.position_on_board(observ['board'], pos2):
                danger_positions.add(pos2)
                if bomb_life[pos2] > 1:
                    covered_bomb_positions.add(pos2)

    def position_has_no_escape(self, observ, position):
        enemies = [constants.Item(e) for e in observ['enemies']]
        board = observ['board']
        blast_st = observ['bomb_blast_strength']
        bomb_life = observ['bomb_life']
        agent_position = observ['position']
        directions = [constants.Action.Left, constants.Action.Up, constants.Action.Right, constants.Action.Down]
        for direction in directions:
            position2 = utility.get_next_position(position, direction)
            if not utility.position_on_board(board, position2):
                continue
            if position2 == agent_position and bomb_life[position2] < 0.1:
                # can go back
                return False
            if position2 == agent_position and bomb_life[position2] > 0.1 and observ['can_kick']:
                pos3 = utility.get_next_position(agent_position, direction)
                if utility.position_on_board(board, pos3) and self.position_is_passable(board, pos3, enemies):
                    # cankick doest matter even there is a bomb
                    return False
            if position2 != agent_position and self.position_is_passable(board, position2, enemies):
                # passage is always ok
                return False
        return True

    def check_for_rewards(self, past, action, obs):
        board_past = np.array(past['board'])
        board = np.array(obs['board'])
        enemies = [constants.Item(e) for e in obs['enemies']]
        enemies_now = set(filter._enemies_positions(board, enemies))
        enemies_past = set(filter._enemies_positions(board_past, enemies))
        ammo = int(past['ammo'])

        if ammo > 0 and action == constants.Action.Bomb.value:
            myposition = tuple(obs['position'])
            self.placedABomb.add(myposition)
            self.newBomb = True

        #      print(board_past)
        #      print(board)
        #      print(enemies_now,enemies_past)

        remove = []
        wood = filter._wood_positions(board_past)
        for position in self.placedABomb:
            if board[position] == constants.Item.Flames.value:  # bomb has exploited
                remove.append(position)
                flames = filter._flames_positions(board)
                for w in wood:
                    if w in flames:
                        self.bombedSomething = True
                        continue
                # Decrease the number of enemies, maybe we bombed someone
                if len(enemies_now) < len(enemies_past):
                    enemies_die = enemies_past - enemies_now
                    extra = set()
                    for position in flames:
                        for direction in movements:
                            if filter.is_valid_position(board, position, direction, 1):
                                extra.add(filter.get_next_position_steps(position, direction, 1))
                    for p in extra:
                        flames.append(p)
                    for e in enemies_die:
                        if e in flames:
                            self.bombedSomeone = True
                            continue

        for posittion_to_remove in remove:
            self.placedABomb.remove(posittion_to_remove)

    def check_kick(self, state, action, nextState):

        def convert_bombs(bomb_map):
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({'position': (r, c), 'blast_strength': int(bomb_map[(r, c)])})
            return ret

        self.kickedSomething = False
        self.canKick = bool(state['can_kick'])

        if self.canKick:
            if action == constants.Action.Bomb.value or action == constants.Action.Stop.value:
                return

            bombs = convert_bombs(np.array(state['bomb_blast_strength']))
            bombsnext = convert_bombs(np.array(nextState['bomb_blast_strength']))

            if len(bombs) == 0:
                return

            prev_position = tuple(state['position'])
            #      print('fail?',prev_position,action, constants.Action(action))
            next = utility.get_next_position(prev_position, constants.Action(action))
            for bomb in bombs:
                if bomb['position'] == next:
                    #            print(bombs)
                    #            print(action, prev_position,next_position)
                    #            print(next)
                    for b in bombsnext:
                        if b['position'] == next:
                            return
                    ###bomb moves 1 step and agent stays in previous position
                    self.kickedSomething = True
                    return


    def get_reward(self, state, action, nextState):
        self.check_for_rewards(state, action, nextState)
        self.check_kick(state, action, nextState)
        r_dense = 0

        if state is not None and nextState is not None:
            can_kick = int(state['can_kick'])
            can_kickNEXT = int(nextState['can_kick'])

            blast_strength = int(state['blast_strength'])
            blast_strengthNEXT = int(nextState['blast_strength'])

            ammoNEXT = int(nextState['ammo'])

            if self.newBomb:  # Get a reward for placing a Bomb
                self.newBomb = False
                r_dense += R_PLACED_BOMB
            #       print('placed a bomb')

            if self.bombedSomeone:
                self.bombedSomeone = False  # Get a reward for killing another Bomber
                r_dense += R_KILL_BOMBER
            #       print('bombed someone')

            if self.bombedSomething:  # get a reward for blasting wood
                self.bombedSomething = False
                r_dense += R_BLAST_WOOD
            #      print('bombed something')

            if self.kickedSomething:
                self.kickedSomething = False
                r_dense += R_KICK_BOMB
            #       print('kicked something')

            if can_kickNEXT - can_kick > 0:  # Reward for getting a kicking powerup
                r_dense += R_POWER_KICK

            if blast_strengthNEXT - blast_strength > 0:  # Reward for getting a blasting powerup
                r_dense += R_POWER_BLAST

            if ammoNEXT - self.maxAmmo > 0:  # Reward for increasing ammo powerup
                self.maxAmmo = ammoNEXT
                r_dense += R_POWER_BOMB

        return r_dense + R_BEING_ALIVE


if __name__ == "__main__":

    print(sys.getrecursionlimit())
    sys.setrecursionlimit(10000)
    print(sys.getrecursionlimit())
    gnet = Net()  # global network

    #multiprocessing.set_start_method('spawn')

    manager = multiprocessing.Manager()  # to enable memory sharing among the processes
    g_res_list = manager.list()  # record episode reward to plot
    g_res_lock = multiprocessing.Lock()
    g_loss_list = manager.list()
    g_loss_lock = multiprocessing.Lock()
    g_res_length_list = manager.list()
    g_res_length_lock = multiprocessing.Lock()

    print(f"list contains {g_res_list}")

    try:
        gnet.load_state_dict(torch.load('A3C_Trained_NN.pt'))
        print("loaded the model!")
    except:
        print(" Model loading failed!")

 #   self.load_state_dict(torch.load(filename))

    print(f" now the pure learn rate is {PURE_LEARNING}")
    # print(f"imitation threshold {IMITATION_RATE}")
    print(f"learning rate is  {LR}")
    print(f"global update {UPDATE_GLOBAL_ITER}")
    print(f"gamma is  {GAMMA}")
    print(f"tile counts are randomized during training {randomize_tile_size}")
    print(f"LSTM layer active {LSTM_ENABLED}")
    print(f"curriculum learning is active{OPPONENT_CURRICULUM}")
    print(f"No Suicide rules {FILTER_RULES}")
    print(f"this is A3C standard ")

    print(gnet)
    gnet.share_memory()  # share the global parameters in multiprocessing

    opt = SharedAdam(gnet.parameters(), lr=LR, weight_decay=1e-5)  # global optimizer with shared learning rates - TODO have these vary per worker ?
    opt.share_memory()

    # opt = SharedRmsprop(gnet.parameters(), lr=LR, weight_decay=1e-5)  # global optimizer with shared learning rates - TODO have these vary per worker
    # opt = torch.optim.SGD(gnet.parameters(),lr=0.000001, momentum=0.9)

    global_ep, global_ep_r, global_ep_length = mp.Value('i', 0), mp.Value('d', -1.), mp.Value('i', 0)

    ###SOFT SWITCH
    global_imitation_p = mp.Value('d', 1.0)

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, g_res_list, g_res_lock, g_res_length_list, g_res_length_lock,
                      g_loss_list, g_loss_lock, i, global_imitation_p) for i in range(CPU_COUNT)]  #
    print(f"using {CPU_COUNT} cpus!!!")


    for w in workers:
        w.start()


    print("before join")
    for w in workers:
        w.join()  # at termination
    print("after join")

    print("threads joined end")
    gnet.save_trained_model(MAX_EP)  # We are saving the global network - which is to be used for testing
    # print("saved models")

    save_rew_len(g_res_list, g_res_length_list, g_loss_list, True, MAX_EP, None)
    print("saved logs for rewrads etc")

    print(f" rew length {len(g_res_list)} len list {len(g_res_length_list)} nad loss list {len(g_res_list)}")

    if DISPLAY_FIGURES is True:
        import matplotlib.pyplot as plt

        plt.plot(g_res_list)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Ep')
        plt.show()

        plt.plot(g_res_length_list)
        plt.ylabel('Game length')
        plt.xlabel('Ep')
        plt.show()

        plt.plot(g_loss_list)
        plt.ylabel('loss')
        plt.xlabel('time')
        plt.show()

    print(f" custom map used {CustomMap} and shuffle used {CustomMapShuffle}")
    print("done")
