# Have a template based UCT-like planner for higher quality action
# input observation as in pommerman-FFA or Team version for Partial-Observability
# output is an action
# Bilal Kartal - September 2018 - BorealisAI

from operator import attrgetter
from collections import defaultdict
from math import sqrt, log
from random import choices, uniform, randint
import numpy as np
import queue
from copy import deepcopy

from pommerman import agents
from pommerman import characters
from pommerman import constants
from pommerman import forward_model
from pommerman import utility

ACTION_SPACE_SIZE = 6 # Pommerman Specific
C = 1.414213562
UCT_BUDGET = 50
UCT_PARTIAL_EXPAND = True
UCT_PARTIAL_EXPAND_THR = 0.1
UCT_ROLLOUT_LENGTH = 13 # to provide time for bomb-explosion
DEBUG_MODE = False

all_actions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down, constants.Action.Bomb]
directions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
_directionsAsVector = {constants.Action.Up: (0, 1),
                       constants.Action.Down: (0, -1),
                       constants.Action.Left: (1, 0),
                       constants.Action.Right: (-1, 0),
                       constants.Action.Stop: (0, 0)}
_directionsAsList = _directionsAsVector.items()


def ucb_value(node):
    return node.win_rate / node.visit_count + C * sqrt(log(node.parent_node.visit_count) / node.visit_count)

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

def _flame(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Flames.value)
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

def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
    assert (depth is not None)

    if exclude is None:
        exclude = [
            constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
        ]

    def out_of_range(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return depth is not None and abs(y2 - y1) + abs(x2 - x1) > depth

    items = defaultdict(list)
    dist = {}
    prev = {}
    Q = queue.PriorityQueue()

    mx, my = my_position
    for r in range(max(0, mx - depth), min(len(board), mx + depth)):
        for c in range(max(0, my - depth), min(len(board), my + depth)):
            position = (r, c)
            if any([
                    out_of_range(my_position, position),
                    utility.position_in_items(board, position, exclude),
            ]):
                continue



            if position == my_position:
                dist[position] = 0
            else:
                dist[position] = np.inf

            prev[position] = None
            Q.put((dist[position], position))

    for bomb in bombs:
        if bomb['position'] == my_position:
            items[constants.Item.Bomb].append(my_position)

    while not Q.empty():
        _, position = Q.get()

        if utility.position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position

        item = constants.Item(board[position])
        items[item].append(position)

    return items, dist, prev

class GameTracker(object):
    # Given  observation, initialize agents from the game start
    # Given old map, and new agent positions, update their skills, e.g. can kick or maximum number of ammo
    # This keeps track of our agent as well for convenience  - can be removed for speed-up
    # It keeps track of flames as well
    @staticmethod
    def _update_skill_and_pos(prev_board, bomber_object, bomber_new_pos):
        bomber_object.position = bomber_new_pos
        prev_board_item_value = prev_board[bomber_new_pos[0]][bomber_new_pos[1]]
        if prev_board_item_value == constants.Item.ExtraBomb.value:
            bomber_object.ammo = min(bomber_object.ammo + 1, 10)
        elif prev_board_item_value == constants.Item.IncrRange.value:
            bomber_object.blast_strength = min(bomber_object.blast_strength + 1, 10)
        elif prev_board_item_value == constants.Item.Kick.value:
            bomber_object.can_kick = True

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        return abs(pos_a[0]-pos_b[0])+abs(pos_a[1]-pos_b[1])


    def _extract_info(self, observation): # check complete
        self.current_board = np.array(observation['board'])
        #print(f" board is at {self.current_board}")
        self.sim_my_position = tuple(observation['position'])
        self.sim_agent_locations = _agents_positions(self.current_board)
        #print(f"\n agent locations from tracker extract info at init {self.sim_agent_locations}")


    def _init_flame_map(self):
        self.prev_flames = np.zeros(self.current_board.shape) # assume that game starts with no flames
        self.current_flames = np.zeros(self.current_board.shape) # flames from observation
        self.global_flame_map = np.zeros(self.current_board.shape) # actual map with lifetimes to query, values from 0 to 2

    def _update_flame_map(self, observation):

        # tick the time for the existing flames - set negative ones to zero if no actual flame
        self.global_flame_map = self.global_flame_map - 1
        self.global_flame_map[self.global_flame_map < 0] = 0

        # get the current flames from the observation - add them to global flame with a lifetime of 2
        self.current_flames = _flame(self.current_board)

        # get the new flames and set their lifetimes to 2 steps
        new_flames_from_obs = 2 * ( self.current_flames - self.prev_flames )
        new_flames_from_obs[new_flames_from_obs < 0] = 0 # to prevent negative values on the 1st step after flame gone

        self.global_flame_map += new_flames_from_obs

        #print('previous flames \n', self.prev_flames, '\n')
        #print('current flames \n', self.current_flames, '\n')
        #print('global flames \n', self.global_flame_map, '\n')

        self.prev_flames = deepcopy(self.current_flames)

    def __init__(self, observation):

        self._extract_info(observation)
        #print(f" observation is {observation}")
        self._init_flame_map()
        self.alive_agents = (observation['alive'])
        #print(f" alive ones are {self.alive_agents}")

        self.id = self.current_board[self.sim_my_position[0]][self.sim_my_position[1]]
        self.our_agent_index = self.id - 10

        self.bombermen_list = []
        for i in range(4): # assume initially 4 agents exist - but set them as dead according to game play
            self.bombermen_list.append(characters.Bomber(i+10, constants.GameType.FFA))
            self.bombermen_list[i].is_alive = False

        for i in range(4):  # assume initially 4 agents exist - but set them as dead according to game play
            if i+10 in self.alive_agents:
                self.bombermen_list[i].is_alive = True



        self.sim_enemies = deepcopy(self.alive_agents)
        self.sim_enemies.remove(self.id) # just keep enemies

        #print(f"alive including our agent {self.alive_agents}")
        #print(f"enemies only  {self.sim_enemies}")
        #print(f"agent locs {self.sim_agent_locations}")

        #print(f"agent locs {self.sim_agent_locations}")

        #print(f"bombers {self.bombermen_list}")

        self.enemy_indices = deepcopy(self.sim_enemies)
        self.enemy_indices = list(map(lambda x: x - 10, self.enemy_indices))


        #print(f" our agent index {self.our_agent_index}")
        #print(f" enemy indices are {self.enemy_indices}")
        #print(f"my id is {self.id}")

        for i in range(4): # 10 and 13
            if i+10 in self.alive_agents:
                #print(self.sim_agent_locations[i])
                self.bombermen_list[i].set_start_position(self.sim_agent_locations[i])
                self.bombermen_list[i].reset()

        self.distances_to_our_agent = [0] * 4 # TODO fix this as well

        for i in range(4):
            if i+10 in self.alive_agents:
                self.distances_to_our_agent[i] = self._manhattan_distance(self.bombermen_list[self.our_agent_index].position, self.bombermen_list[i].position)
                #print(f" distance computed is {self.distances_to_our_agent[i]}")


        self.prev_board = deepcopy(self.current_board) # save this as prev_board for next time step

    def run(self, current_observation):

        #print(f" \n our agent index is {self.our_agent_index} and enemy indices are {self.enemy_indices}\n")

        self._update_flame_map(current_observation)

        self._extract_info(current_observation)
        self._update_skill_and_pos(self.prev_board, self.bombermen_list[self.our_agent_index], self.sim_my_position) # update our agents skills

        self.enemy_indices = []
        for i in range(len(self.sim_agent_locations)):
            if list(self.sim_agent_locations.keys())[i] != self.our_agent_index:
                self.enemy_indices.append(list(self.sim_agent_locations.keys())[i])


        for i in range(4):
            if i != self.our_agent_index:
                self.bombermen_list[i].is_alive = False # set them all as dead

        for i in range(len(self.enemy_indices)):
            self.distances_to_our_agent[self.enemy_indices[i]] = self._manhattan_distance(self.bombermen_list[self.our_agent_index].position,self.bombermen_list[self.enemy_indices[i]].position)
            self.bombermen_list[self.enemy_indices[i]].is_alive = True # resurrect only the actual alive ones
            self._update_skill_and_pos(self.prev_board,self.bombermen_list[self.enemy_indices[i]],self.sim_agent_locations[self.enemy_indices[i]])

            #print(f"values are {self.enemy_indices[i]} and {self.bombermen_list[self.enemy_indices[i]].is_alive} and "
            #      f"locatiuons are {self.sim_agent_locations} ")

        self.prev_board = deepcopy(self.current_board)

    def query_agent(self, agentId): # returns a dictionary of agent properties
        index_enemy = agentId - 10 # from constants agent ids are 10, 11, 12, 13 ...
        ret = dict()
        ret["max_ammo"] = self.bombermen_list[index_enemy].ammo
        ret["blast_strength"] = self.bombermen_list[index_enemy].blast_strength
        ret["can_kick"] = self.bombermen_list[index_enemy].can_kick
        ret["is_alive"] = self.bombermen_list[index_enemy].is_alive
        ret["position"] = self.bombermen_list[index_enemy].position
        ret["distance_to_us"] = self.distances_to_our_agent[index_enemy]
        return ret

    def query_flames(self, position):
        return self.global_flame_map[position]


    def print_agents(self):
        for i in range(4):
            if self.bombermen_list[i].is_alive:
                print(f"id is {self.bombermen_list[i].agent_id} - "
                      f"is alive {self.bombermen_list[i].is_alive} - "
                      f"pos is {self.bombermen_list[i].position} - "
                      f"ammo is {self.bombermen_list[i].ammo} - "
                      f"blast is {self.bombermen_list[i].blast_strength} - "
                      f"can kick {self.bombermen_list[i].can_kick}  - "
                      f"distance is {self.distances_to_our_agent[i]} ")

class TreeNode(object):
    __tree_size = 0  # keeps the overalL tree size for debugging

    def __init__(self, edge_action=None, parent_node=None):
        self.__tree_size += 1 # debugging purpose
        self.win_rate = 0.0
        self.visit_count = 1.0
        self.depth = (parent_node.depth+1) if parent_node is not None else 0
        self.edge_action = edge_action  # set this to the edge action (function pointer respective c++) that generates self node
        self.parent_node = parent_node
        self.current_max_child_size = ACTION_SPACE_SIZE
        self.available_actions = [] # This might be set beforehand - and overriden
        self.children_nodes = []  # this should be filled on demand for memory efficiency

    def add_child_node(self,new_edge_action):
        new_born = TreeNode(new_edge_action,self) # pass self as parent of new node
        self.children_nodes.append(new_born) # actually adds the new child to the children list
        return new_born

    def is_full_explored(self): # Currenly fixed to full action_space TODO this must be called after populating the number of moves
        return len(self.children_nodes) == len(self.available_actions) # max_children_size is dynamic

    def best_ucb_child(self):
        return max(self.children_nodes, key=ucb_value)

    def best_visit_child(self):
        #for i in range(len(self.children_nodes)):
            #print(f"action stats are {i+1} children {self.children_nodes[i].visit_count} and rewards are {self.children_nodes[i].win_rate/self.children_nodes[i].visit_count}")

        return max(self.children_nodes, key=attrgetter('visit_count'))

    def _tree_printer(self):
        print('depth at ', self.depth, 'action ', self.edge_action)
        if len(self.children_nodes) > 0:
            for i in range(len(self.children_nodes)):
                print("it has ", len(self.children_nodes), 'kids')
                if self.depth < 3:
                    self.treePrinter(self.children_nodes[i])

class StateHelper(object):

    def precompute_possible_actions(self, board):
        listoflistoflists = []
        for i in range(0, board.shape[0]):
            sublist = []
            for j in range(0, board.shape[0]):
                action_list = [constants.Action.Stop.value]  # stay action
                if i - 1 >= 0 and board[i - 1][j] != constants.Item.Rigid.value:  # north
                    action_list.append(constants.Action.Up.value)
                if i + 1 < board.shape[0] and board[i + 1][
                    j] != constants.Item.Rigid.value:  # south
                    action_list.append(constants.Action.Down.value)
                if j - 1 >= 0 and board[i][j - 1] != constants.Item.Rigid.value:  # west
                    action_list.append(constants.Action.Left.value)
                if j + 1 < board.shape[0] and board[i][
                    j + 1] != constants.Item.Rigid.value:  # east
                    action_list.append(constants.Action.Right.value)

                sublist.append(action_list)
            listoflistoflists.append(sublist)

        return listoflistoflists

    def __init__(self, observation, game_tracker):

        self.sim_joint_obs = {}
        self.state_game_tracker = game_tracker # pointer to the game tracker to utilize
        self.sim_my_position = tuple(observation['position'])
        self.sim_board = np.array(observation['board'])
        self.lookup_possible_acts = self.precompute_possible_actions(self.sim_board)
        self.sim_enemy_locations = _enemies_positions(self.sim_board, tuple(observation['enemies']))
        self._reserve_agents = [agents.RandomAgent(), agents.RandomAgent(), agents.RandomAgent(), agents.RandomAgent()]

        for i in range(4):
            self._reserve_agents[i].init_agent(i, constants.GameType.FFA)

        # TODO populate only alive enemies

        for i in range(len(self.sim_enemy_locations)):
            if DEBUG_MODE:
                print(i,'th enemy at', self.sim_enemy_locations[i])

        self.sim_bombs_dict = convert_bombs(np.array(observation['bomb_blast_strength']))
        self.sim_enemies = [constants.Item(e) for e in observation['enemies']]

        if DEBUG_MODE:
            print('enemies are', self.sim_enemies)

        game_tracker_flames = self.state_game_tracker.global_flame_map
        self.sim_flames_ind = np.transpose(np.nonzero(game_tracker_flames)) # get indices of flames

        #if uct_parameters.DEBUG_MODE:
        #    print('flames are',self.sim_flames_dict)

        self.sim_ammo = int(observation['ammo'])

        self.sim_blast_strength = int(observation['blast_strength'])
        self.sim_actions_for_four = [None] * 4 # TODO  set it to the number of remaining agents -  must be overridden

        self.sim_agent_list = [] # starts with uct dummy agent - first agent is our agent indeed
        self.sim_agent_list.append(self._reserve_agents[0])

        for i in range(len(self.sim_enemy_locations)): # go over all enemies EXCLUDING recently dead ones
            self.sim_agent_list.append(self._reserve_agents[i+1])

        self.sim_bombs = []
        for i in range(len(self.sim_bombs_dict)): # TODO associate the bomb with the bomber efficiently
            self.sim_bombs.append(characters.Bomb(self.sim_agent_list[randint(0,len(self.sim_agent_list)-1)], self.sim_bombs_dict[i]['position'],
                                                  observation['bomb_life'][self.sim_bombs_dict[i]['position'][0]][self.sim_bombs_dict[i]['position'][1]],
                                                  self.sim_bombs_dict[i]['blast_strength'], moving_direction=None))

        self.sim_flames = []
        for i in range(np.count_nonzero(game_tracker_flames)):
            self.sim_flames.append(characters.Flame(tuple(self.sim_flames_ind[i]), life=game_tracker_flames[self.sim_flames_ind[i][0]][self.sim_flames_ind[i][1]]))

        self.sim_items, self.sim_dist, self.sim_prev = _djikstra(self.sim_board, self.sim_my_position, self.sim_bombs_dict, self.sim_enemies, depth=8)

    def reset_obs(self,observation):

        self.sim_my_position = tuple(observation['position'])
        self.sim_board = np.array(observation['board'])
        self.sim_bombs_dict = convert_bombs(np.array(observation['bomb_blast_strength']))
        self.sim_enemies = [constants.Item(e) for e in observation['enemies']]
        self.sim_enemy_locations = _enemies_positions(self.sim_board, tuple(observation['enemies']))

        #self.sim_flames_dict = uct_helper_utils.convert_flames(uct_helper_utils._flame(self.sim_board))
        game_tracker_flames = self.state_game_tracker.global_flame_map
        self.sim_flames_ind = np.transpose(np.nonzero(game_tracker_flames)) # get indices of flames

        self.sim_ammo = int(observation['ammo'])
        self.sim_blast_strength = int(observation['blast_strength'])
        self.sim_items, self.sim_dist, self.sim_prev = _djikstra(self.sim_board, self.sim_my_position, self.sim_bombs_dict, self.sim_enemies, depth=8)

        # TODO opponent modeling must fill the information correctly here

        # TODO Tricky - how to track bomb bomber relation to reset these values correctly?
        # Agent Modeling has to update this part
        # TODO : Associate bombs with enemies- correlate bomb lifes with bomb & enemy locations

        self._reserve_agents[0].set_start_position(self.sim_my_position)
        self._reserve_agents[0].reset(self.sim_ammo, True, self.sim_blast_strength, observation['can_kick'])
        self.sim_actions_for_four = [None] * 4
        self.sim_agent_list = [self._reserve_agents[0]]  # first agent is our agent indeed

        for i in range(len(self.sim_enemy_locations)):  # go over all enemies EXCLUDING recently dead ones
            self._reserve_agents[i+1].set_start_position(self.sim_enemy_locations[i])
            self._reserve_agents[i+1].reset(1, is_alive=True, blast_strength=None, can_kick=False)
            self.sim_agent_list.append(self._reserve_agents[i+1])

        self.sim_bombs = []
        for i in range(len(self.sim_bombs_dict)): # TODO currently moving bombs do not transfer to the UCT as moving.
            self.sim_bombs.append(characters.Bomb(self.sim_agent_list[randint(0,len(self.sim_agent_list)-1)], self.sim_bombs_dict[i]['position'],
                                                   observation['bomb_life'][self.sim_bombs_dict[i]['position'][0]][
                                                   self.sim_bombs_dict[i]['position'][1]],
                                                   self.sim_bombs_dict[i]['blast_strength'], moving_direction=None))

        self.sim_flames = []
        for i in range(np.count_nonzero(game_tracker_flames)):
            self.sim_flames.append(characters.Flame(tuple(self.sim_flames_ind[i]), life=game_tracker_flames[self.sim_flames_ind[i][0]][self.sim_flames_ind[i][1]]))  # now flames have correct lifetimes!!!

class MCTS(object):

    def __init__(self, obs, game_tracker, game_type, game_env):

        self.game_state = StateHelper(obs, game_tracker)  # game state object
        self.game_state.reset_obs(obs)

        if DEBUG_MODE:
            print('world is reset')

        self.frwd_model = forward_model.ForwardModel()  # to simulate the game dynamics during rollout
        self.root_node = None
        self.game_type = game_type
        self.game_env = game_env

    def run(self, search_budget, obs):

        self.root_node = TreeNode(None, None)
        self.game_state.reset_obs(obs)
        self.root_node.available_actions = self.possible_actions()

        if DEBUG_MODE:
            print('forward model is generated')
            print('tree node is generated ', self.root_node.depth)
            print('\n\n', self.game_state.sim_agent_list)
            print('\n')
            print('printed bombers and agents ')
            print(self.game_state.sim_bombs, ' just printed bombs ')
            print(self.game_state.sim_joint_obs)
            print('received obs for all agents')
            print('forward model run')
            print(self.game_state.sim_actions_for_four)

        for i in range(search_budget):  # Main MCTS part

            if i % 99 == 0 and DEBUG_MODE:
                print("\n Iteration ", i, " started and root node stats ", self.root_node.visit_count, " win rate ",
                      self.root_node.win_rate / self.root_node.visit_count, " and children size of ",
                      len(self.root_node.children_nodes))
                for i in range(len(self.root_node.children_nodes)):
                    print(i, "th kid has visit count", self.root_node.children_nodes[i].visit_count, " win rate",
                          self.root_node.children_nodes[i].win_rate, " UCB score of ",
                          ucb_value(self.root_node.children_nodes[i]))

            self.game_state.reset_obs(obs)

            node_to_expand_from = self.selection(self.root_node)  # this already called the forward.step() several times

            if DEBUG_MODE:
                print('flames are', self.game_state.sim_flames)
                print('actions are ', self.game_state.sim_actions_for_four)
                print('check ')
                print(all_actions[len(node_to_expand_from.children_nodes)].value)
                print('check ')

            # game_state.sim_actions_for_four[0] = all_actions[len(node_to_expand_from.children_nodes)].value # for now assume all actions are possible always
            # game_state.sim_actions_for_four = frwd_model.act(game_state.sim_agent_list, game_state.sim_joint_obs, action_space) # TODO HACK omit this as opponents are random agents

            self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4, 5], k=3)  # TODO This line replaced the forward model act method

            self.game_state.sim_actions_for_four[0] = node_to_expand_from.available_actions[len(node_to_expand_from.children_nodes)]

            new_added_node = node_to_expand_from.add_child_node(self.game_state.sim_actions_for_four[0])  # TODO refactor sim_action[0] hardcoding

            new_added_node.available_actions = self.possible_actions()  # TODO call this next time lazy fashion

            #TODO Hack - as we do not track bomb-bomber relation - set my agent accordingly

            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames \
                = self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list,
                                  self.game_state.sim_bombs,
                                  self.game_state.sim_items, self.game_state.sim_flames)

            rollout_reward = self.rollout(new_added_node)

            if DEBUG_MODE:
                print(' rollout reward is ', rollout_reward, " \n \n")

            self.backprop(new_added_node, rollout_reward)

        #self.root_node = self.root_node.best_visit_child() # update the root to the next best child - TODO do we need to prune the remaining sub-trees?
        #self.root_node.parent_node = None
        return constants.Action(self.root_node.best_visit_child().edge_action).value  # if actions are not turned off, then simply return best_visit_child()

    def possible_actions(self):  # precompute this and lookup
        action_list = list(self.game_state.lookup_possible_acts[self.game_state.sim_my_position[0]][self.game_state.sim_my_position[1]])
        if self.game_state.sim_agent_list[0].ammo > 0:
            action_list.append(constants.Action.Bomb.value)

        return action_list

    def selection(self, node):
        # return the node with highest ucb value and do this recursively
        while node.is_full_explored() or ( UCT_PARTIAL_EXPAND and len(node.children_nodes) > 0 and uniform(0.0, 1.0) < UCT_PARTIAL_EXPAND_THR ):  # keep recursively selecting until finding a terminal node or a node that can produce more children nodes.

            # act() will return a random action for our agent, override it with tree action here before passing to step() TODO Hack Alert

            node = node.best_ucb_child()
            self.game_state.sim_joint_obs = self.frwd_model.get_observations(self.game_state.sim_board, self.game_state.sim_agent_list,
                                                                   self.game_state.sim_bombs, False, constants.BOARD_SIZE, self.game_type, self.game_env)

            #game_state.sim_actions_for_four = frwd_model.act(game_state.sim_agent_list, game_state.sim_joint_obs, action_space) # TODO this is removed for FFA assuming random agents as opponents - need this for team message
            self.game_state.sim_actions_for_four[0] = constants.Action(node.edge_action).value # override forward_model action with tree action selection

            if uniform(0.0, 1.0) < 1: # currently leave as vanilla
                self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4, 5], k=3)
            else:
                self.game_state.sim_actions_for_four[1:4] = choices([0, 1, 2, 3, 4], k=3)


            if DEBUG_MODE:
                print(self.game_state.sim_actions_for_four)


            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames = \
                self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs,
                                     self.game_state.sim_items, self.game_state.sim_flames) # tick the time and update board and agents

            # TODO also augment here the action taken for the best strategy generated
        #print('selection ended')
        return node # returned node is either a leaf or needs more children

    def rollout(self, bottom_node):
        rollout_length = min( (constants.MAX_STEPS - bottom_node.depth), UCT_ROLLOUT_LENGTH )
        actual_game_length = 0
        for i in range(0, rollout_length):

            self.game_state.sim_joint_obs = self.frwd_model.get_observations(self.game_state.sim_board, self.game_state.sim_agent_list,
                                                                             self.game_state.sim_bombs, False, constants.BOARD_SIZE, self.game_type, self.game_env)


            #self.game_state.sim_actions_for_four = self.frwd_model.act(self.game_state.sim_agent_list, self.game_state.sim_joint_obs, action_space) # TODO Hackthis is removed for FFA assuming random agents as opponents - need this for team message

            self.game_state.sim_actions_for_four[0:4] = choices([0, 1, 2, 3, 4, 5], k=4)
            # current_possible_actions = possible_actions() # returns possible actions for our agent TODO do not provide extra advantage to our agent
            # game_state.sim_actions_for_four[0] = choice(current_possible_actions) # TODO only possible action for our agent

            self.game_state.sim_board, self.game_state.sim_agent_list, self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames = \
                self.frwd_model.step(self.game_state.sim_actions_for_four, self.game_state.sim_board, self.game_state.sim_agent_list,
                                     self.game_state.sim_bombs, self.game_state.sim_items, self.game_state.sim_flames)  # tick the time and update board and agents

            done = self.frwd_model.get_done(self.game_state.sim_agent_list,  i, constants.MAX_STEPS, constants.GameType.FFA, 0)

            if done:
                actual_game_length = i
                break
        rewards_for_all = self.frwd_model.get_rewards(self.game_state.sim_agent_list, constants.GameType.FFA, constants.MAX_STEPS, constants.MAX_STEPS)  # call this way as run() finished the game
        if DEBUG_MODE:
            print('all rewards are', rewards_for_all)

        return rewards_for_all[0]

    def backprop(self, bottom_node, reward):
        while bottom_node != self.root_node:
            bottom_node.win_rate += reward
            bottom_node.visit_count += 1
            bottom_node = bottom_node.parent_node

        bottom_node.win_rate += reward  # For root node update
        bottom_node.visit_count += 1  # For root node update

class Search_Oracle(object):
    # this should be a template class - that can access to other planers as well - maybe BFS - DFS etc ...

    def __init__(self, game_type, game_env):
        #print(f"Search Agent is Initialized by A3C")

        self.turnSize = 0
        self.game_type = game_type
        self.game_env = game_env
        self.uctSearcher = None
        self.usage_counter = 0
        self.game_tracker = None
        self.buffer = [] # TODO We can keep state-action-new_state-reward tuples here ...
        # HOW TO MAKE USE OF ROLLOUT TRAJECTORIES FOR EFFICIENCY

    def keep_tracking_game(self, observation): # this function needs to be called everytime now for each game actor
        if self.turnSize == 0:
            self.game_tracker = GameTracker(observation) # init the tracker
            #self.game_tracker.print_agents()
            self.uctSearcher = MCTS(observation, self.game_tracker, self.game_type, self.game_env)  # try to keep useful portion of the tree during game play
        self.turnSize += 1
        self.game_tracker.run(observation)
        #self.game_tracker.print_agents()

    def return_action(self, observation):
        self.usage_counter += 1
        return constants.Action(self.uctSearcher.run(UCT_BUDGET, observation)).value # this is to call if we need to use MCTS action


