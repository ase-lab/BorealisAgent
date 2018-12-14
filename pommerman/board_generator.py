
import pommerman
from pommerman import agents
import copy
import numpy as np
from pommerman import utility
import random
from pommerman import constants
import gym

def swapAgentsPositions(env,id0,id1, kick,ammo,blast):
    tmp = copy.copy(env._agents[id0].position)
    env._board[tmp]=utility.agent_value(env._agents[id1].agent_id)
    env._board[env._agents[id1].position]=utility.agent_value(env._agents[id0].agent_id)

    env._agents[id0].set_start_position(env._agents[id1].position)
    env._agents[id0].reset(ammo,True,blast,kick)

    env._agents[id1].set_start_position(tmp)
    env._agents[id1].reset()

 #   print(env.get_json_info())

    env._init_game_state=env.get_json_info()
    return env


def randomizeBoard(config,agents,size, rigid,wood,items,step_count,kick,ammo,blast):

    env = pommerman.make(config, agents)
    env.reset()
    #print(env.get_json_info())
    env._board_size=size
    env._step_count=step_count
    env._num_rigid = rigid
    env._num_wood = wood
    env._num_items = items

    env.reset()
    env._step_count = step_count

#    if ammo > 1:
#        for i in range(1, ammo):
#            env._agents[0].incr_ammo()
  #  if kick:
 #       env._agents[0].pick_up(constants.Item.Kick, None)
 #       env._agents[0].reset()
  #  if blast > 2:
 #       for i in range(2, blast):
 #           env._agents[0].pick_up(constants.Item.IncrRange, 10)
 #       env._agents[0].reset()
    env._agents[0].reset(ammo, True, blast, kick)
 #   print(env.get_json_info())
    env._init_game_state = env.get_json_info()
 #   print(env.get_json_info())

    return env

def killAgent(env, idkill0):
    env._agents[idkill0].die()
    env._board[env._agents[idkill0].position] = constants.Item.Passage.value

    env._init_game_state = env.get_json_info()
    env.reset()
  #  print(env.get_json_info())
    return env

def killTwoAgents(env, idkill0, idkill1):
    env._agents[idkill0].die()
    env._agents[idkill1].die()
    env._board[env._agents[idkill0].position] = constants.Item.Passage.value
    env._board[env._agents[idkill1].position] = constants.Item.Passage.value

    env._init_game_state = env.get_json_info()
    env.reset()
#    print(env.get_json_info())
    return env


def randomizeWithTwoAgents(config,agents,size, rigid,wood,items,step_count):
    env = randomizeBoard(config,agents,size, rigid, wood, items, step_count)
    env = swapAgentsPositions(env, 0, 3)
    die=list(range(1,4))
   # print (die)
    die.remove(random.choice(die))
   # print (die)
    env=swapAgentsPositions(env,0,random.randint(1,3))
    env = killTwoAgents(env, die[0], die[1])
    return env


def randomizeWithAgents(config,agents,size, rigid,wood,items,step_count, numberofopponents, kick=False, ammo=1, blast=2):
    env = randomizeBoard(config,agents,size, rigid, wood, items, step_count,kick,ammo,blast)
    die=list(range(1,4))
   # print (die)
    die.remove(random.choice(die))
   # print (die)
    env=swapAgentsPositions(env,0,random.randint(1,3), kick,ammo,blast)
    if numberofopponents==1:
        die = list(range(1, 4))
        die.remove(random.choice(die))
        env = killTwoAgents(env, die[0], die[1])
    elif numberofopponents==2:
        die = list(range(1, 4))
        env = killAgent(env, random.choice(die))

#    env._agents[0].reset(ammo, True, blast, kick)
#    print(env.get_json_info())

    return env



def randomizeWithTeamAgents(config,agents,size, rigid,wood,items,step_count, numberofopponents, kick=False, ammo=1, blast=2):
    env = randomizeBoard(config,agents,size, rigid, wood, items, step_count,kick,ammo,blast)

  #   if random.random()>0.5: #Move to second diagonal
  #       if random.random() > 0.5:  # Move to second diagonal
  # #           print('this one')
  #            env = swapAgentsPositions(env, 0, 1, kick, ammo, blast)
  #            env = swapAgentsPositions(env, 2, 3, kick, ammo, blast)
  #       else:
  # #           print('or this one')
  #            env = swapAgentsPositions(env, 0, 3, kick, ammo, blast)
  #            env = swapAgentsPositions(env, 2, 1, kick, ammo, blast)
  #   else:
  #      if random.random() > 0.5:
  # #           print('last one')
  #            env = swapAgentsPositions(env, 0, 2, kick, ammo, blast)

   # if numberofopponents==1:
   #     die = [1,3]
   #     env = killAgent(env, random.choice(die))

    env._agents[0].reset(ammo, True, blast, kick)
#    print(env.get_json_info())
    return env


def noshuffle(env, step_count):
    env.reset()
    env._step_count = step_count
    env._init_game_state = env.get_json_info()


def shuffle(env, config, size, rigid, wood, items, step_count, numberofopponents,kick=False,ammo=1,blast=2):
    agents=env._agents
    env = gym.make(config)
    env._board_size = size
    env._num_rigid = rigid
    env._num_wood = wood
    env._num_items = items
    env.set_agents(agents)
    env.set_init_game_state(None)
    env.reset()
    env._step_count = step_count
    env._init_game_state = env.get_json_info()

    index=random.randint(0, 3)
    if index !=0:
        env = swapAgentsPositions(env, 0, index,kick,ammo,blast)
    if numberofopponents == 1:
        die = list(range(1, 4))
        die.remove(random.choice(die))
        env = killTwoAgents(env, die[0], die[1])
    elif numberofopponents== 2:
        die = list(range(1, 4))
        env = killAgent(env, random.choice(die))

    return env


def shuffleTeam(env, config, size, rigid, wood, items, step_count, numberofopponents,kick=False,ammo=1,blast=2):
    agents=env._agents
    env = gym.make(config)
    env._board_size = size
    env._num_rigid = rigid
    env._num_wood = wood
    env._num_items = items
    env.set_agents(agents)
    env.set_init_game_state(None)
    env.reset()
    env._step_count = step_count
    env._init_game_state = env.get_json_info()

    if random.random() > 0.5:  # Move to second diagonal
#        if random.random() > 0.5:  # Move to second diagonal
            #           print('this one')
#            env = swapAgentsPositions(env, 0, 1, kick, ammo, blast)
#            env = swapAgentsPositions(env, 2, 3, kick, ammo, blast)
#        else:
            #           print('or this one')
            env = swapAgentsPositions(env, 0, 3, kick, ammo, blast)
            env = swapAgentsPositions(env, 2, 1, kick, ammo, blast)
   # else:
  #      if random.random() > 0.5:
            #           print('last one')
  #          env = swapAgentsPositions(env, 0, 2, kick, ammo, blast)

    # if numberofopponents==1:
    #     die = [1,3]
    #     env = killAgent(env, random.choice(die))

    env._agents[0].reset(ammo, True, blast, kick)

    return env


if __name__ == "__main__":
    size=10
    step_count=700
    rigid=20
    wood=10
    items=10






