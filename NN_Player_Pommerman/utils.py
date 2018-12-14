"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import os
import math

# place global parameters here as well

NN_SAVE_PERIOD = 300000

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.1)


def push_and_pull(worker, opt, lnet, gnet, done, s_, bs, ba, br, gamma, buffer_hx=None, buffer_cx=None, hx=None, cx=None): # this will be called per agent
    if done:
        v_s_ = 0.0               # terminal
    else:
        v_s_ = lnet.forward(s_, hx, cx)[1].data.numpy()[0, 0]

    def _precompute_exp_rewards(rewards, n_step, target_state_index  , gamma): # Todo VECTORIZE THIS ...
        sum = 0
        for i in range(0, n_step):
            if i+target_state_index < len(rewards): # TODO check
                sum += rewards[target_state_index+i]*math.pow(gamma,i)
        return sum

    def bootstrapped_value (state, bootstrap_exp, index): # TODO make sure the terminal state case
        if buffer_hx is None:
            return math.pow(gamma,bootstrap_exp)*lnet.forward(state)[1].data.numpy()[0, 0] # TODO bootstrap from next state
        else:
            return math.pow(gamma,bootstrap_exp)*lnet.forward(state, torch.index_select(buffer_hx,0,torch.tensor([index+1])), torch.index_select(buffer_cx,0,torch.tensor([index+1])))[1].data.numpy()[0, 0] # TODO bootstrap from next state


    USE_N_STEP_RETURN = True
    n_step = 20
    buffer_v_target = []

    # TODO need to fix target for terminal state? It should be 0 - verify.

    if USE_N_STEP_RETURN: # ANOTHER CHANGE IS DONE HERE
        for index, value in enumerate(bs):
            v_s_ = _precompute_exp_rewards(br, n_step, index, gamma)
            if index + n_step < len(bs):
                v_s_ += bootstrapped_value(bs[index + n_step], n_step, index) # for those states whose final value is bootstrapped from a non-zero value non-terminal state

            buffer_v_target.append(v_s_)
            #print(buffer_v_target)
    else:
        for r in br[::-1]:    # reverse buffer r - one step return
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

    # make batch out of hx and cx for the interactions

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]), worker.imitation_continue, worker.pure_searcher, buffer_hx, buffer_cx)

    # calculate local gradients and push local parameters to global

    opt.zero_grad()
    loss.backward()

    #for param in lnet.parameters():
    #    param.grad.data.clamp_(-5,5)

    nn.utils.clip_grad_norm_(lnet.parameters(),100,2) # TODO try different numbers for clipping...

    #torch.nn.utils.clip_grad_norm_(lnet.parameters, max_norm=5, norm_type=2)

    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad

    opt.step()
    #print(f"loss is {loss.item()}")

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
    #print(f" worker updated from global network ")

    with worker.g_loss_lock:
        worker.g_loss_list.append(loss.item())




def record(worker):
    with worker.g_ep.get_lock():
        worker.g_ep.value += 1

        if worker.g_ep.value % NN_SAVE_PERIOD == 0: # save network and also write values to the files
            worker.gnet.save_trained_model(worker.g_ep.value)
            print(f"saved model at record at {worker.g_ep.value} ")
            with worker.g_loss_lock and worker.g_res_length_lock and worker.g_res_lock:
                save_rew_len(worker.g_res_list, worker.g_res_length_list, worker.g_loss_list, False, worker.g_ep.value, worker.expert_query_histogram)



def save_game_value_policy(episode_counter, value_list=None, policy_list=None): # This saves values and policies per game
    # ASSUMES VALUES ARE PASSED AS A LIST
    # POLICY_LIST IS PASSED AS A 2D NUMPY ARRAY

    here = os.path.dirname(os.path.realpath(__file__))
    folder_name = 'game_value_policy'

    if os.path.isdir(folder_name) is False:
        print("creating the folder")
        os.mkdir(folder_name)

    if value_list is not None:
        filename = "values_for_visualization_"+str(episode_counter)+"_.txt"
        filepath = os.path.join(here, folder_name, filename)
        with open(filepath, 'w') as file:
            for item in value_list:
                file.write(f"{str(item[0])}\n") # due to tensor to numpy conversion, need to index 0 to get rid of brackets ..
            file.close()

    if policy_list is not None:
        filename = "probs_for_visualization_"+str(episode_counter)+"_.txt"
        filepath = os.path.join(here, folder_name, filename)
        np.savetxt(filepath, policy_list, fmt="%.3f")

def save_rew_len(rewards, lengths, loss_list, is_final, episode_played, expert_query_histogram=None): # saves game rewards and lengths
    N = 50
    folder_name = 'results'
    if os.path.isdir(folder_name) is False:
        print("creating the folder")
        os.mkdir(folder_name)

    here = os.path.dirname(os.path.realpath(__file__))

    if is_final:
        filename = "a3c_summary_final" + str(episode_played)
    else:
        filename = "a3c_summary_intermediate" + str(episode_played)

    moving_rewards = np.convolve(rewards, np.ones((N,)) / N, mode='valid')
    moving_durations = np.convolve(lengths, np.ones((N,)) / N, mode='valid').astype(int)

    rm_filename = "_rm_" + filename + ".txt"
    filepath = os.path.join(here, folder_name, rm_filename)
    np.savetxt(filepath, moving_rewards)

    tm_filename = "_tm_" + filename + ".txt"
    filepath = os.path.join(here, folder_name, tm_filename)
    np.savetxt(filepath, moving_durations)

    if expert_query_histogram is not None:
        hist_filename = "_hist_"+filename + ".txt"
        filepath = os.path.join(here, folder_name, hist_filename)
        np.savetxt(filepath, expert_query_histogram)

    r_filename = "_r_" + filename + ".txt"
    filepath = os.path.join(here, folder_name, r_filename)
    with open(filepath, 'w') as file:
        for item in rewards:
            file.write(f"{str(item)}\n")
        file.close()

    t_filename = "_t_" + filename + ".txt"
    filepath = os.path.join(here, folder_name, t_filename)
    with open(filepath, 'w') as file:
        for item in lengths:
            file.write(f"{str(item)}\n")
        file.close()

    loss_filename = "_loss_" + filename + ".txt"
    filepath = os.path.join(here, folder_name, loss_filename)
    with open(filepath, 'w') as file:
        for item in loss_list:
            file.write(f"{str(item)}\n")
        file.close()
