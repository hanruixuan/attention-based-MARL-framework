import numpy as np
import os
import datetime
from common.rollout import RolloutWorker, RolloutWorker_hybrid, RolloutWorker_rainbow
from agent.agent_UAV import Agents
from common.replay_buffer import ReplayBuffer, ReplayBuffer_hybrid

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.save_path = self.args.result_dir + '/' + args.alg + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps, train_idx = 0, 0, -1, 0
        while time_steps < self.args.n_steps:
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                # print(datetime.datetime.now())
                win_rate, episode_reward, collision_tag = self.evaluate(evaluate_steps, num)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                evaluate_steps += 1
                print('evaluate_steps {}:  {}  is_conflict: {}  is_collision: {}'.format(evaluate_steps, episode_reward,
                                                                                         win_rate, collision_tag))
            episodes = []
            episode, _, _, steps, _ = self.rolloutWorker.generate_episode(train_idx)
            episodes.append(episode)
            time_steps += steps
            train_idx += 1
            episode_batch = episodes[0]
            episodes.pop(0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        win_rate, episode_reward = self.evaluate(evaluate_steps, num)
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        reward = np.array(self.episode_rewards)
        max_r = np.max(reward)
        max_r_idx = np.argmax(reward)
        print('--------max reward----------')
        print(max_r)
        print(max_r_idx)

    def evaluate(self, evaluate_steps, slice):
        episode_rewards = 0
        win_tag = False
        collision_tag = False
        for epi_num in range(self.args.num_scene):
            _, episode_reward, win_tag_temp, _, collision_tag_temp = self.rolloutWorker.generate_episode(slice=slice,
                                                                                                         episode_num=epi_num,
                                                                                                         evaluate=True,
                                                                                                         evaluate_steps=evaluate_steps)
            episode_rewards += episode_reward
            win_tag = win_tag or win_tag_temp
            collision_tag = collision_tag or collision_tag_temp
        return win_tag, episode_rewards/self.args.num_scene, collision_tag
