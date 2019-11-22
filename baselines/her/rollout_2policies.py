from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

import pdb

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=2,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.success_pos_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.first_policy_done = False

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']
        #pdb.set_trace()

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes, successes_pos = [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            success_pos = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success'][1]
                        success_pos[i] = info['is_success'][0]
                    if 'done' in info:
                        self.first_policy_done = info['done']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if self.first_policy_done:
                break

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            successes_pos.append(success_pos.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            #Qs.append(np.linalg.norm(self.g.copy()-ag.copy(),axis=-1))
            o[...] = o_new
            ag[...] = ag_new

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        successful_pos = np.array(successes_pos)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        success_rate_pos = np.mean(successful_pos)
        self.success_history.append(success_rate)
        self.success_pos_history.append(success_rate_pos)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        for t in range(self.T):

            # self.g = np.array([[1, 1, 1, 1,
            #                   0.82950088,  0.19504257,  0.74951634,  0.82558665,  0.19408095,  0.72752193,
            #                   0.8294237,  0.19509856,  0.70551644,  0.83616574,  0.19685965,  0.6825068]], 'Float32')

            self.g = np.array([[1, 1, 1, 1,
                                0.81399449,
                                0.08906187,
                                0.36651383,
                                0.80723628,
                                0.08749478,
                                0.34525658,
                                0.80821288,
                                0.08766061,
                                0.32291785,
                                0.81195864,
                                0.08844444,
                                0.29918275]], 'Float32')



            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)


            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            success_pos = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success'][1]
                        success_pos[i] = info['is_success'][0]
                    if 'done' in info:
                        self.first_policy_done = info['done']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if self.first_policy_done:
                break

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            successes_pos.append(success_pos.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            #Qs.append(np.linalg.norm(self.g.copy()-ag.copy(),axis=-1))
            o[...] = o_new
            ag[...] = ag_new

        # for t in range(self.T):
        #
        #     # self.g = np.array([[1, 1, 1, 1, 0.77939238,
        #     #        0.01007279,
        #     #        0.77396591,
        #     #        0.78406789,
        #     #        0.01003857,
        #     #        0.75209954,
        #     #        0.7807472,
        #     #        0.01000307,
        #     #        0.72998683,
        #     #        0.77445873,
        #     #        0.00996551,
        #     #        0.70678223,
        #     #        0.86935003,
        #     #        0.00708711,
        #     #        0.7767419,
        #     #        0.87402555,
        #     #        0.00705288,
        #     #        0.75487552,
        #     #        0.87070486,
        #     #        0.00701738,
        #     #        0.73276282,
        #     #        0.86441638,
        #     #        0.00697982,
        #     #        0.70955821]], 'Float32')
        #
        #
        #
        #     policy_output = self.policy.get_actions(
        #         o, ag, self.g,
        #         compute_Q=self.compute_Q,
        #         noise_eps=self.noise_eps if not self.exploit else 0.,
        #         random_eps=self.random_eps if not self.exploit else 0.,
        #         use_target_net=self.use_target_net)
        #
        #
        #     if self.compute_Q:
        #         u, Q = policy_output
        #         Qs.append(Q)
        #     else:
        #         u = policy_output
        #
        #     if u.ndim == 1:
        #         # The non-batched case should still have a reasonable shape.
        #         u = u.reshape(1, -1)
        #
        #     o_new = np.empty((self.rollout_batch_size, self.dims['o']))
        #     ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
        #     success = np.zeros(self.rollout_batch_size)
        #     success_pos = np.zeros(self.rollout_batch_size)
        #     # compute new states and observations
        #     for i in range(self.rollout_batch_size):
        #         try:
        #             # We fully ignore the reward here because it will have to be re-computed
        #             # for HER.
        #             curr_o_new, _, _, info = self.envs[i].step(u[i])
        #             if 'is_success' in info:
        #                 success[i] = info['is_success'][1]
        #                 success_pos[i] = info['is_success'][0]
        #             o_new[i] = curr_o_new['observation']
        #             ag_new[i] = curr_o_new['achieved_goal']
        #             for idx, key in enumerate(self.info_keys):
        #                 info_values[idx][t, i] = info[key]
        #             if self.render:
        #                 self.envs[i].render()
        #         except MujocoException as e:
        #             return self.generate_rollouts()
        #
        #     if np.isnan(o_new).any():
        #         self.logger.warning('NaN caught during rollout generation. Trying again...')
        #         self.reset_all_rollouts()
        #         return self.generate_rollouts()
        #
        #     obs.append(o.copy())
        #     achieved_goals.append(ag.copy())
        #     successes.append(success.copy())
        #     successes_pos.append(success_pos.copy())
        #     acts.append(u.copy())
        #     goals.append(self.g.copy())
        #     #Qs.append(np.linalg.norm(self.g.copy()-ag.copy(),axis=-1))
        #     o[...] = o_new
        #     ag[...] = ag_new


        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.success_pos_history.clear()
        self.Q_history.clear()


    def current_success_rate(self):
        return [np.mean(self.success_history)/5.0+np.mean(self.success_pos_history)*1.0, np.mean(self.success_pos_history), np.mean(self.success_history)]

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('mean_force_error', np.mean(self.success_history))]
        logs += [('mean_pos_error', np.mean(self.success_pos_history))]
        logs += [('succes_rate', np.mean(self.success_history)/5.0+np.mean(self.success_pos_history)*1.0)]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
