import numpy as np
from gym import utils
from envs_gene.tasks.gene_env import GeneEnv

class Obstacle(GeneEnv, utils.EzPickle):
    def __init__(self, task_id, args):
        GeneEnv.__init__(self, 'ant_obstacle.xml', 5, task_id, args)
        utils.EzPickle.__init__(self)
        self.args = args
        

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        forward_reward = (yposafter - yposbefore)/self.dt
        ctrl_cost = 0.2 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        terminate_reward = 100

        reward = - ctrl_cost + forward_reward
        
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= -30.0 and state[2] <= 3.0              
        done = not notdone

        if yposafter>57:
            reward += terminate_reward
            done = True

        ob = self._get_obs()
        return ob, reward, done, dict(
               reward_forward=forward_reward,
               reward_ctrl=-ctrl_cost,
               reward_contact=-contact_cost,
               reward_survive=survive_reward)

    def _get_obs(self):                  
        return np.concatenate([
            self.sim.data.qpos.flat[2:],                            
            self.sim.data.qvel.flat,
            # self.get_env_sur()
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_env_sur(self):
        pos = self.get_body_com("torso")
        hfield = self.metadata['hfield']
        sur_len = 10 

        hfield_ = np.zeros((hfield.shape[0]+2*sur_len, hfield.shape[1]+2*sur_len))
        hfield_[:, :] = -2
        hfield_[sur_len:sur_len+hfield.shape[0], sur_len:sur_len+hfield.shape[1]] = hfield

        pos0 = int(5 * pos[0] + hfield_.shape[1] / 2)
        pos1 = int(5 * pos[1] + hfield_.shape[0] / 2)

        sur = hfield_[pos1-sur_len:pos1+sur_len+1, pos0-sur_len:pos0+sur_len+1]
        return sur.flatten()



