from gym.envs.registration import register
# from tasks.obstacle import Obstacle
register(
    id="Antgene-v0",
    entry_point='envs_gene.tasks:Obstacle',
    max_episode_steps=3000,
    reward_threshold=6000.0,
)
