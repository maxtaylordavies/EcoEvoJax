import os
import random
import sys

from flax.struct import dataclass
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pickle
from tqdm import tqdm

from source.gridworld import Gridworld
from source.utils import VideoWriter
from evojax.util import load_model
from evojax.task.base import TaskState

sys.path.append(os.getcwd())


# these classes are used in the lab environment
@dataclass
class AgentStates_noparam(object):
    posx: jnp.uint16
    posy: jnp.uint16
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State_noparam(TaskState):
    last_actions: jnp.int8
    rewards: jnp.int8
    agents: AgentStates_noparam
    steps: jnp.int32


def write_frame(state, writer):
    # make rgb frame
    rgb_im = jnp.clip(state.state[:, :, :3], 0, 1)

    # change color scheme to white green and black
    rgb_im = jnp.clip(rgb_im + jnp.expand_dims(state.state[:, :, 1], axis=-1), 0, 1)
    rgb_im = rgb_im.at[:, :, 1].set(0)
    rgb_im = 1 - rgb_im
    rgb_im = rgb_im - jnp.expand_dims(state.state[:, :, 0], axis=-1)

    # write frame
    rgb_im = jnp.repeat(rgb_im, 5, axis=0)
    rgb_im = jnp.repeat(rgb_im, 5, axis=1)
    writer.add(np.array(rgb_im))


def one_hot(x, dim=5, dtype=jnp.float32):
    return jnp.zeros(dim, dtype=dtype).at[x].set(1)


def greedy_policy(agent_x, agent_y, food_locs):
    # first, find the coordinates of all the food
    food_coords = jnp.argwhere(food_locs == 1)

    # next, find the distance from the agent to each food
    distances = jnp.sum(jnp.abs(food_coords - jnp.array([agent_x, agent_y])), axis=1)

    # finally, find the food that is closest to the agent
    closest_food = food_coords[jnp.argmin(distances)]

    # now, move the agent towards the closest food
    dy = closest_food[0] - agent_x
    dx = closest_food[1] - agent_y

    poss_actions = []
    if dx > 0:  # can move right
        poss_actions.append(4)
    elif dx < 0:  # can move left
        poss_actions.append(2)
    if dy > 0:  # can move down
        poss_actions.append(3)
    elif dy < 0:  # can move up
        poss_actions.append(1)

    if len(poss_actions) == 0:
        return 0

    return random.choice(poss_actions)


SEED = 0
key = jr.PRNGKey(SEED)

config = {
    "grid_width": 30,
    "grid_length": 30,
    "nb_agents": 10,
    "hard_coded": 0,
    "gen_length": 500,
    "init_food": 10,
    "place_agent": False,
    "place_resources": False,
    "regrowth_scale": 0,
    "agent_view": 7,
}

# load params for all agents in final generation
project_dir, gen = "projects/pretrained/seed3/train", 950
params, obs_param = load_model(f"{project_dir}/models", f"gen_{gen}.npz")

# sample 10 random agents
with open(f"{project_dir}/data/gen_{gen}states.pkl", "rb") as f:
    agent_info = pickle.load(f)["states"][-1].agents.time_alive
    potential_agents = [idx for idx, el in enumerate(agent_info) if el > 300]
    idx = random.choice(potential_agents)
    params_test = params[[(idx - el) % (params.shape[0]) for el in range(10)], :]

env = Gridworld(
    SX=config["grid_length"],
    SY=config["grid_width"],
    init_food=config["init_food"],
    nb_agents=config["nb_agents"],
    reproduction_on=False,
    regrowth_scale=config["regrowth_scale"],
    place_agent=config["place_agent"],
    place_resources=config["place_resources"],
    params=params_test,
    time_death=config["gen_length"] + 1,  # agents never die
    time_reproduce=800,
)


agent_locs = jnp.array([[(2 * i) + 1, (2 * i) + 1] for i in range(10)])
food_locs = jnp.array([[2 * i, 2 * i] for i in range(10)])

next_key, key = jr.split(key)
state = env.reset(next_key, agent_locs=agent_locs, resource_locs=food_locs)

with VideoWriter("./scratch.mp4", 5.0) as vw:
    write_frame(state, vw)
    # for t in tqdm(range(config["gen_length"])):
    for t in tqdm(range(10)):

        agent_x = state.agents.posx[9]
        agent_y = state.agents.posy[9]
        food_locs = state.state[:, :, 1]
        a = greedy_policy(agent_x, agent_y, food_locs)
        actions = {i: one_hot(1) for i in range(10)}
        actions[9] = one_hot(a)

        state, reward, _ = env.step(state, override_actions=actions)
        write_frame(state, vw)
        tqdm.write(f"t: {t}, a: {a} reward: {reward}")
    vw.close()
