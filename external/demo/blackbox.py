import random
import string
from typing import List, Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import gym

from agent import Agent

app = FastAPI()

agents_array = {}


class BoxSpace(BaseModel):
    low: float
    high: float
    shape: List[int]
    dtype: str


class Args(BaseModel):
    mode: str
    env: str
    algo: str
    exp_name: str
    load_config: str
    attack: str
    attack_algo: str
    victim: str
    load_victim: str
    load_external: str
    run: str

class ConnectRequest(BaseModel):
    obs_space: BoxSpace
    act_space: BoxSpace
    num_agents: int

class EvalRequest(BaseModel):
    token: str
    agent_id: int
    eval_obs: List[List[float]]
    eval_masks: List[List[float]]
    eval_available_actions: Optional[Any] = None


def reconstruct_box(box_space: BoxSpace) -> gym.spaces.Box:
    low = -np.inf if box_space.low == "-inf" else float(box_space.low)
    high = np.inf if box_space.high == "inf" else float(box_space.high)
    shape = tuple(box_space.shape)
    dtype = np.float64 if box_space.dtype == "float64" else np.float32
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


@app.post("/connect")
async def connect(eval_request: ConnectRequest):
    num_agents = eval_request.num_agents
    # Reconstruct gym spaces
    obs_space = reconstruct_box(eval_request.obs_space)
    act_space = reconstruct_box(eval_request.act_space)

    # 生成一个随机token
    token = ''.join(random.sample(string.ascii_letters + string.digits, 32))
    agent = Agent(obs_space, act_space)
    agents_array[token] = [agent for _ in range(num_agents)]

    return {"connected": True, "token": token}

@app.post("/perform")
async def evaluate(eval_request: EvalRequest):
    token = eval_request.token
    agent_id = eval_request.agent_id
    eval_obs = np.array(eval_request.eval_obs, dtype=np.float64)
    eval_masks = np.array(eval_request.eval_masks, dtype=np.float64)
    eval_available_actions = None
    if eval_request.eval_available_actions is not None:
        eval_available_actions = np.array(eval_request.eval_available_actions, dtype=np.float64)

    # check agents
    if token not in agents_array or agents_array[token] is None:
        return {"error": "Agent not found, please connect first."}

    # Return a respons
    agents = agents_array[token]
    eval_adv_actions, _ = agents[agent_id].perform(
        eval_obs,
        None,
        eval_masks,
        eval_available_actions,
        deterministic=True,
    )

    return {"actions": eval_adv_actions.tolist()}


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
