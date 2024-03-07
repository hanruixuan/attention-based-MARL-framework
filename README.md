# Attention-based VDN MARL Framework
This repository utilizes the Attention-based VDN MARL to perform spectrum allocation and velocity level selection (Discrete Action Space). 
- The Python environment is Python 3.6 and higher and needs PyTorch and NumPy libraries.
- This repo tests the Attention-based VDN MARL against a simulated cellular-connected UAM scenario
- The ground cellular users are randomly positioned, their locations for training and evaluation are in (`scenario_sim_training.txt`) and (`scenario_sim_evaluation.txt`)
- The aerial vehicles are assumed to moving between vertiports, and vertiports locations are shown in (`Multi_UAV_env.py`)


For the code, the `main.py` file is the entry of the whole program and it will call the (`runner.py`) file to let the DRL algorithm training and evaluation. 
In the (`runner.py`) file, it will utilize the (`/common/rollout.py`) to let the agent (`/agent/agent_UAV.py`) and the environment (`Multi_UAV_env.py`) have interactions to generate episodes.
The generated episodes will be saved in the replay memory buffer ('/common/replay_buffer.py') and be utilized to train the neural networks.
For the agent, (`/agent/agent_UAV.py`) will call (`/policy/VDN_UAV.py`) to set their policies and initialize the neural networks 
