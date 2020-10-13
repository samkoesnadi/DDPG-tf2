# Deep Deterministic Policy Gradient (DDPG) in Tensorflow 2

![python 3](https://img.shields.io/badge/python-3-blue.svg)
![tensorflow 2](https://img.shields.io/badge/tensorflow-2-orange.svg)

My implementation of DDPG based on paper https://arxiv.org/abs/1509.02971. 
This implementation also contains some modification of the algorithm that
are mainly aimed to speed up the learning process.

##### Table of Contents  
- [Why?](#why)  
- [Requirements](#requirements)
- [Training](#training)
- [Sampling](#sampling)
- [Future improvements](#future-improvements)
- [CONTRIBUTING](#contributing)
- [LICENSE](#license)

## Why?
Reinforcement learning is important when it comes to real environment. As
there is no definite right way to achieve a goal, the AI can be optimized based
on reward function instead of continuously supervised by human.

In continuous action space, DDPG algorithm shines as one of the best in
the field. In contrast to discrete action space, 
continuous action space mimics the reality of the world.

The original implementation is in PyTorch. Additionally, there are several
modifications of the original algorithm that may improve it.

## Changes from original paper
As mentioned above, there are several changes with different aims:
- The loss function of Q-function uses **Mean Absolute Error** instead of Mean
Squared Error. After experimenting, this speeds up training by 
a lot of margin. One possible cause is because Mean Squared Error may
overestimate value above one and underestime value below one (x^2 function).
This might be unfavorable for the Q-function update as all value range should
be treated similarly.
- **Epsilon-greedy** is implemented in addition to the policy's action. This
increases faster exploration. Sometimes the agent can stuck with one policy's
action, this can be exited with random policy action introduced by epsilon-greedy.
As DDPG is off-policy, this surely is fine. The epsilon-greedy and noise are turned off in the testing state.
- **Unbalance replay buffer**. Recent entries in the replay buffer are more likely to be taken
than the earlier ones. This reduces repetitive current mistakes that the agent
does.

## Requirements
`pip3 install -r requirements.txt`

## Training
```python3
python3 main.py [-h] [--env [ENV]]
                 [--render_env [RENDER_ENV]]
                 [--train [TRAIN]]
                 [--use_noise [USE_NOISE]]
                 [--eps_greedy [EPS_GREEDY]]
                 [--warm_up [WARM_UP]]
                 [--save_weights [SAVE_WEIGHTS]]

optional arguments:
  -h, --help            show this help message and exit
  --env [ENV]           The OpenAI Gym environment to train on
  --render_env [RENDER_ENV]
                        Render the environment to be visually visible
  --train [TRAIN]       Train the network on the modified DDPG algorithm
  --use_noise [USE_NOISE]
                        OU Noise will be applied to the policy action
  --eps_greedy [EPS_GREEDY]
                        The epsilon for Epsilon-greedy in the policy's action
  --warm_up [WARM_UP]   Following recommendation from OpenAI Spinning Up, the
                        actions in the early epochs can be set random to
                        increase exploration. This warm up defines how many
                        epochs are initially set to do this.
  --save_weights [SAVE_WEIGHTS]
                        Save the weight of the network in the defined
                        checkpoint file directory.
```
After every epoch, the network's weights will be stored in the checkpoints directory defined in `common_definitions.py`.
There are 4 weights files that represent each networks, namely critic network,
actor network, target critic, and target actor. 
Additionally, TensorBoard is used to track the resultive losses and rewards.

The pretrained weights can be retrieved from these links:
- [BipedalWalker-v3](https://github.com/samuelmat19/DDPG-tf2/releases/download/0.0.1/Bipedal_checkpoints.zip)
- [LunarLanderContinuous-v2](https://github.com/samuelmat19/DDPG-tf2/releases/download/0.0.2/Lunar_checkpoints.zip)

## Testing (Sampling)
Testing is done with the same file as training (`main.py`), but with
specific parameters as such.

```python3
python3 main.py --render_env True --train False --use_noise False
                --eps_greedy 1.0
                --warm_up 0
                --save_weights False
```

## Future improvements
- [ ] Improve documentation
- [ ] ...

## CONTRIBUTING
To contribute to the project, these steps can be followed. Anyone that contributes will surely be recognized and mentioned here!

Contributions to the project are made using the "Fork & Pull" model. The typical steps would be:

1. create an account on [github](https://github.com)
2. fork this repository
3. make a local clone
4. make changes on the local copy
5. commit changes `git commit -m "my message"`
6. `push` to your GitHub account: `git push origin`
7. create a Pull Request (PR) from your GitHub fork
(go to your fork's webpage and click on "Pull Request."
You can then add a message to describe your proposal.)


## LICENSE
This open-source project is licensed under MIT License.
