import gym
from model import *
from common_definitions import *
from buffer import *
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make(RL_TASK)
    # env = gym.make('Pendulum-v0')
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]

    brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], action_space_high, action_space_low)
    tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

    # load weights if available
    print(brain.load_weights(CHECKPOINTS_PATH))

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            prev_state = env.reset()
            acc_reward.reset_states()
            actions_squared.reset_states()
            Q_loss.reset_states()
            A_loss.reset_states()
            brain.noise.reset()

            for _ in range(2000):
                if RENDER_ENV: env.render()  # render the environment into GUI

                # Recieve state and reward from environment.
                cur_act = brain.act(tf.expand_dims(prev_state,0), _notrandom=(ep >= WARM_UP) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES))
                state, reward, done, _ = env.step(cur_act)
                # print(cur_act)
                brain.remember(prev_state, reward, state, int(done))

                # update weights
                c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                Q_loss(c)
                A_loss(a)

                # post update for next step
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))
                prev_state = state

                if done: break

            ep_reward_list.append(acc_reward.result().numpy())
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # print the average reward
            t.set_postfix(r=avg_reward)
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            # save weights
            if ep % 5 == 0:
                if SAVE_WEIGHTS: brain.save_weights(CHECKPOINTS_PATH)

    env.close()
    brain.save_weights(CHECKPOINTS_PATH)

    print("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()