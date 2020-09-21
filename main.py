import gym
from model import *
from common_definitions import *
from buffer import *
from tqdm import tqdm, trange

if __name__ == "__main__":
    env = gym.make(RL_TASK)
    # env = gym.make('Pendulum-v0')
    action_space_high = env.action_space.high

    brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], action_space_high)
    tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

    # load weights if available
    print(brain.load_weights(CHECKPOINTS_PATH))

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            prev_state = env.reset()
            acc_reward.reset_states()
            actions_squared.reset_states()
            Q_loss.reset_states()
            A_loss.reset_states()

            while True:
                if RENDER_ENV: env.render()  # render the environment into GUI

                # Recieve state and reward from environment.
                cur_act = brain.act(tf.expand_dims(prev_state,0), ep >= WARM_UP)

                # print(cur_act.shape)
                state, reward, done, info = env.step(cur_act)
                prev_state = state
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))

                brain.remember(prev_state, reward, state, int(done))

                # update weights
                if len(brain.buffer.buffer) >= BATCH_SIZE:
                    c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                    Q_loss(c)
                    A_loss(a)

                if done: break

            # print the average reward
            t.set_postfix(r=acc_reward.result().numpy())
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            # save weights
            if ep % 5 == 0:
                brain.save_weights(CHECKPOINTS_PATH)

    env.close()
    brain.save_weights(CHECKPOINTS_PATH)

    print("Training done...")