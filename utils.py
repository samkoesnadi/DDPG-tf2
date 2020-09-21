from common_definitions import *
import datetime

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Tensorboard:
    def __init__(self, log_dir):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = log_dir + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def __call__(self, epoch, reward, actions_squared, Q_loss, A_loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('reward', reward.result(), step=epoch)
            tf.summary.scalar('actions squared', actions_squared.result(), step=epoch)
            tf.summary.scalar('critic loss', Q_loss.result(), step=epoch)
            tf.summary.scalar('actor loss', A_loss.result(), step=epoch)
