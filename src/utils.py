"""
The extra classes or function that will be used in the main ones
"""

import argparse
import datetime
import sys

import numpy as np
import tensorflow as tf


class ArgumentParserShowHelpOnError(argparse.ArgumentParser):
    """
    It overrides the error() method of
    the ArgumentParser class
    to print the help message when an error occurs
    """

    def error(self, message):
        """
        It's a function that takes a string as an argument and prints it to the screen

        Args:
            message: The error message to print
        """
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


class OUActionNoise:
    """
    Noise as defined in the DDPG algorithm
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):  # pylint: disable=too-many-arguments
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = None
        self.reset()

    def __call__(self):
        """
        The Ornstein-Uhlenbeck process is a mean-reverting stochastic process,
        which means that the values tend to stay around the mean value.
        Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.

        Return:
            The noise is being returned.
        """
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
        """
        Reset the x_prev variable which is intrinsic to the algorithm
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Tensorboard:  # pylint: disable=too-few-public-methods
    """
    Custom tensorboard for the training loop
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir: directory of the logging
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = log_dir + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def __call__(self, epoch, reward, actions_squared, Q_loss, A_loss):  # pylint: disable=too-many-arguments
        """
        Storing all relevant variables
        """
        with self.train_summary_writer.as_default():
            tf.summary.scalar('reward', reward.result(), step=epoch)
            tf.summary.scalar('actions squared', actions_squared.result(), step=epoch)
            tf.summary.scalar('critic loss', Q_loss.result(), step=epoch)
            tf.summary.scalar('actor loss', A_loss.result(), step=epoch)
