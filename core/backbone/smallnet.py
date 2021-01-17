# (c) Copyright 2021 SmallBox. and others.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Small Box Backbone network. """
import tensorflow as tf

class SmallBox(tf.keras.Model):
    def __init__(self, config):
        super(SmallBox, self).__init__()
        self.config = config

        # define the backbone.
        self.shallow_conv = tf.keras.layers.Conv2D(
            32, (2, 2), strides=1, kernel_initializer="glorot_uniform", padding="same"
        )
        self.shallow_conv1 = tf.keras.layers.Conv2D(
            64, (2, 2), strides=1, kernel_initializer="glorot_uniform", padding="same"
        )
        self.shallow_conv2 = tf.keras.layers.Conv2D(
            256, (2, 2), strides=1, kernel_initializer="glorot_uniform", padding="same"
        )

    def call(self, x):
        x = self.shallow_conv(x)
        x = self.shallow_conv1(x)
        x = self.shallow_conv2(x)
        return x
