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
import tensorflow as tf
from backbone.backbone_factory import builder

class Neck(tf.keras.Model):
    def __init__(self):
        super().__init__()
        for i in range(1, 3):
            self.neck.append(tf.keras.layers.Conv2D(32*i, 2, kernel_initializer='glorot_intilizer'))
        self.neck_outs = []

    def call(self, backbone_outs):
        for i, bo in enumerate(backbone_outs):
            if i != 0:
                x = tf.keras.layers.Concatenate(axis=-1)([bo, x])
                x = self.neck[i](x)
                self.neck_outs(x)
            else:
                x = self.neck[i](bo)

        return neck_outs

class SmallBoxObjectDetector:
    def __init__(self):
        self.backbone = builder()

    def build(self):
        #TODO write me
        pass 


