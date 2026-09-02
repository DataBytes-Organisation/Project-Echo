# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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

"""Installation test for YAMNet."""
from pathlib import Path
YAMNET_DIR = Path(__file__).resolve().parent
import numpy as np
import tensorflow as tf

from yamnet_dir import params
from yamnet_dir import yamnet

class YAMNetTest(tf.test.TestCase):

  _yamnet_graph = None
  _yamnet = None
  _yamnet_classes = None

  @classmethod
  def setUpClass(cls):
    super(YAMNetTest, cls).setUpClass()
    cls._yamnet_graph = tf.Graph()
    with cls._yamnet_graph.as_default():
      cls._yamnet = yamnet.yamnet_frames_model(params)
      yamnet_weights = Path(__file__).resolve().parent / "yamnet.h5"
      cls._yamnet.load_weights(str(yamnet_weights))
      yamnet_class_map = Path(__file__).resolve().parent / "yamnet_class_map.csv"
      cls._yamnet_classes = yamnet.class_names(str(yamnet_class_map))
      
  def clip_test(self, waveform, expected_class_name, top_n=10):
    """Run the model on the waveform, check that expected class is in top-n."""
    with YAMNetTest._yamnet_graph.as_default():
      prediction = np.mean(YAMNetTest._yamnet.predict(
        np.reshape(waveform, [1, -1]), steps=1)[0], axis=0)
      top_n_class_names = YAMNetTest._yamnet_classes[
        np.argsort(prediction)[-top_n:]]
      self.assertIn(expected_class_name, top_n_class_names)

  def testZeros(self):
    self.clip_test(
        waveform=np.zeros((1, int(3 * params.SAMPLE_RATE))),
        expected_class_name='Silence')

  def testRandom(self):
    np.random.seed(51773)  # Ensure repeatability.
    self.clip_test(
        waveform=np.random.uniform(-1.0, +1.0,
                                   (1, int(3 * params.SAMPLE_RATE))),
        expected_class_name='White noise')

  def testSine(self):
    self.clip_test(
        waveform=np.reshape(
            np.sin(2 * np.pi * 440 * np.linspace(
                0, 3, int(3 *params.SAMPLE_RATE))),
            [1, -1]),
        expected_class_name='Sine wave')


if __name__ == '__main__':
  tf.test.main()
