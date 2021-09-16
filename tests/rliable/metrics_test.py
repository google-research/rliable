# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests various aggregate metrics."""

from absl.testing import absltest
import numpy as np
import rliable.metrics as metrics


class MetricsTest(absltest.TestCase):
  """Tests the main functions in the library."""

  def setUp(self):
    super().setUp()
    # Score matrices are of the form num_runs x num_tasks
    self._x = np.array([[1, 2], [2, 2], [1, 1], [2, 1]])
    self._y = np.array([[1, 1], [2, 2], [3, 3]])

  def test_aggregate_metrics(self):

    self.assertEqual(metrics.aggregate_median(self._x), 1.5)
    self.assertEqual(metrics.aggregate_mean(self._x), 1.5)
    self.assertEqual(metrics.aggregate_iqm(self._x), 1.5)

    self.assertEqual(metrics.aggregate_median(self._y), 2)
    self.assertEqual(metrics.aggregate_mean(self._y), 2)
    self.assertEqual(metrics.aggregate_iqm(self._y), 2)
    self.assertAlmostEqual(
        metrics.probability_of_improvement(self._x, self._y), 1/3)

if __name__ == "__main__":
  absltest.main()
