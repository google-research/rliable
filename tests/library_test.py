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
"""Tests functionality of library functions."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import rliable.library as rly
import rliable.metrics as metrics


class LibraryTest(parameterized.TestCase):
  """Tests the main functions in the library."""

  def setUp(self):
    super().setUp()
    num_runs, num_tasks = 5, 16
    self._x = np.arange(num_runs * num_tasks).reshape(num_runs, num_tasks)
    self._y = np.flip(self._x, axis=0)
    self._z = np.stack([self._x, 2*self._x], axis=-1)

  @parameterized.named_parameters(
      dict(testcase_name="runs_only", task_bootstrap=False),
      dict(testcase_name="runs_and_tasks", task_bootstrap=True))
  def test_stratified_bootstrap(self, task_bootstrap):
    """Tests StratifiedBootstrap."""
    bs = rly.StratifiedBootstrap(
        self._x, y=self._y, z=self._z, task_bootstrap=task_bootstrap)
    for data, kwdata in bs.bootstrap(5):
      index = bs.index
      self.assertLen(data, 1)
      self.assertLen(list(kwdata.keys()), 2)
      np.testing.assert_array_equal(self._x[index], data[0])
      np.testing.assert_array_equal(self._y[index], kwdata["y"])
      np.testing.assert_array_equal(self._z[index], kwdata["z"])
      np.testing.assert_array_equal(self._y[index], bs.y)
      np.testing.assert_array_equal(self._z[index], bs.z)

  def test_stratified_independent_bootstrap(self):
    """Tests StratifiedIndependentBootstrap."""
    bs = rly.StratifiedIndependentBootstrap(self._x, y=self._y, z=self._z)
    for data, kwdata in bs.bootstrap(2):
      self.assertIsInstance(bs.index, tuple)
      index_x, kw_index = bs.index
      self.assertLen(data, 1)
      self.assertLen(list(kwdata.keys()), 2)
      np.testing.assert_array_equal(self._x[index_x[0]], data[0])
      np.testing.assert_array_equal(self._y[kw_index["y"]], kwdata["y"])
      np.testing.assert_array_equal(self._z[kw_index["z"]], kwdata["z"])
      np.testing.assert_array_equal(self._y[kw_index["y"]], bs.y)
      np.testing.assert_array_equal(self._z[kw_index["z"]], bs.z)

  @parameterized.named_parameters(
      dict(
          testcase_name="percentile", method="percentile",
          task_bootstrap=False),
      dict(testcase_name="basic", method="basic", task_bootstrap=True),
      dict(testcase_name="bc", method="bc", task_bootstrap=False),
      dict(testcase_name="bca", method="bca", task_bootstrap=False))
  def test_interval_estimation(self, method, task_bootstrap):

    def metric_func(x):
      return np.array([
          metrics.aggregate_iqm(x),
          metrics.aggregate_mean(x),
          metrics.aggregate_median(x),
          metrics.aggregate_optimality_gap(x)
      ])

    score_dict = {"method": self._x, "baseline": self._y}
    # Use small number of bootstrap samples for testing.
    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict,
        metric_func,
        method=method,
        task_bootstrap=task_bootstrap,
        reps=100)
    for key, scores in score_dict.items():
      lower_ci_estimates, upper_ci_estimates = interval_estimates[key]
      np.testing.assert_array_equal(point_estimates[key], metric_func(scores))
      np.testing.assert_array_less(lower_ci_estimates, point_estimates[key])
      np.testing.assert_array_less(point_estimates[key], upper_ci_estimates)

  @parameterized.named_parameters(
      dict(testcase_name="score", use_score_distribution=True),
      dict(testcase_name="average_scores", use_score_distribution=False))
  def test_performance_profiles(self, use_score_distribution):
    tau_list = [0.0, 0.25, 0.5, 1.0]
    score_dict = {"x": self._x, "y": self._y}
    profiles, profile_cis = rly.create_performance_profile(
        score_dict,
        tau_list,
        use_score_distribution=use_score_distribution,
        reps=100)
    for key in score_dict:
      self.assertIsInstance(profiles[key], np.ndarray)
      np.testing.assert_array_equal(profiles[key],
                                    sorted(profiles[key], reverse=True))
      self.assertLen(profiles[key], len(tau_list))
      self.assertEqual(profile_cis[key].shape, (2, len(tau_list)))

  def test_improvement_probability_cis(self):
    score_dict = {"x,y": [self._x, self._y]}
    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict, metrics.probability_of_improvement, reps=100)
    for key, scores in score_dict.items():
      lower_ci_estimates, upper_ci_estimates = interval_estimates[key]
      np.testing.assert_array_equal(point_estimates[key],
                                    metrics.probability_of_improvement(*scores))
      np.testing.assert_array_less(lower_ci_estimates, point_estimates[key])
      np.testing.assert_array_less(point_estimates[key], upper_ci_estimates)


if __name__ == "__main__":
  absltest.main()
