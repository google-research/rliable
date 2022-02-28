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
"""Plotting utility functions for aggregate metrics and performance profiles."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _non_linear_scaling(performance_profiles,
                        tau_list,
                        xticklabels=None,
                        num_points=5,
                        log_base=2):
  """Returns non linearly scaled tau as well as corresponding xticks.

  The non-linear scaling of a certain range of threshold values is proportional
  to fraction of runs that lie within that range.

  Args:
    performance_profiles: A dictionary mapping a method to its performance
      profile, where each profile is computed using thresholds in `tau_list`.
    tau_list: List or 1D numpy array of threshold values on which the profile is
      evaluated.
    xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
    num_points: If `xticklabels` are not passed, then specifices the number of
      indices to be generated on a log scale.
    log_base: Base of the logarithm scale for non-linear scaling.

  Returns:
    nonlinear_tau: Non-linearly scaled threshold values.
    new_xticks: x-axis ticks from `nonlinear_tau` that would be plotted.
    xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
  """

  methods = list(performance_profiles.keys())
  nonlinear_tau = np.zeros_like(performance_profiles[methods[0]])
  for method in methods:
    nonlinear_tau += performance_profiles[method]
  nonlinear_tau /= len(methods)
  nonlinear_tau = 1 - nonlinear_tau

  if xticklabels is None:
    tau_indices = np.int32(
        np.logspace(
            start=0,
            stop=np.log2(len(tau_list) - 1),
            base=log_base,
            num=num_points))
    xticklabels = [tau_list[i] for i in tau_indices]
  else:
    tau_as_list = list(tau_list)
    # Find indices of x which are in `tau`
    tau_indices = [tau_as_list.index(x) for x in xticklabels]
  new_xticks = nonlinear_tau[tau_indices]
  return nonlinear_tau, new_xticks, xticklabels


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
  """Helper function for decorating plots."""
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  # Deal with ticks and the blank space at the origin
  ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
  ax.spines['left'].set_position(('outward', hrect))
  ax.spines['bottom'].set_position(('outward', wrect))
  return ax


def _annotate_and_decorate_axis(ax,
                                labelsize='x-large',
                                ticklabelsize='x-large',
                                xticks=None,
                                xticklabels=None,
                                yticks=None,
                                legend=False,
                                grid_alpha=0.2,
                                legendsize='x-large',
                                xlabel='',
                                ylabel='',
                                wrect=10,
                                hrect=10):
  """Annotates and decorates the plot."""
  ax.set_xlabel(xlabel, fontsize=labelsize)
  ax.set_ylabel(ylabel, fontsize=labelsize)
  if xticks is not None:
    ax.set_xticks(ticks=xticks)
    ax.set_xticklabels(xticklabels)
  if yticks is not None:
    ax.set_yticks(yticks)
  ax.grid(True, alpha=grid_alpha)
  ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
  if legend:
    ax.legend(fontsize=legendsize)
  return ax


def plot_performance_profiles(performance_profiles,
                              tau_list,
                              performance_profile_cis=None,
                              use_non_linear_scaling=False,
                              ax=None,
                              colors=None,
                              color_palette='colorblind',
                              alpha=0.15,
                              figsize=(10, 5),
                              xticks=None,
                              yticks=None,
                              xlabel=r'Normalized Score ($\tau$)',
                              ylabel=r'Fraction of runs with score $> \tau$',
                              linestyles=None,
                              **kwargs):
  """Plots performance profiles with stratified confidence intervals.

  Args:
    performance_profiles: A dictionary mapping a method to its performance
      profile, where each profile is computed using thresholds in `tau_list`.
    tau_list: List or 1D numpy array of threshold values on which the profile is
      evaluated.
    performance_profile_cis: The confidence intervals (default 95%) of
      performance profiles evaluated at all threshdolds in `tau_list`.
    use_non_linear_scaling: Whether to scale the x-axis in proportion to the
      number of runs within any specified range.
    ax: `matplotlib.axes` object.
    colors: Maps each method to a color. If None, then this mapping is created
      based on `color_palette`.
    color_palette: `seaborn.color_palette` object. Used when `colors` is None.
    alpha: Changes the transparency of the shaded regions corresponding to the
      confidence intervals.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    xticks: The list of x-axis tick locations. Passing an empty list removes all
      xticks.
    yticks: The list of y-axis tick locations between 0 and 1. If None, defaults
      to `[0, 0.25, 0.5, 0.75, 1.0]`.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    linestyles: Maps each method to a linestyle. If None, then the 'solid'
      linestyle is used for all methods.
    **kwargs: Arbitrary keyword arguments for annotating and decorating the
      figure. For valid arguments, refer to `_annotate_and_decorate_axis`.

  Returns:
    `matplotlib.axes.Axes` object used for plotting.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=figsize)

  if colors is None:
    keys = performance_profiles.keys()
    color_palette = sns.color_palette(color_palette, n_colors=len(keys))
    colors = dict(zip(list(keys), color_palette))

  if linestyles is None:
    linestyles = {key: 'solid' for key in performance_profiles.keys()}

  if use_non_linear_scaling:
    tau_list, xticks, xticklabels = _non_linear_scaling(performance_profiles,
                                                        tau_list, xticks)
  else:
    xticklabels = xticks

  for method, profile in performance_profiles.items():
    ax.plot(
        tau_list,
        profile,
        color=colors[method],
        linestyle=linestyles[method],
        linewidth=kwargs.pop('linewidth', 2.0),
        label=method)
    if performance_profile_cis is not None:
      if method in performance_profile_cis:
        lower_ci, upper_ci = performance_profile_cis[method]
        ax.fill_between(
            tau_list, lower_ci, upper_ci, color=colors[method], alpha=alpha)

  if yticks is None:
    yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
  return _annotate_and_decorate_axis(
      ax,
      xticks=xticks,
      yticks=yticks,
      xticklabels=xticklabels,
      xlabel=xlabel,
      ylabel=ylabel,
      **kwargs)


def plot_interval_estimates(point_estimates,
                            interval_estimates,
                            metric_names,
                            algorithms=None,
                            colors=None,
                            color_palette='colorblind',
                            max_ticks=4,
                            subfigure_width=3.4,
                            row_height=0.37,
                            xlabel_y_coordinate=-0.1,
                            xlabel='Normalized Score',
                            **kwargs):
  """Plots various metrics with confidence intervals.

  Args:
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metrics to plot.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    metric_names: Names of the metrics corresponding to `point_estimates`.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Maps each method to a color. If None, then this mapping is created
      based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
      `plt.MaxNLocator`.
    subfigure_width: Width of each subfigure.
    row_height: Height of each row in a subfigure.
    xlabel_y_coordinate: y-coordinate of the x-axis label.
    xlabel: Label for the x-axis.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    fig: A matplotlib Figure.
    axes: `axes.Axes` or array of Axes.
  """

  if algorithms is None:
    algorithms = point_estimates.keys()
  num_metrics = len(point_estimates[algorithms[0]])
  figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
  fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
  if colors is None:
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))
  h = kwargs.pop('interval_height', 0.6)

  for idx, metric_name in enumerate(metric_names):
    for alg_idx, algorithm in enumerate(algorithms):
      ax = axes[idx]
      # Plot interval estimates.
      lower, upper = interval_estimates[algorithm][:, idx]
      ax.barh(
          y=alg_idx,
          width=upper - lower,
          height=h,
          left=lower,
          color=colors[algorithm],
          alpha=0.75,
          label=algorithm)
      # Plot point estimates.
      ax.vlines(
          x=point_estimates[algorithm][idx],
          ymin=alg_idx - (7.5 * h / 16),
          ymax=alg_idx + (6 * h / 16),
          label=algorithm,
          color='k',
          alpha=0.5)

    ax.set_yticks(list(range(len(algorithms))))
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    if idx != 0:
      ax.set_yticks([])
    else:
      ax.set_yticklabels(algorithms, fontsize='x-large')
    ax.set_title(metric_name, fontsize='xx-large')
    ax.tick_params(axis='both', which='major')
    _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)
  fig.text(0.4, xlabel_y_coordinate, xlabel, ha='center', fontsize='xx-large')
  plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
  return fig, axes


def plot_sample_efficiency_curve(frames,
                                 point_estimates,
                                 interval_estimates,
                                 algorithms,
                                 colors=None,
                                 color_palette='colorblind',
                                 figsize=(7, 5),
                                 xlabel=r'Number of Frames (in millions)',
                                 ylabel='Aggregate Human Normalized Score',
                                 ax=None,
                                 labelsize='xx-large',
                                 ticklabelsize='xx-large',
                                 **kwargs):
  """Plots an aggregate metric with CIs as a function of environment frames.

  Args:
    frames: Array or list containing environment frames to mark on the x-axis.
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metric corresponding to the values in `frames`.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Dictionary that maps each algorithm to a color. If None, then this
      mapping is created based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    ax: `matplotlib.axes` object.
    labelsize: Font size of the x-axis label.
    ticklabelsize: Font size of the ticks.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    `axes.Axes` object containing the plot.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)
  if algorithms is None:
    algorithms = list(point_estimates.keys())
  if colors is None:
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))

  for algorithm in algorithms:
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ax.plot(
        frames,
        metric_values,
        color=colors[algorithm],
        marker=kwargs.pop('marker', 'o'),
        linewidth=kwargs.pop('linewidth', 2),
        label=algorithm)
    ax.fill_between(
        frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.2)

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)


def plot_probability_of_improvement(
    probability_estimates,
    probability_interval_estimates,
    pair_separator=',',
    ax=None,
    figsize=(4, 3),
    colors=None,
    color_palette='colorblind',
    alpha=0.75,
    xticks=None,
    xlabel='P(X > Y)',
    left_ylabel='Algorithm X',
    right_ylabel='Algorithm Y',
    **kwargs):
  """Plots probability of improvement with confidence intervals.

  Args:
    probability_estimates: Dictionary mapping algorithm pairs (X, Y) to a
      list or array containing probability of improvement of X over Y.
    probability_interval_estimates: Dictionary mapping algorithm pairs (X, Y)
      to interval estimates corresponding to the `probability_estimates`.
      Typically, consists of stratified independent bootstrap CIs.
    pair_separator: Each algorithm pair name in dictionaries above is joined by
      a string separator. For example, if the pairs are specified as 'X;Y', then
      the separator corresponds to ';'. Defaults to ','.
    ax: `matplotlib.axes` object.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    colors: Maps each algorithm pair id to a color. If None, then this mapping
      is created based on `color_palette`.
    color_palette: `seaborn.color_palette` object. Used when `colors` is None.
    alpha: Changes the transparency of the shaded regions corresponding to the
      confidence intervals.
    xticks: The list of x-axis tick locations. Passing an empty list removes all
      xticks.
    xlabel: Label for the x-axis. Defaults to 'P(X > Y)'.
    left_ylabel: Label for the left y-axis. Defaults to 'Algorithm X'.
    right_ylabel: Label for the left y-axis. Defaults to 'Algorithm Y'.
    **kwargs: Arbitrary keyword arguments for annotating and decorating the
      figure. For valid arguments, refer to `_annotate_and_decorate_axis`.

  Returns:
    `axes.Axes` which contains the plot for probability of improvement.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=figsize)
  if not colors:
    colors = sns.color_palette(
        color_palette, n_colors=len(probability_estimates))
  h = kwargs.pop('interval_height', 0.6)
  wrect = kwargs.pop('wrect', 5)
  ticklabelsize = kwargs.pop('ticklabelsize', 'x-large')
  labelsize = kwargs.pop('labelsize', 'x-large')
  # x-position of the y-label
  ylabel_x_coordinate = kwargs.pop('ylabel_x_coordinate', 0.2)
  # x-position of the y-label

  twin_ax = ax.twinx()
  all_algorithm_x, all_algorithm_y = [], []

  # Main plotting code
  for idx, (algorithm_pair, prob) in enumerate(probability_estimates.items()):
    lower, upper = probability_interval_estimates[algorithm_pair]
    algorithm_x, algorithm_y = algorithm_pair.split(pair_separator)
    all_algorithm_x.append(algorithm_x)
    all_algorithm_y.append(algorithm_y)

    ax.barh(
        y=idx,
        width=upper - lower,
        height=h,
        left=lower,
        color=colors[idx],
        alpha=alpha,
        label=algorithm_x)
    twin_ax.barh(
        y=idx,
        width=upper - lower,
        height=h,
        left=lower,
        color=colors[idx],
        alpha=0.0,
        label=algorithm_y)
    ax.vlines(
        x=prob,
        ymin=idx - 7.5 * h / 16,
        ymax=idx + (6 * h / 16),
        color='k',
        alpha=min(alpha + 0.1, 1.0))

  # Beautify plots
  yticks = range(len(probability_estimates))
  ax = _annotate_and_decorate_axis(
      ax,
      xticks=xticks,
      yticks=yticks,
      xticklabels=xticks,
      xlabel=xlabel,
      ylabel=left_ylabel,
      wrect=wrect,
      ticklabelsize=ticklabelsize,
      labelsize=labelsize,
      **kwargs)
  twin_ax = _annotate_and_decorate_axis(
      twin_ax,
      xticks=xticks,
      yticks=yticks,
      xticklabels=xticks,
      xlabel=xlabel,
      ylabel=right_ylabel,
      wrect=wrect,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      grid_alpha=0.0,
      **kwargs)
  twin_ax.set_yticklabels(all_algorithm_y, fontsize='large')
  ax.set_yticklabels(all_algorithm_x, fontsize='large')
  twin_ax.set_ylabel(
      right_ylabel,
      fontweight='bold',
      rotation='horizontal',
      fontsize=labelsize)
  ax.set_ylabel(
      left_ylabel,
      fontweight='bold',
      rotation='horizontal',
      fontsize=labelsize)
  twin_ax.set_yticklabels(all_algorithm_y, fontsize=ticklabelsize)
  ax.set_yticklabels(all_algorithm_x, fontsize=ticklabelsize)
  ax.tick_params(axis='both', which='major')
  twin_ax.tick_params(axis='both', which='major')
  ax.spines['left'].set_visible(False)
  twin_ax.spines['left'].set_visible(False)
  ax.yaxis.set_label_coords(-ylabel_x_coordinate, 1.0)
  twin_ax.yaxis.set_label_coords(1 + 0.7 * ylabel_x_coordinate,
                                 1 + 0.6 * ylabel_x_coordinate)

  return ax
