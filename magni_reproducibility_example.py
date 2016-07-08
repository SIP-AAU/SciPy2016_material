"""
Copyright (c) 2016,
Christian Schou Oxvig, Thomas Arildsen, and Torben Larsen
Aalborg University, Department of Electronic Systems, Signal and Information
Processing, Fredrik Bajers Vej 7, DK-9220 Aalborg, Denmark.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

An example of how to use the magni.reproducibility package for storing metadata
along with results from a computational experiment. The example is based on
simulating the Mandelbrot set. The Mandelbrot set simulation is in no way
optimised since the purpose of this example is to showcase the functionality in
magni.reproducibility.

Magni is available here: http://dx.doi.org/10.5278/VBN/MISC/Magni

"""

from __future__ import division
import argparse
from pprint import pprint

import matplotlib as mpl; mpl.use('Agg')
import magni
from magni.utils.validation import decorate_validation as _decorate_validation
from magni.utils.validation import validate_generic as _generic
from magni.utils.validation import validate_levels as _levels
from magni.utils.validation import validate_numeric as _numeric
import numpy as np
import psutil


def compute_mandelbrot_point(c, max_iterations, threshold):
    """
    Determine if a complex plane point is part of the Mandelbrot set.

    Parameters
    ----------
    c : complex
        The complex plane point.
    max_iterations : int
        The maximum number of iterations to use in the point stability check.
    threshold : int
        The threshold used in the point stability check.

    Returns
    -------
    score : float
        The stability score.

    Notes
    -----
    The Mandelbrot set consists of all values `c` in the complex plane for
    which the so-called orbit of 0 when iterated under the quadratic map
    z_{n+1} = z_{n}^2 + c remains bounded [1]_.

    In this simulation we iterate the quadratic map for up to `max_iterations`
    and assign a stability `score` based on the percentage of the
    `max_iterations` it takes to the surpass the `threshold`. A `score` <= 10
    indicates an unstable point which is not considered part of the Mandelbrot
    set whereas a `score` > 10 indicates a stable point which is considered
    part of the Mandelbrot set.

    .. [1] http://en.wikipedia.org/wiki/Mandelbrot_set

    """

    @_decorate_validation
    def validate_input():
        _numeric('c', 'complex')
        _numeric('max_iterations', 'integer', range_='(0;inf)')
        _numeric('threshold', 'integer', range_='(0;inf)')

    validate_input()

    z = np.complex(0)
    score = 100

    for t in range(max_iterations):
        z = z**2 + c
        z_modulus = np.abs(z)

        if z_modulus > threshold or not np.isfinite(z_modulus):
            score = 100 * (t + 1) / max_iterations
            break

    return score


def get_mandelbrot_tasks(re_min, re_max, im_min, im_max, num_points):
    """
    Return a list of tasks for a parallel computation of the Mandelbrot set.

    Parameters
    ----------
    re_min : float
        Real axis minimum in the complex plane grid.
    re_max : float
        Real axis maximum in the complex plane grid.
    im_min : float
        Imaginary axis minimum in the complex plane grid.
    im_max : float
        Imaginary axis maximum in the complex plane grid.
    num_points : int
        Number of points along each direction in complex plane grid.

    Returns
    -------
    tasks : list
        The the list of simulation tasks for the Mandelbrot set simulation.

    """

    @_decorate_validation
    def validate_input():
        _numeric('re_min', 'floating')
        _numeric('re_max', 'floating', range_='({};inf)'.format(re_min))
        _numeric('im_min', 'floating')
        _numeric('im_max', 'floating', range_='({};inf)'.format(im_min))
        _numeric('num_points', 'integer', range_='[1;inf)')

    validate_input()

    re = np.linspace(re_min, re_max, num_points)
    im = np.linspace(im_min, im_max, num_points)

    tasks = [{'complex_plane_point_value': re[re_ix] + 1j * im[im_ix],
              'complex_plane_point_index': (re_ix, im_ix),
              'max_iterations': 10000,
              'threshold': 100}
             for re_ix in range(num_points)
             for im_ix in range(num_points)]

    return tasks


def run_mandelbrot_simulation(h5_path=None, task=None):
    """
    Run a Mandelbrot task simulation.

    Parameters
    ----------
    h5_path : str
        The path to the HDF5 file in which the result is to be saved.
    task : dict
        The simulation task specification.

    """

    @_decorate_validation
    def validate_input():
        _generic('h5_path', 'string')
        _generic('task', 'mapping')

    validate_input()

    pprint('Now handling complex plane point: {}'.format(
        task['complex_plane_point_index']))

    re_ix = task['complex_plane_point_index'][0]
    im_ix = task['complex_plane_point_index'][1]

    mandelbrot_result = compute_mandelbrot_point(
        task['complex_plane_point_value'],
        task['max_iterations'], task['threshold'])

    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5_file:
        mandelbrot_array = h5_file.root.simulation_result.mandelbrot
        mandelbrot_array[im_ix, re_ix] = mandelbrot_result


if __name__ == '__main__':
    """
    Do a parallel computation of the Mandelbrot set.

    See http://en.wikipedia.org/wiki/Mandelbrot_set for more info about the
    Mandelbrot set.

    usage: scipy_2016_magni_reproducibility_example.py [-h]
                                                   re_min re_max im_min im_max
                                                   num_points

    magni.reproducibility Mandelbrot example

    positional arguments:
      re_min      Real axis minimum
      re_max      Real axis maximum
      im_min      Imaginary axis minimum
      im_max      Imaginary axis maximum
      num_points  Number of points along each direction in complex plane grid

    optional arguments:
      -h, --help  show this help message and exit

    """

    # Argument parsing
    arg_parser = argparse.ArgumentParser(
        description='magni.reproducibility Mandelbrot example')
    arg_parser.add_argument(
        're_min', action='store', type=float, help='Real axis minimum')
    arg_parser.add_argument(
        're_max', action='store', type=float, help='Real axis maximum')
    arg_parser.add_argument(
        'im_min', action='store', type=float, help='Imaginary axis minimum')
    arg_parser.add_argument(
        'im_max', action='store', type=float, help='Imaginary axis maximum')
    arg_parser.add_argument(
        'num_points', action='store', type=int,
        help='Number of points along each direction in complex plane grid')
    args = arg_parser.parse_args()

    # Path to database to store results in
    h5_path = 'mandelbrot.hdf5'
    pprint('Results database: {}'.format(h5_path))

    # Get tasks to run in parallel
    tasks = get_mandelbrot_tasks(
        args.re_min, args.re_max, args.im_min, args.im_max, args.num_points)
    kwargs = [{'h5_path': h5_path, 'task': task} for task in tasks]
    pprint('Total number of simulation tasks: {}'.format(len(tasks)))

    # Setup magni multiprocessing
    workers = psutil.cpu_count(logical=False)
    magni.utils.multiprocessing.config.update(
        {'workers': workers, 'prefer_futures': True,
         're_raise_exceptions': True, 'max_broken_pool_restarts': 10})
    pprint('Magni multiprocessing config:')
    pprint(dict(magni.utils.multiprocessing.config.items()))

    # Create results database and array
    magni.reproducibility.io.create_database(h5_path)
    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5_file:
        h5_file.create_array(
            '/simulation_result', 'mandelbrot', createparents=True,
            obj=np.zeros((args.num_points, args.num_points)))

    # Store experiment parameters and checksums of this script
    ann_sub_group = 'custom_annotations'
    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5_file:
        magni.reproducibility.io.write_custom_annotation(
            h5_file, 'script_input_parameters', vars(args),
            annotations_sub_group=ann_sub_group)
        magni.reproducibility.io.write_custom_annotation(
            h5_file, 'script_checksums',
            magni.reproducibility.data.get_file_hashes(__file__),
            annotations_sub_group=ann_sub_group)

    # Run simulation
    magni.utils.multiprocessing.process(
        run_mandelbrot_simulation, kwargs_list=kwargs, maxtasks=1)

    # Write end time annotation
    with magni.utils.multiprocessing.File(h5_path, mode='a') as h5_file:
        magni.reproducibility.io.write_custom_annotation(
            h5_file, 'end_time', magni.reproducibility.data.get_datetime())
