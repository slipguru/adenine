"""Master slave."""
import os
import imp
import logging
import shutil

from collections import deque

from adenine.core import define_pipeline
from adenine.core.pipelines import pipe_worker

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()

    IS_MPI_JOB = True

except ImportError as e:
    print("mpi4py module not found. Adenine cannot run on multiple machines.")
    COMM = None
    SIZE = 1
    RANK = 0
    NAME = 'localhost'

    IS_MPI_JOB = False

MAX_RESUBMISSIONS = 2
# constants to use as tags in communications
DO_WORK = 100
EXIT = 200

# VERBOSITY
VERBOSITY = 1


def master(config):
    # Pipelines Definition
    pipes = define_pipeline.parse_steps(
        [config.step0, config.step1,
         config.step2, config.step3])

    print("master - end parse_steps")
    print(pipes)
    # RUN PIPELINES
    procs_ = COMM.Get_size()
    print("start running slaves", procs_)
    queue = deque(list(enumerate(pipes)))

    pipe_dump = dict()
    count = 0
    n_pipes = len(queue)

    # seed the slaves by sending work to each processor
    for rankk in range(1, min(procs_, n_pipes)):
        pipe_tuple = queue.popleft()
        COMM.send(pipe_tuple, dest=rankk, tag=DO_WORK)
        print("send to rank", rankk)

    # loop until there's no more work to do. If queue is empty skips the loop.
    while queue:
        pipe_tuple = queue.popleft()
        # receive result from slave
        status = MPI.Status()
        pipe_id, step_dump = COMM.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        pipe_dump[pipe_id] = step_dump
        count += 1
        # send to the same slave new work
        COMM.send(pipe_tuple, dest=status.source, tag=DO_WORK)

    # there's no more work to do, so receive all the results from the slaves
    for rankk in range(1, min(procs_, n_pipes)):
        print("master - waiting from", rankk)
        status = MPI.Status()
        pipe_id, step_dump = COMM.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        pipe_dump[pipe_id] = step_dump
        count += 1

    # tell all the slaves to exit by sending an empty message with the EXIT_TAG
    for rankk in range(1, procs_):
        print("master - killing", rankk)
        COMM.send(0, dest=rankk, tag=EXIT)

    print("terminating master")
    return pipe_dump


def slave(X):
    # Pipelines Evaluatio
    try:
        while True:
            status_ = MPI.Status()
            print("slave waiting", RANK)
            received = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status_)
            # check the tag of the received message
            if status_.tag == EXIT:
                return
            # do the work
            i, pipe = received
            print("slave received", RANK, i)
            pipe_id = 'pipe' + str(i)
            step_dump = pipe_worker(
                pipe_id, pipe, None, X)
            COMM.send((pipe_id, step_dump), dest=0, tag=0)

    except Exception as e:
        print("Quitting ... TB:", str(e))


def main(config_file):
    """Generate the pipelines."""
    # Load the configuration file
    config_path = os.path.abspath(config_file)

    # For some reason, it must be atomic
    imp.acquire_lock()
    config = imp.load_source('ade_config', config_path)
    imp.release_lock()

    from adenine.utils import extra
    extra.set_module_defaults(
        config, {
            'step0': {'Impute': [False]},
            'step1': {'None': [True]},
            'step2': {'None': [True]},
            'step3': {'None': [False]},
            'exp_tag': 'debug',
            'output_root_folder': 'results',
            'verbose': False})

    # Read the variables from the config file
    X, y = config.X, config.y

    if RANK == 0:
        # Get the experiment tag and the output root folder
        exp_tag, root = config.exp_tag, config.output_root_folder
        if not os.path.exists(root):
            os.makedirs(root)

        filename = '_'.join(('ade', exp_tag, extra.get_time()))
        logfile = os.path.join(root, filename + '.log')
        logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                            format='%(levelname)s (%(name)s): %(message)s')
        root_logger = logging.getLogger()
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if config.verbose else logging.ERROR)
        ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
        root_logger.addHandler(ch)
        pipes_dump = master(config)
    else:
        slave(X)

    if IS_MPI_JOB:
        # Wait for all jobs to end
        COMM.barrier()

    if RANK == 0:
        # Output Name
        output_filename = filename
        outfolder = os.path.join(root, output_filename)

        # Create exp folder into the root folder
        os.makedirs(outfolder)

        # pkl Dump
        import cPickle as pkl
        import gzip
        with gzip.open(os.path.join(outfolder,
                                    output_filename + '.pkl.tz'), 'w+') as f:
            pkl.dump(pipes_dump, f)
        logging.info("Dumped : {}".format(os.path.join(outfolder,
                                                       output_filename + '.pkl.tz')))
        with gzip.open(os.path.join(outfolder, '__data.pkl.tz'), 'w+') as f:
            pkl.dump({'X': X, 'y': y, 'index': config.index}, f)
        logging.info("Dumped : {}".format(os.path.join(outfolder,
                                                       '__data.pkl.tz')))

        # Copy the ade_config just used into the outFolder
        shutil.copy(config_path, os.path.join(outfolder, 'ade_config.py'))

        root_logger.handlers[0].close()

        # Move the logging file into the outFolder
        shutil.move(logfile, outfolder)
