import abc
import gc
import logging
import os
import time
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from multiprocessing import Pipe, Process
from multiprocessing.pool import ThreadPool

import numpy as np

from jass_mu_zero.environment.networking.worker_config import WorkerConfig
from jass_mu_zero.factory import get_network
from jass_mu_zero.mu_zero.network.network_base import AbstractNetwork


class BaseAsyncMetric:

    def __init__(
            self,
            worker_config: WorkerConfig,
            network_path: str,
            parallel_threads: int,
            metric_method,
            init_method=None):
        self.init_method = init_method
        self.metric_method = metric_method
        self.worker_config = worker_config
        self.parallel_threads = parallel_threads
        self.network_path = network_path
        self._latest_result = None
        self.parent_conn, self.child_conn = Pipe(duplex=True)

        self._start_calculation()

    def _start_calculation(self):
        self.collecting_process = Process(target=self._calculate_continuously)
        self.collecting_process.start()

    def _calculate_continuously(self):
        import tensorflow as tf
        #tf.config.run_functions_eagerly(True)
        tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

        while not os.path.exists(self.network_path):
            logging.info(f"waiting for model to be saved at {self.network_path}")
            time.sleep(1)

        pool = ThreadPool(processes=self.parallel_threads)

        if self.init_method is not None:
            init_vars = self.init_method()

        network = get_network(self.worker_config, network_path=self.network_path)

        while True:
            try:
                network.load(self.network_path)

                if self.init_method is None:
                    params = [self.get_params(i, network) for i in range(self.parallel_threads)]
                else:
                    params = [self.get_params(i, network, init_vars) for i in range(self.parallel_threads)]

                results = pool.starmap(self.metric_method, params)

                if len(results) == 1 and type(results[0]) is dict:
                    self.child_conn.send(results[0])
                else:
                    self.child_conn.send(float(np.mean(results)))

                del results, params
                gc.collect()

                # wait until latest result is fetched
                self.child_conn.recv()
                while self.child_conn.poll():
                    self.child_conn.recv()

            except Exception as e:
                logging.error(f"{type(self)}: Encountered error {e}, continuing anyways")

    @abc.abstractmethod
    def get_params(self, thread_nr: int, network: AbstractNetwork, init_vars=None) -> []:
        pass

    def get_latest_result(self) -> dict:
        while self.parent_conn.poll():
            del self._latest_result
            self._latest_result = self.parent_conn.recv()
            self.parent_conn.send(True)

        if type(self._latest_result) is dict:
            return self._latest_result
        else:
            return {
                self.get_name(): self._latest_result
            }

    def poll_till_next_result_available(self, timeout=.1):
        while not self.parent_conn.poll():
            time.sleep(timeout)

    @abc.abstractmethod
    def get_name(self):
        pass

    def __del__(self):
        self.collecting_process.terminate()

