import logging
import numpy
import time
import threading

R = threading.Lock()

class Logger:
    def __init__(self, log_name, graph_path, args):
        # 第一步，创建一个logger
        self.logger = self.build_log(log_name)
        self.build_tb_log(graph_path)
        self.args = args
        self.keys = dict()
        if 'summary_output_times' in self.args.keys():
            self.summary_times = self.args['summary_output_times']
        else:
            self.summary_times = 1


    def build_log(self, name):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = name + '/out.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)
        return logger

    def build_tb_log(self, path):
        from tensorboard_logger import configure, log_value, log_histogram
        configure(path)
        self.tb_logger = log_value
        self.tb_h_logger = log_histogram

    def write_tb_log(self, key, value, t):
        if self.args['output_graph']:
            if t % self.summary_times != 0:
                return
            #print(key, value)
            if type(value) is numpy.ndarray or type(value) is list:
                R.acquire()
                # Keep TensorBoard's horizontal axis in training episodes.
                self.tb_h_logger(key, value, t)
                # Reward vectors are per-agent returns. Log their sum as the
                # team return as well, which is the scalar learning curve used
                # in the MAPTF paper's MPE figures.
                if key == 'reward_mean':
                    self.tb_logger('team/reward_mean', float(numpy.sum(value)), t)
                R.release()
            else:
                R.acquire()
                self.tb_logger(key, value, t)
                R.release()
        else:
            return


    def write_log(self, msg, type='info'):
        if type == 'debug':
            self.logger.debug(msg)
        elif type == 'info':
            self.logger.log(msg)
        elif type == 'warning':
            self.logger.warning(msg)
        elif type == 'error':
            self.logger.error(msg)
        elif type == 'critical':
            self.logger.critical(msg)

