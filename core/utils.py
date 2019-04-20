import logging
import time


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    logging.info(f'-------------------------------------')
    logging.info(f'Elapsed running time: {t_hour}h : {t_min}m : {t_sec}s')


def tac_api():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return t_hour, t_min, t_sec


def repeat_to_length(s, wanted):
    return (s * (wanted//len(s) + 1))[:wanted]