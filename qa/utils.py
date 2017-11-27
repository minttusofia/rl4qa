import time


def print_time_taken(prev_t):
    new_t = time.time()
    print(' ' + str(new_t - prev_t) + ' s')
    return new_t


