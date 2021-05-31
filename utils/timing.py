from timeit import default_timer as timer
from functools import wraps
from datetime import timedelta


def timing(func):
    # see https://stackoverflow.com/a/27737385/5838091
    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = timer()

        result = func(*args, **kwargs)

        time_elapsed = timer() - start_time
        print(f"Total time for {func.__name__}: {str(timedelta(seconds=time_elapsed))}", flush=True)

        return result

    return wrap
