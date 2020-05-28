import time
import datetime

class contexttimer:
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, typ, value, tb):
        self.seconds = time.time() - self.start
        self.time_str = datetime.timedelta(seconds=round(self.seconds))


def now():
    return datetime.datetime.now().strftime('%b/%d/%Y, %H:%M:%S')


def tic():
    return time.time()


def today():
    return datetime.date.today().strftime('%b%d')


def sec2str(seconds):
    ms = int(seconds * 1e3)
    ss, ms = divmod(ms, 1000)
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    dd, hh = divmod(hh, 24)
    if dd > 1:
        str_ = f'{dd} days, {hh}:{mm:02d}:{ss:02d}:{ms:02d}'
    elif dd == 1:
        str_ =  f'{dd} day, {hh}:{mm:02d}:{ss:02d}:{ms:02d}'
    else:
        str_ =            f'{hh}:{mm:02d}:{ss:02d}:{ms:02d}'
    return str_


if __name__ == "__main__":
    print(f'today: {today()}, now: {now()}')
    with contexttimer() as t:
        time.sleep(0.234)
    print(t.seconds, t.time_str)
    print(sec2str(time.time()))