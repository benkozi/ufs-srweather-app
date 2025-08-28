import os

if __name__ == '__main__':
    for key, value in os.environ.items():
        print(f'{key}: {value}')
    raise ValueError('in exsrw_aqm_mm.py')