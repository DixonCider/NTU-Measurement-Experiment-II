import os
import numpy as np
import string
import random
import shutil
from functools import partial
from openpiv import tools, process, validation, filters, scaling, pyprocess, windef
from multiprocessing import Pool
import matplotlib.pyplot as plt
import warnings

ROOT = '/tmp2/cybai/piv/'

VISUALIZE_SCALES = {
    'proc_10Lpm': 0.5,
    'proc_20Lpm': 0.7,
    'proc_30Lpm': 1.1,
    'proc_40Lpm': 1.5,
    'proc_50Lpm': 2.5,
    'proc_60Lpm': 3.7,
}
SCALING_FACTOR = 12300
MIN_WINDOW_SIZE = 32
DT = 0.010264
LINE_WIDTH = 0.002

def run_single(index, scale=1, src_dir=None, save_dir=None):
    frame_a = tools.imread(os.path.join(src_dir, f'{index:06}.tif'))
    frame_b = tools.imread(os.path.join(src_dir, f'{index + 1:06}.tif'))
    # no background removal will be performed so 'mask' is initialized to 1 everywhere
    mask = np.ones(frame_a.shape, dtype=np.int32)

    # main algorithm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y, u, v, mask = process.WiDIM(frame_a.astype(np.int32), 
                                         frame_b.astype(np.int32),
                                         mask,
                                         min_window_size=MIN_WINDOW_SIZE,
                                         overlap_ratio=0.0,
                                         coarse_factor=2,
                                         dt=DT,
                                         validation_method='mean_velocity', 
                                         trust_1st_iter=1, 
                                         validation_iter=1, 
                                         tolerance=0.4,
                                         nb_iter_max=3,
                                         sig2noise_method='peak2peak')

    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=SCALING_FACTOR)


    tmp_fname = '.tmp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=32))
    tools.save(x, y, u, v, mask, filename=tmp_fname)
    tools.display_vector_field(tmp_fname, scale=scale, width=LINE_WIDTH) # scale: vector length ratio; width: line width of vector arrows
    os.remove(tmp_fname)

    # plt.quiver(x, y, u3, v3, color='blue')
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'{index:06}.pdf')
        print(save_path)

        plt.savefig(save_path)

def run_multiple(proc_dir):
    src_dir = os.path.join(ROOT, proc_dir)
    save_dir = os.path.join(ROOT, f'vf_{proc_dir}')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    wrapped_run_single = partial(run_single, scale=VISUALIZE_SCALES[proc_dir], src_dir=src_dir, save_dir=save_dir)

    n_instance = len([f for f in os.listdir(src_dir) if f.endswith('.tif')])
    n_cpus = 48

    with Pool(n_cpus) as p:
        p.map(wrapped_run_single, list(range(1, n_instance)))

def main():
    proc_dirs= [f'proc_{i}Lpm' for i in range(10, 70, 10)]

    for proc_dir in proc_dirs:
        run_multiple(proc_dir)

if __name__ == '__main__':
    main()
