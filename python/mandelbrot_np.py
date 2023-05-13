import time
import numpy as np

xmin = -1.75
xmax = 0.75
width = 4096
ymin = -1.25
ymax = 1.25
height = 4096
max_iter = 500

def mandelbrot_kernel(C):
    Z = np.zeros_like(C)
    step = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        Z = np.where(np.abs(Z) <= 2, Z ** 2 + C, Z)
        step += np.abs(Z) <= 2

    return step

def compute_mandelbrot():
    R, I = np.meshgrid(
        np.linspace(xmin, xmax, width),
        np.linspace(ymin, ymax, height),
    )
    C = R + I * 1j
    return mandelbrot_kernel(C)

t0 = time.time()
image = compute_mandelbrot()
print(image.shape)
t1 = time.time()
print(f"elapsed time: {(t1-t0) * 1000:.3f} ms")