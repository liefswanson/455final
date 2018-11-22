import numpy as np
import imageio
from timeit import default_timer as timer
from numba import vectorize, cuda, jit, uint16
from matplotlib import pyplot as plt
import itertools

@vectorize(['uint8(uint8, uint8)'], target='cuda')
def brighten_vectorized(im, bright):
    return min(im*bright**5, 255)

@vectorize(['uint8(uint8, uint8)'], target='cpu')
def brighten(im, bright):
    return min(im*bright**5, 255)

@vectorize(['uint8(uint8, uint8)'], target='cuda')
def control_vectorized(im, bright):
    return im

@vectorize(['uint8(uint8, uint8)'], target='cpu')
def control(im, bright):
    return im


def test(frames, name, fn, ctrl, im, bright):
    fn(im, bright)
    start = timer()
    for _ in range(frames):
        result = fn(im, bright)
    total = timer() - start

    ctrl(im, bright)
    start = timer()
    for _ in range(frames):
        result = ctrl(im, bright)
    control_total = timer() - start

    print(name)
    print('%f fps' % (frames / total)) # actual time
    print('%f fps' % (frames / control_total)) # travel time
    print('%f fps' % (frames / (total-control_total))) # theoretical best case (actual - travel)


im = imageio.imread('shot.png')
bright = np.full(im.shape, 2, dtype=np.uint8)
print(im.shape)
print(im.dtype)

test(150, 'vectorized', brighten_vectorized, control_vectorized, im, bright)
test(150, 'unvectorized', brighten, control, im, bright)

exit()

after = brighten_vectorized(im, bright)
fig = plt.figure(figsize=(2,2))
fig.add_subplot(2,2,1)
plt.imshow(im)
fig.add_subplot(2,2,2)
plt.imshow(after)
plt.show()