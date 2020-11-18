import numpy as np
import cv2


def load_img(filepath):
    img = cv2.imread(filepath)
    assert(img != None)
    return img


def get_window(src, r, c, windowsize=3, add_weight=False):
    """
    Return a windowsize x windowsize window surrounding (x, y)
    """

    # Src must have only 2 dimension
    assert(len(src.shape) == 2)

    rows, cols = src.shape

    # x, y must be inside of src
    assert(0 <= r and r < rows)
    assert(0 <= c and c < cols)

    k = windowsize // 2
    tmp = 1 if windowsize % 2 == 1 else 0

    rmin = max(0, r - k)
    rmax = min(rows, r + k + tmp)
    cmin = max(0, c - k)
    cmax = min(cols, c + k + tmp)

    top_padding = 0 if r >= k else k - r
    left_padding = 0 if c >= k else k - c
    bottom_padding = 0 if r + k + tmp <= rows else r + k + tmp - rows
    right_padding = 0 if c + k + tmp <= cols else c + k + tmp - cols

    window = np.zeros((windowsize, windowsize), dtype=np.float)

    window[top_padding:windowsize-bottom_padding, left_padding:windowsize-right_padding] = src[rmin:rmax, cmin:cmax]

    return window


def convolve(src, kernel, r, c):
    # Support square shape kernel only
    assert(kernel.shape[0] == kernel.shape[1])

    kernel = np.flip(kernel, (0, 1))

    window = get_window(src, r, c, kernel.shape[0])

    return (window * kernel).sum()

def mySobel(src):
    xkernel = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=np.float)
    ykernel = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]], dtype=np.float)

    Ix = np.zeros_like(src, dtype=np.float)
    Iy = np.zeros_like(src, dtype=np.float)

    for r in range(src.shape[0]):
        for c in range(src.shape[1]):
            Ix[r, c] = convolve(src, xkernel, r, c)
            Iy[r, c] = convolve(src, ykernel, r, c)

    return Ix, Iy

def derivative(src):
    Ix = np.zeros_like(src, dtype=np.float)
    Iy = np.zeros_like(src, dtype=np.float)

    Ix[:, :-1] = src[:, 1:]
    Iy[:-1, :] = src[1:, :]

    return Ix - src, Iy - src

def get_M(r, c, src, Ix, Iy):
    w = get_window(src, r, c)
    Ix = w * get_window(Ix, r, c)
    Iy = w * get_window(Iy, r, c)

    Ix_2 = Ix ** 2
    Iy_2 = Iy ** 2
    IxIy = Ix * Iy

    return np.block([[Ix_2, IxIy], 
                     [IxIy, Iy_2]])



def simple_harris(src):
    img = src.astype(np.float)
    response = np.zeros_like(img)
    Ix, Iy = derivative(img)

    for r in range(src.shape[0]):
        for c in range(src.shape[1]):
            M = get_M(r, c, img, Ix, Iy)
            response[r, c] = np.linalg.det(M) / np.trace(M)

    return response       

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('/home/thinh/code/python/CS231.L11.KHCL/images/council_house.jpg', 0)
    img = cv2.resize(img, (500 * img.shape[1] // img.shape[0], 500))
    open('/home/thinh/code/python/CS231.L11.KHCL/images/council_house.jpg')
    assert(img is not None)

    response = simple_harris(img)
    print('Finished')
    plt.imshow(response)
    plt.show()
