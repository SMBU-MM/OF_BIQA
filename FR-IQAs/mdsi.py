import sys, getopt
from PIL import Image
import numpy as np

def pooling(mat, ksize, method='mean', pad=False):
    m, n = mat.shape[:2]

    if pad:
        _ceil = lambda x,y: int(numpy.ceil(x/float(y)))
        ny = _ceil(m, ksize)
        nx = _ceil(n, ksize)
        size = (ny * ksize, nx * ksize) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m,:n,...] = mat
    else:
        ny = m // ksize
        nx = n // ksize
        mat_pad = mat[:ny*ksize, :nx*ksize, ...]

    new_shape = (ny, ksize, nx, ksize) + mat.shape[2:]

    if method=='max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1,3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1,3))

    return result

def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def mad(x, axis=None):
    return np.mean(np.absolute(x - np.mean(x, axis)), axis)

def mdsi(file1, file2, f=0, alpha=0.6):
    C1 = 140
    C2 = 55
    C3 = 550
    dx = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]]) / 3.
    dy = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]]) / 3.

    image1 = Image.open(file1).convert('RGB')
    image2 = Image.open(file2).convert('RGB')

    if not image1.size == image2.size:
        print("Dimensions must be equal!")
        return

    width, height = image1.size
    image1 = np.frombuffer(image1.tobytes(), dtype=np.uint8).reshape(height, width, 3)
    image2 = np.frombuffer(image2.tobytes(), dtype=np.uint8).reshape(height, width, 3)

    f = np.maximum(1, int(round(np.minimum(width, height) / 256))) if f==0 else f

    R1 = pooling(image1[:,:,0], f)
    R2 = pooling(image2[:,:,0], f)
    G1 = pooling(image1[:,:,1], f)
    G2 = pooling(image2[:,:,1], f)
    B1 = pooling(image1[:,:,2], f)
    B2 = pooling(image2[:,:,2], f)

    L1 = np.pad(0.299 * R1 + 0.587 * G1 + 0.114 * B1, ((1,1), (1,1)), "edge")
    L2 = np.pad(0.299 * R2 + 0.587 * G2 + 0.114 * B2, ((1,1), (1,1)), "edge")
    F = 0.5 * (L1 + L2)

    H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
    H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
    M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
    M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

    IxL1 = conv2d(L1, dx)
    IyL1 = conv2d(L1, dy)
    gR = np.power(np.power(IxL1, 2) + np.power(IyL1, 2), 0.5)

    IxL2 = conv2d(L2, dx)
    IyL2 = conv2d(L2, dy)
    gD = np.power(np.power(IxL2, 2) + np.power(IyL2, 2), 0.5)

    IxF = conv2d(F, dx)
    IyF = conv2d(F, dy)
    gF = np.power(np.power(IxF, 2) + np.power(IyF, 2), 0.5)

    GS12 = (2. * gR * gD + C1) / (np.power(gR, 2) + np.power(gD, 2) + C1)
    GS13 = (2. * gR * gF + C2) / (np.power(gR, 2) + np.power(gF, 2) + C2)
    GS23 = (2. * gD * gF + C2) / (np.power(gD, 2) + np.power(gF, 2) + C2)
    GS_HVS = np.clip(GS12 + GS23 - GS13, 0., 1.)

    CS = (2. * (H1 * H2 + M1 * M2) + C3).clip(min=0) / (np.power(H1, 2) + np.power(H2, 2) + np.power(M1, 2) + np.power(M2, 2) + C3)

    GCS = alpha * GS_HVS + (1. - alpha) * CS

    return np.power(mad(np.power(GCS, 0.25)), 0.5)

def usage():
    print("Usage: python mdsi.py [options] <reference_image> <image1> [<image2>...]\n")
    print("    Options:\n")
    print("    -f[n]  Downscaling factor (0 - auto). Default: 0")
    print("    -y     Compare only luma channel")
    print("    -h     Show this help")

def main():
    f = 0
    alpha = 0.6

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hyf:")
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-y":
            alpha = 1
        elif o == "-f":
            f = int(a)
        elif o in ("-h"):
            usage()
            sys.exit()
        else:
            assert False

    for arg in args[1:]:
        score = mdsi(args[0], arg, f, alpha)
        try:
            ind = arg.rindex('/') + 1
        except ValueError:
            ind = 0
        print(str(score) + "\t" + arg[ind:])

if __name__ == '__main__':
    main()
