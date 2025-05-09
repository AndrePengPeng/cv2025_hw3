import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = u[:, 0].reshape((N,1))
    uy = u[:, 1].reshape((N,1))
    vx = v[:, 0].reshape((N,1))
    vy = v[:, 1].reshape((N,1))
    
    A1 = np.concatenate((ux, uy, np.ones((N, 1)), np.zeros((N, 3)), -1 * np.multiply(ux, vx), -1 * np.multiply(uy, vx), -1 * vx), axis = 1)
    A2 = np.concatenate((np.zeros((N, 3)), ux, uy, np.ones((N, 1)), -1 * np.multiply(ux, vy), -1 * np.multiply(uy, vy), -1 * vy), axis = 1)
    stacked = np.stack((A1, A2))
    A = stacked.transpose(1, 0, 2).reshape(-1, A1.shape[1])
    # TODO: 2.solve H with A
    U, sigma, VT = np.linalg.svd(A)
    H = VT[-1,:]/VT[-1,-1]
    H = H.reshape(3,3)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xc, yc = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xr = xc.reshape((1, (xmax-xmin)*(ymax-ymin)))
    yr = yc.reshape((1, (xmax-xmin)*(ymax-ymin)))
    U = np.concatenate((xr, yr, np.ones((1, (xmax-xmin)*(ymax-ymin)))), axis=0)
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H_inv, U)
        Vx, Vy, _ = V / V[2,:]
        srcx = np.round(Vx.reshape(ymax-ymin,xmax-xmin)).astype(int)
        srcy = np.round(Vy.reshape(ymax-ymin,xmax-xmin)).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = ((srcx < w_src) & (srcx >= 0) & (srcy < h_src) & (srcy >= 0))
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking
        dst[yc[mask], xc[mask]] = src[srcy[mask], srcx[mask]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U)
        Vx, Vy, _ = V / V[2,:]
        Vx = np.round(Vx.reshape(ymax-ymin, xmax-xmin)).astype(int)
        Vy = np.round(Vy.reshape(ymax-ymin, xmax-xmin)).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = ((Vx < w_dst) & (Vx >= 0)) & ((Vy < h_dst) & (Vy >= 0))
        # TODO: 5.filter the valid coordinates using previous obtained mask
        Vx = Vx[mask]
        Vy = Vy[mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[Vy, Vx, :] = src[mask]

    return dst 
