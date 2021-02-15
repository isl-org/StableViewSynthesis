import numpy as np
import matplotlib as mpl
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D
import os
import time


def setup_for_latex(font_size=28):
    # mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    mpl.rc("text", usetex=True)
    params = {
        "legend.fontsize": "large",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size * 0.75,
        "ytick.labelsize": font_size * 0.75,
        "axes.titlepad": 25,
    }
    plt.rcParams.update(params)


def save(path, remove_axis=False, dpi=300, fig=None):
    if fig is None:
        fig = plt.gcf()
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    if remove_axis:
        for ax in fig.axes:
            ax.axis("off")
            ax.margins(0, 0)
        fig.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        for ax in fig.axes:
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)


def non_annoying_pause(interval, focus_figure=False):
    # interval is in seconds
    # https://github.com/matplotlib/matplotlib/issues/11131
    backend = mpl.rcParams["backend"]
    if backend in _interactive_bk:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            if focus_figure:
                plt.show(block=False)
            canvas.start_event_loop(interval)
            return
    time.sleep(interval)


def remove_all_ticks(fig=None):
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


def tight_no_ticks(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    remove_all_ticks(fig=fig)


def tight_no_ticks_show(fig=None):
    tight_no_ticks()
    plt.show()


def maximize_window():
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


def image_colorcode(
    im_, cmap="viridis", vmin=None, vmax=None, nan_color=(1, 1, 1)
):
    cm = plt.get_cmap(cmap)
    im = im_.copy()
    if vmin is None:
        vmin = np.nanmin(im)
    if vmax is None:
        vmax = np.nanmax(im)
    mask = np.logical_not(np.isfinite(im))
    im[mask] = vmin
    im = (im.clip(vmin, vmax) - vmin) / (vmax - vmin)
    im = cm(im)
    im = im[..., :3]
    im[mask] = nan_color
    return im


def image_matrix(ims, bgval=0, x=1, y=1):
    n = len(ims)
    cols = x * int(np.ceil(np.sqrt(n / (x * y))))
    rows = y * int(np.ceil(np.sqrt(n / (x * y))))
    if (rows - 1) * cols >= n:
        rows -= 1
    if rows * (cols - 1) >= n:
        cols -= 1
    h = ims[0].shape[0]
    w = ims[0].shape[1]
    mat = np.full(
        (rows * h, cols * w, *ims[0].shape[2:]), bgval, dtype=ims[0].dtype
    )
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                mat[r * h : (r + 1) * h, c * w : (c + 1) * w] = ims[idx]
                idx += 1
    return mat


def image_cat2(ims, cmap="viridis", default=(0, 0, 0), max_ims_per_row=None):
    # ims ... 2d list of images
    if max_ims_per_row is not None:
        ims_ = []
        for im_row in ims:
            row = []
            for im in im_row:
                row.append(im)
                if len(row) == max_ims_per_row:
                    ims_.append(row)
                    row = []
            if len(row) > 0:
                ims_.append(row)
        ims = ims_

    rows, cols = 0, 0
    for row in ims:
        max_r, c = 0, 0
        for im in row:
            if im is None:
                continue
            h, w = im.shape[0], im.shape[1]
            max_r = max(max_r, h)
            c += w
        rows += max_r
        cols = max(cols, c)

    out = np.zeros((rows, cols, 3), dtype=im.dtype)
    out[..., :] = default
    offr = 0
    for row in ims:
        max_r, offc = 0, 0
        for im in row:
            if im is None:
                continue
            h, w = im.shape[0], im.shape[1]
            out[offr : offr + h, offc : offc + w] = im
            max_r = max(max_r, h)
            offc += w
        offr += max_r

    return out


def depthshow(depth, *args, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    d = depth.copy()
    d[d <= 0] = np.NaN
    ax.imshow(d, *args, **kwargs)


def normalshow(normals, *args, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    normals = 0.5 * normals + 0.5
    ax.imshow(normals, *args, **kwargs)


def create_ax_3d(fig=None):
    if fig is None:
        fig = plt.gcf()
    return fig.add_subplot(111, projection="3d")


def axis_equal_3d(ax=None):
    if ax is None:
        ax = plt.gca()
    extents = np.array(
        [getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"]
    )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def axis_label_3d(ax=None, x="x", y="y", z="z"):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)


def _cmap(name, cm_data, reverse=False):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, cm_data)
    if reverse:
        return cmap.reversed()
    else:
        return cmap


if __name__ == "__main__":
    cmaps = ["viridis", "plasma", "magma", "inferno", "cubehelix", "jet"]
    n = len(cmaps)

    plt.figure(figsize=(16, 16))
    for cidx, cmap in enumerate(cmaps):
        ax = plt.subplot(n, 1, 1 + cidx)
        ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
        ax.title.set_text(cmap)
    plt.tight_layout()

    plt.figure(figsize=(16, 16))
    for cidx, cmap in enumerate(cmaps):
        cm = plt.get_cmap(cmap)
        im = cm(np.linspace(0, 1, 256)[None, :])
        ax = plt.subplot(n, 1, 1 + cidx)
        # im[..., :3] = np.mean(im[...,:3], axis=2, keepdims=True)
        # ax.imshow(im, aspect='auto')
        im = np.mean(im[..., :3], axis=2)
        ax.imshow(im, aspect="auto", cmap="gray")
        ax.title.set_text(cmap)
    plt.tight_layout()

    plt.show()
