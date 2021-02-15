import numpy as np
from pathlib import Path
import time
import argparse
import multiprocessing
import itertools
import sys

from create_data_common import (
    render_depth_maps_mesh,
    load_depth_maps,
    imread,
    write_im_scaled,
    combine_counts,
)

sys.path.append("..")
import config
import co
import ext


def compute_and_write_count_pw(count_path, idx, dms, Ks, Rs, ts):
    print(f"compute pw count {count_path}")
    count = ext.preprocess.count_nbs(
        dms[idx],
        Ks[idx],
        Rs[idx],
        ts[idx],
        dms,
        Ks,
        Rs,
        ts,
        bwd_depth_thresh=0.1,
    )
    count[idx] = 0
    np.save(count_path, count)


def compute_and_write_count_tgt(
    count_path, tgt_dm_path, tgt_K, tgt_R, tgt_t, dms, Ks, Rs, ts
):
    tgt_dm = np.load(tgt_dm_path)
    print(f"compute count for {count_path}")
    count = ext.preprocess.count_nbs(
        tgt_dm, tgt_K, tgt_R, tgt_t, dms, Ks, Rs, ts, bwd_depth_thresh=0.1
    )
    np.save(count_path, count)


def run(dense_dir, scale, dm_write_vis=False):
    run_tic = time.time()
    dense_dir = Path(dense_dir)

    pw_dir = dense_dir / f"ibr3d_pw_{scale:.2f}"
    pw_dir.mkdir(parents=True, exist_ok=True)

    src_im_paths = []
    src_im_paths += sorted((dense_dir / "images").glob("src*.png"))
    src_im_paths += sorted((dense_dir / "images").glob("src*.jpg"))
    src_im_paths += sorted((dense_dir / "images").glob("src*.jpeg"))

    src_Ks, src_Rs, src_ts = co.colmap.load_cameras(
        dense_dir / "sparse", src_im_paths, scale
    )

    print(f"write src camera params to")
    np.save(pw_dir / "Ks.npy", src_Ks)
    np.save(pw_dir / "Rs.npy", src_Rs)
    np.save(pw_dir / "ts.npy", src_ts)

    print(f"write scaled src images if needed to {pw_dir}")
    write_im_scaled(src_im_paths, scale, pw_dir)

    im0 = imread(src_im_paths[0], scale)
    height, width = im0.shape[:2]

    print(f"render src depth maps if needed to {pw_dir}")
    mesh_path = dense_dir / "delaunay_photometric.ply"
    src_dm_paths = render_depth_maps_mesh(
        pw_dir,
        mesh_path,
        src_Ks,
        src_Rs,
        src_ts,
        height,
        width,
        write_vis=dm_write_vis,
    )
    src_dms = load_depth_maps(src_dm_paths)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        args = []
        for idx in range(len(src_im_paths)):
            count_path = pw_dir / f"count_{idx:08d}.npy"
            if not count_path.exists():
                print(f"add {count_path} to list to process")
                args.append((count_path, idx, src_dms, src_Ks, src_Rs, src_ts))
        p.starmap(compute_and_write_count_pw, args)

    print("combine counts src")
    combine_counts(pw_dir)

    tgt_dir = dense_dir / f"ibr3d_tgt_{scale:.2f}"
    tgt_dir.mkdir(parents=True, exist_ok=True)

    tgt_im_paths = []
    tgt_im_paths += sorted((dense_dir / "images").glob("tgt*.png"))
    tgt_im_paths += sorted((dense_dir / "images").glob("tgt*.jpg"))
    tgt_im_paths += sorted((dense_dir / "images").glob("tgt*.jpeg"))

    write_im_scaled(tgt_im_paths, scale, tgt_dir)

    tgt_Ks, tgt_Rs, tgt_ts = co.colmap.load_cameras(
        dense_dir / "sparse", tgt_im_paths, scale
    )

    print(f"write tgt camera params to")
    np.save(tgt_dir / "Ks.npy", tgt_Ks)
    np.save(tgt_dir / "Rs.npy", tgt_Rs)
    np.save(tgt_dir / "ts.npy", tgt_ts)

    print(f"render tgt depth maps if needed to {tgt_dir}")
    mesh_path = dense_dir / "delaunay_photometric.ply"
    tgt_dm_paths = render_depth_maps_mesh(
        tgt_dir,
        mesh_path,
        tgt_Ks,
        tgt_Rs,
        tgt_ts,
        height,
        width,
        write_vis=dm_write_vis,
    )

    # compute count for tgt
    print(f"compute tgt counts")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        args = []
        for tgt_idx, tgt_dm_path in enumerate(tgt_dm_paths):
            count_path = tgt_dir / f"count_{tgt_idx:08d}.npy"
            if not count_path.exists():
                args.append(
                    (
                        count_path,
                        tgt_dm_paths[tgt_idx],
                        tgt_Ks[tgt_idx],
                        tgt_Rs[tgt_idx],
                        tgt_ts[tgt_idx],
                        src_dms,
                        src_Ks,
                        src_Rs,
                        src_ts,
                    )
                )
        p.starmap(compute_and_write_count_tgt, args)

    print("combine counts")
    combine_counts(tgt_dir)

    print(f"took {time.time() - run_tic}[s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scales", nargs="+", type=float, default=[0.5])
    parser.add_argument(
        "-d", "--datasets", nargs="+", type=str, default=config.fvs_sets
    )
    args = parser.parse_args()

    for dset, scale in itertools.product(args.datasets, args.scales):
        dense_dir = config.fvs_root / dset / "dense"
        print(f"create_data_pw for {dense_dir}")
        run(dense_dir, scale, dm_write_vis=False)
