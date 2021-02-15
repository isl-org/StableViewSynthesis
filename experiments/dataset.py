import numpy as np
import PIL
import gzip
import pickle
import logging
import time
from pathlib import Path
import sys

sys.path.append("../")
import co
import ext


def load(p, height=None, width=None):
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        im = PIL.Image.open(p)
        im = np.array(im)
        if (
            height is not None
            and width is not None
            and (im.shape[0] != height or im.shape[1] != width)
        ):
            raise Exception("invalid size of image")
        im = (im.astype(np.float32) / 255) * 2 - 1
        im = im.transpose(2, 0, 1)
        return im
    else:
        raise Exception("invalid suffix")


class Dataset(co.mytorch.BaseDataset):
    def __init__(
        self,
        *,
        name,
        tgt_im_paths,
        tgt_dm_paths,
        tgt_Ks,
        tgt_Rs,
        tgt_ts,
        tgt_counts,
        src_im_paths,
        src_dm_paths,
        src_Ks,
        src_Rs,
        src_ts,
        tgt_ma_paths=None,
        im_size=None,
        pad_width=None,
        n_nbs=5,
        src_mode="image",
        nbs_mode="sample",
        n_max_sources=-1,
        rank_mode="pointdir",
        invalid_depth=1e9,
        m2l_mode="backward",
        point_aux_data=["dirs"],
        point_edges_mode="penone",
        mode="train",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.tgt_im_paths = tgt_im_paths
        self.tgt_dm_paths = tgt_dm_paths
        self.tgt_ma_paths = tgt_ma_paths
        self.tgt_Ks = tgt_Ks
        self.tgt_Rs = tgt_Rs
        self.tgt_ts = tgt_ts
        self.tgt_counts = tgt_counts

        self.src_im_paths = src_im_paths
        self.src_dm_paths = src_dm_paths
        self.src_Ks = src_Ks
        self.src_Rs = src_Rs
        self.src_ts = src_ts

        self.im_size = im_size
        self.pad_width = pad_width
        self.n_nbs = n_nbs
        self.src_mode = src_mode
        self.nbs_mode = nbs_mode
        self.n_max_sources = n_max_sources
        self.rank_mode = rank_mode
        self.invalid_depth = invalid_depth
        self.m2l_mode = m2l_mode
        self.point_aux_data = point_aux_data
        self.point_edges_mode = point_edges_mode
        self.mode = mode

        tmp = np.load(tgt_dm_paths[0])
        self.height, self.width = tmp.shape
        del tmp

        tgt_im_len = "None"
        tgt_shape_str = "None"
        if tgt_im_paths:
            tgt_im_len = str(len(tgt_im_paths))
            tgt_shape_str = str(self.load_pad(tgt_im_paths[0]).shape)
        logging.info(
            f"    #tgt_im_paths={tgt_im_len}, #tgt_counts={tgt_counts.shape}, tgt_im={tgt_shape_str}, tgt_dm={self.load_pad(tgt_dm_paths[0]).shape}, train={self.train}"
        )

    def pad(self, im):
        if self.im_size is not None:
            shape = [s for s in im.shape]
            shape[-2] = self.im_size[0]
            shape[-1] = self.im_size[1]
            im_p = np.zeros(shape, dtype=im.dtype)
            sh = min(im_p.shape[-2], im.shape[-2])
            sw = min(im_p.shape[-1], im.shape[-1])
            im_p[..., :sh, :sw] = im[..., :sh, :sw]
            im = im_p
        if self.pad_width is not None:
            h, w = im.shape[-2:]
            mh = h % self.pad_width
            ph = 0 if mh == 0 else self.pad_width - mh
            mw = w % self.pad_width
            pw = 0 if mw == 0 else self.pad_width - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im

    def load_pad(self, p):
        im = load(p)
        return self.pad(im)

    def get_item_tgt(self, idx, rng, nbs, ret_tgt_dm=False):
        ret = {}

        # tic = time.time()
        tgt_dm = load(self.tgt_dm_paths[idx])
        orig_height, orig_width = tgt_dm.shape
        tgt_dm = self.pad(tgt_dm)
        if self.invalid_depth > 0:
            tgt_dm[tgt_dm <= 0] = self.invalid_depth
        height, width = tgt_dm.shape
        if ret_tgt_dm:
            ret["tgt_dm"] = tgt_dm

        # Target images used for training [3 x height x width]
        if self.tgt_im_paths:
            # if self.train:
            #     ret["tgt"] = self.load_pad(self.tgt_im_paths[idx])
            # else:
            #     ret["tgt"] = load(self.tgt_im_paths[idx])
            ret["tgt"] = load(self.tgt_im_paths[idx])
        else:
            ret["tgt"] = np.zeros((3, orig_height, orig_width), dtype=np.float32)

        if self.tgt_ma_paths:
            tgt_ma = PIL.Image.open(self.tgt_ma_paths[idx])
            tgt_ma = tgt_ma.resize((orig_width, orig_height), PIL.Image.NEAREST)
            tgt_ma = np.array(tgt_ma)
            tgt_ma = (tgt_ma[..., 0] != 0) | (tgt_ma[..., 1] != 0) | (tgt_ma[..., 2] != 0)
            ret["tgt_ma"] = tgt_ma[None].astype(np.float32)

        src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
        if self.invalid_depth > 0:
            src_dms[src_dms <= 0] = self.invalid_depth
        src_dms = self.pad(src_dms)

        tgt_counts = np.array([self.tgt_counts[idx, ii] for ii in nbs])

        src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        src_ts = np.array([self.src_ts[ii] for ii in nbs])
        (
            src_pos,
            src_idx,
            src_dirs,
            tgt_dirs,
            point_key,
            tgt_idx,
        ) = ext.preprocess.map_source_points(
            tgt_dm,
            self.tgt_Ks[idx],
            self.tgt_Rs[idx],
            self.tgt_ts[idx],
            tgt_counts,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            bwd_depth_thresh=0.01,
            n_max_sources=self.n_max_sources,
            rank_mode=self.rank_mode,
        )

        if self.m2l_mode == "backward":
            # Position in src images where to sample features from
            # Pixel center is (x+.5, y+.5) [N]
            ret["m2l_src_pos"] = src_pos
            # Index to neighbour image where to sample from
            # zero based and continuous [N]
            ret["m2l_src_idx"] = src_idx
        elif self.m2l_mode == "forward":
            (
                m2l_prefix,
                m2l_tgt_idx,
                src_pos,
            ) = ext.preprocess.inverse_map_to_list(src_idx, src_pos, src_dms.shape[0])
            ret["m2l_prefix"] = m2l_prefix
            ret["m2l_tgt_idx"] = m2l_tgt_idx
            ret["m2l_src_pos"] = src_pos
        else:
            raise Exception("invalid m2l_mode")

        # Directions src - 3D and tgt - 3D [N, 3]
        if "dirs" in self.point_aux_data:
            ret["point_src_dirs"] = src_dirs
            ret["point_tgt_dirs"] = tgt_dirs
        elif "offdirs" in self.point_aux_data:
            (
                ret["point_src_dirs"],
                ret["point_tgt_dirs"],
            ) = ext.preprocess.o3d_cconv_data(point_key, src_dirs, tgt_dirs[tgt_idx], 6)
        # if not np.all(np.isfinite(dirs)):
        #     __import__("ipdb").set_trace()
        # Key that indicates to which 3D point the above information is related
        # Zero based and continuous (from 0 to M) [N]
        # as prefix sum
        ret["point_key"] = point_key.astype(np.int64)

        # point_edges defines a graph on the 3D points
        # if self.point_edges_mode == "fc_per_point":
        #     point_edges = ext.preprocess.keys_to_fc_edges(point_key)
        #     ret["point_edges"] = point_edges.astype(np.int64)
        if self.point_edges_mode == "penone":
            pass
        elif self.point_edges_mode.startswith("pe"):
            mode = self.point_edges_mode[len("pe") :]
            if mode not in ["sf", "st", "sfst", "fc", "fcst", "fcsf", "fcstsf"]:
                raise Exception("invalid point_edges")
            point_edges = ext.preprocess.point_edges(point_key, tgt_idx, mode)
            # if point_edges.shape[0] <= 0:
            #     raise Exception("invalid point_edges")
            ret["point_edges"] = point_edges.astype(np.int64)
        else:
            raise Exception("invalid point_edges_mode")

        ret["pixel_tgt_idx"] = tgt_idx
        # assert tgt_idx.shape[0] == np.unique(tgt_idx).shape[0]

        return ret

    def get_item_eval_src(self, idx, rng):
        ret = {}
        if self.src_mode in ["image", "image+index"]:
            ret["src_ims"] = self.load_pad(self.src_im_paths[idx])[None]
        if self.src_mode in ["index", "image+index"]:
            ret["src_ind"] = np.array([idx])
        return ret

    def get_item_eval_tgt(self, idx, rng):
        nbs = [ii for ii in range(len(self.src_im_paths))]
        return self.get_item_tgt(idx, rng, nbs, ret_tgt_dm=True)

    def get_item_train(self, idx, rng):
        count = self.tgt_counts[idx]
        if self.nbs_mode == "argmax":
            nbs = np.argsort(count)[::-1]
            nbs = nbs[: self.n_nbs]
        elif self.nbs_mode == "sample":
            nbs = rng.choice(count.shape[0], self.n_nbs, replace=False, p=count / count.sum())
        else:
            raise Exception("invalid nbs_mode")

        ret = self.get_item_tgt(idx, rng, nbs, ret_tgt_dm=False)
        if self.src_mode in ["image", "image+index"]:
            ret["src_ims"] = np.array([self.load_pad(self.src_im_paths[ii]) for ii in nbs])
        if self.src_mode in ["index", "image+index"]:
            ret["src_ind"] = np.array(nbs)
        return ret

    def base_len(self):
        if self.mode == "train":
            return len(self.tgt_dm_paths)
        elif self.mode == "eval_src":
            return len(self.src_dm_paths)
        elif self.mode == "eval_tgt":
            return len(self.tgt_dm_paths)
        else:
            print(self.mode)
            raise Exception("invalid mode")

    def base_getitem(self, idx, rng):
        if self.mode == "train":
            return self.get_item_train(idx, rng)
        elif self.mode == "eval_src":
            return self.get_item_eval_src(idx, rng)
        elif self.mode == "eval_tgt":
            return self.get_item_eval_tgt(idx, rng)
        else:
            print(self.mode)
            raise Exception("invalid mode")
