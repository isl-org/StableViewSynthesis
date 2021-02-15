import torch
import torch.nn as nn
import numpy as np
import sys
import logging
import PIL
from pathlib import Path

import dataset
import modules

sys.path.append("../")
import co
import ext
import config


class EvalCache(object):
    def __init__(self, paths, cache_size=20):
        self.paths = paths
        self.cache_size = cache_size

        self.cache = {}

    def _load(self, idx):
        return torch.load(self.paths[idx], map_location="cpu")

    def get_ind(self, prefix):
        # print(f"CACHE {len(self.cache)}")
        cur_cache_ind = np.array(list(self.cache.keys()), dtype=np.int64)
        needed_ind = np.argsort(prefix)[::-1]
        new_cache_ind = needed_ind[: self.cache_size]

        all_cache_ind = np.union1d(cur_cache_ind, new_cache_ind)
        if all_cache_ind.shape[0] > self.cache_size:
            diff_cache_ind = np.setdiff1d(all_cache_ind, new_cache_ind)
            rm_size = all_cache_ind.shape[0] - self.cache_size
            for rm_idx in diff_cache_ind[:rm_size]:
                del self.cache[rm_idx]
        # print(f"  CACHE after del {len(self.cache)}")

        new_add_cache_ind = np.setdiff1d(new_cache_ind, cur_cache_ind)
        for add_idx in new_add_cache_ind:
            self.cache[add_idx] = self._load(add_idx)
        # print(f"  CACHE after add {len(self.cache)}")

        return needed_ind

    def load(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            return self._load(idx)


class Worker(co.mytorch.Worker):
    def __init__(
        self,
        train_dsets,
        eval_dsets="",
        train_n_nbs=1,
        train_nbs_mode="argmax",
        train_scale=1,
        eval_scale=-1,
        train_loss="vgg",
        eval_n_max_sources=5,
        train_rank_mode="pointdir",
        eval_rank_mode="",
        n_train_iters=-65536,
        num_workers=6,
        **kwargs,
    ):
        super().__init__(
            n_train_iters=n_train_iters,
            num_workers=num_workers,
            train_device=config.train_device,
            eval_device=config.eval_device,
            **kwargs,
        )

        self.train_dsets = train_dsets
        self.eval_dsets = eval_dsets
        self.train_n_nbs = train_n_nbs
        self.train_src_mode = "image"
        self.train_nbs_mode = train_nbs_mode
        self.train_scale = train_scale
        self.eval_scale = train_scale if eval_scale <= 0 else eval_scale
        self.invalid_depth = 1e9
        # self.invalid_depth = -1
        # self.point_key_mode = "prefix"
        self.point_aux_data = ["dirs"]
        self.point_edges_mode = "penone"
        self.eval_n_max_sources = eval_n_max_sources
        self.train_rank_mode = train_rank_mode
        self.eval_rank_mode = (
            eval_rank_mode if len(eval_rank_mode) > 0 else train_rank_mode
        )

        if train_loss == "vgg":
            self.train_loss = modules.VGGPerceptualLoss()
        elif train_loss == "ssiml1":
            self.train_loss = modules.SSIML1Loss()
        elif train_loss == "l1":
            self.train_loss = modules.SSIML1Loss()
        else:
            raise Exception("invalid train loss")

        if config.lpips_root is None:
            self.eval_loss = nn.L1Loss()
        else:
            self.eval_loss = modules.LPIPS()

    def get_dataset(
        self,
        *,
        name,
        ibr_dir,
        scale,
        im_size,
        pad_width,
        n_nbs,
        nbs_mode,
        im_ext,
        tgt_ind=None,
        src_ind=None,
        n_max_sources=-1,
        rank_mode="pointdir",
        train=False,
    ):
        logging.info(f"  create dataset for {name}")
        im_paths = sorted(ibr_dir.glob(f"im_*{im_ext}"))
        dm_paths = sorted(ibr_dir.glob("dm_*.npy"))
        counts = np.load(ibr_dir / "counts.npy")
        Ks = np.load(ibr_dir / "Ks.npy")
        Rs = np.load(ibr_dir / "Rs.npy")
        ts = np.load(ibr_dir / "ts.npy")

        if tgt_ind is None or src_ind is None:
            tgt_ind = np.arange(len(im_paths))
            src_ind = np.arange(len(im_paths))

        counts = counts[tgt_ind]
        counts = counts[:, src_ind]

        kwargs = {
            "name": name,
            "tgt_im_paths": [im_paths[idx] for idx in tgt_ind],
            "tgt_dm_paths": [dm_paths[idx] for idx in tgt_ind],
            "tgt_Ks": Ks[tgt_ind],
            "tgt_Rs": Rs[tgt_ind],
            "tgt_ts": ts[tgt_ind],
            "tgt_counts": counts,
            "src_im_paths": [im_paths[idx] for idx in src_ind],
            "src_dm_paths": [dm_paths[idx] for idx in src_ind],
            "src_Ks": Ks[src_ind],
            "src_Rs": Rs[src_ind],
            "src_ts": ts[src_ind],
            "im_size": im_size,
            "pad_width": pad_width,
            "n_nbs": n_nbs,
            "nbs_mode": nbs_mode,
            "n_max_sources": n_max_sources,
            "rank_mode": rank_mode,
            "invalid_depth": self.invalid_depth,
            "point_aux_data": self.point_aux_data,
            "point_edges_mode": self.point_edges_mode,
            "train": train,
        }
        dset = dataset.Dataset(**kwargs)
        return dset

    def get_train_set_tat(self, dset, mode, track_ind=None):
        # if self.train_scale == 0.25:
        #     im_size = (288, 512)
        # elif self.train_scale == 0.5:
        #     im_size = (288 * 2, 512 * 2)
        # else:
        #     raise Exception("invalid scale for tat")
        if self.train_batch_size != 1:
            raise Exception("invalid batch size for tat train set")
        im_ext = ".jpg"
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.train_scale:.2f}"
        im_paths = sorted(ibr_dir.glob(f"im_*{im_ext}"))
        if mode == "all":
            tgt_ind = None
            src_ind = None
        elif mode == "subseq":
            tgt_ind, src_ind = [], []
            for idx, im_path in enumerate(im_paths):
                if idx not in track_ind:
                    tgt_ind.append(idx)
                    src_ind.append(idx)
        else:
            raise Exception("invalid mode for get_train_set_tat")
        dset = self.get_dataset(
            name=f'tat_{mode}_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            scale=self.train_scale,
            # im_size=im_size,
            # pad_width=None,
            im_size=None,
            pad_width=32,
            n_nbs=self.train_n_nbs,
            nbs_mode=self.train_nbs_mode,
            rank_mode=self.train_rank_mode,
            im_ext=im_ext,
            tgt_ind=tgt_ind,
            src_ind=src_ind,
            train=True,
        )
        return dset

    def get_train_set(self):
        logging.info("Create train datasets")
        dsets = co.mytorch.MultiDataset(name="train")
        if "tat" in self.train_dsets:
            for dset in config.tat_train_sets:
                dsets.append(self.get_train_set_tat(dset, "all"))
        if "tat-wo-val" in self.train_dsets:
            for dset in config.tat_train_wo_val_sets:
                dsets.append(self.get_train_set_tat(dset, "all"))
        for dset in dsets.datasets:
            dset.src_mode = self.train_src_mode
        return dsets

    def get_eval_set_tat(self, dset, mode, track_ind=None):
        # if self.eval_scale == 0.25:
        #     im_size = (288, 512)
        # elif self.eval_scale == 0.5:
        #     im_size = (288 * 2, 512 * 2)
        # else:
        #     raise Exception("invalid scale for tat")
        if self.eval_batch_size != 1:
            raise Exception("invalid batch size for tat eval set")
        im_ext = ".jpg"
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.eval_scale:.2f}"
        im_paths = sorted(ibr_dir.glob(f"im_*{im_ext}"))
        if mode == "all":
            tgt_ind = None
            src_ind = None
        elif mode == "subseq":
            tgt_ind, src_ind = [], []
            for idx, im_path in enumerate(im_paths):
                if idx in track_ind:
                    tgt_ind.append(idx)
                else:
                    src_ind.append(idx)
        else:
            raise Exception("invalid mode for get_eval_set_tat")
        dset = self.get_dataset(
            name=f'tat_{mode}_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            scale=self.eval_scale,
            # im_size=im_size,
            # pad_width=None,
            im_size=None,
            pad_width=32,
            n_nbs=-1,
            nbs_mode="sample",
            im_ext=im_ext,
            n_max_sources=self.eval_n_max_sources,
            rank_mode=self.eval_rank_mode,
            tgt_ind=tgt_ind,
            src_ind=src_ind,
            train=False,
        )
        return dset

    def get_eval_set_trk(self, name, trk_dir, pw_dir):
        logging.info(f"  create dataset for {name}")
        trk_dir = Path(trk_dir)
        pw_dir = Path(pw_dir)
        src_im_paths = sorted(pw_dir.glob("im_*.jpg"))
        src_dm_paths = sorted(pw_dir.glob("dm_*.npy"))
        tgt_im_paths = sorted(trk_dir.glob("im_*.jpg"))
        tgt_im_paths += sorted(trk_dir.glob("im_*.jpeg"))
        tgt_im_paths += sorted(trk_dir.glob("im_*.png"))
        if len(tgt_im_paths) == 0:
            tgt_im_paths = None
        tgt_dm_paths = sorted(trk_dir.glob("dm_*.npy"))
        counts = np.load(trk_dir / "counts.npy")
        src_Ks = np.load(pw_dir / "Ks.npy")
        src_Rs = np.load(pw_dir / "Rs.npy")
        src_ts = np.load(pw_dir / "ts.npy")
        tgt_Ks = np.load(trk_dir / "Ks.npy")
        tgt_Rs = np.load(trk_dir / "Rs.npy")
        tgt_ts = np.load(trk_dir / "ts.npy")

        kwargs = {
            "name": name,
            "tgt_im_paths": tgt_im_paths,
            "tgt_dm_paths": tgt_dm_paths,
            "tgt_Ks": tgt_Ks,
            "tgt_Rs": tgt_Rs,
            "tgt_ts": tgt_ts,
            "tgt_counts": counts,
            "src_im_paths": src_im_paths,
            "src_dm_paths": src_dm_paths,
            "src_Ks": src_Ks,
            "src_Rs": src_Rs,
            "src_ts": src_ts,
            "im_size": None,
            "pad_width": 32,
            "n_nbs": -1,
            "nbs_mode": "sample",
            "n_max_sources": self.eval_n_max_sources,
            "rank_mode": self.eval_rank_mode,
            "invalid_depth": self.invalid_depth,
            "point_aux_data": self.point_aux_data,
            "point_edges_mode": self.point_edges_mode,
            "train": False,
        }
        dset = dataset.Dataset(**kwargs)
        return dset

    def get_eval_sets(self):
        logging.info("Create eval datasets")
        eval_sets = []
        if "tat" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                eval_sets.append(self.get_eval_set_tat(dset, "all"))
        if "tat-subseq" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                eval_sets.append(
                    self.get_eval_set_tat(
                        dset, "subseq", config.tat_tracks[dset]
                    )
                )
        if "tat-val-subseq" in self.eval_dsets:
            for dset in config.tat_val_sets:
                eval_sets.append(
                    self.get_eval_set_tat(
                        dset, "subseq", config.tat_tracks[dset]
                    )
                )
        for dset in self.eval_dsets:
            if dset.startswith("tat-scene-subseq-"):
                dset = dset[len("tat-scene-subseq-") :]
                eval_sets.append(
                    self.get_eval_set_tat(
                        dset, "subseq", config.tat_tracks[dset]
                    )
                )
            elif dset.startswith("tat-scene-"):
                dset = dset[len("tat-scene-") :]
                eval_sets.append(self.get_eval_set_tat(dset, "all"))
        for set_name in self.eval_dsets:
            if set_name.startswith("tat-track-"):
                set_name = set_name[len("tat-track-") :]
                set_name_us = set_name.replace("/", "_")
                eval_sets.append(
                    self.get_eval_set_trk(
                        f"tat-track-{set_name_us}",
                        config.tat_root
                        / f"{set_name}/dense/ibr3d_trk2_{self.eval_scale:0.2f}",
                        config.tat_root
                        / f"{set_name}/dense/ibr3d_pw_{self.eval_scale:0.2f}",
                    )
                )
        if "fvs" in self.eval_dsets:
            for set_name in config.fvs_sets:
                eval_sets.append(
                    self.get_eval_set_trk(
                        f"fvs_{set_name}",
                        config.fvs_root
                        / f"{set_name}/dense/ibr3d_tgt_{self.eval_scale:0.2f}",
                        config.fvs_root
                        / f"{set_name}/dense/ibr3d_pw_{self.eval_scale:0.2f}",
                    )
                )
        for dset in self.eval_dsets:
            if dset.startswith("fvs-scene-"):
                set_name = set_name[len("fvs-scene-") :]
                eval_sets.append(
                    self.get_eval_set_trk(
                        f"fvs_{set_name}",
                        config.fvs_root
                        / f"{set_name}/dense/ibr3d_tgt_{self.eval_scale:0.2f}",
                        config.fvs_root
                        / f"{set_name}/dense/ibr3d_pw_{self.eval_scale:0.2f}",
                    )
                )
        for dset in eval_sets:
            dset.logging_rate = 1
            dset.src_mode = self.train_src_mode
        return eval_sets

    def collate_fn(self, batch):
        def collate_cat(batch, k):
            return torch.cat([torch.from_numpy(b[k]) for b in batch])

        def collate_entangle_batch_idx(batch, k):
            batch_size = len(batch)
            for bidx, b in enumerate(batch):
                b[k] = b[k] * batch_size + bidx
            return collate_cat(batch, k)

        def collate_entanlge_height_width(batch, k, height, width):
            for bidx, b in enumerate(batch):
                b[k] = bidx * height * width + b[k]
            return collate_cat(batch, k)

        def collate_continue(batch, k):
            offset = 0
            for bidx, b in enumerate(batch):
                b[k] = b[k] + offset
                offset = b[k][-1] + 1
            ret = collate_cat(batch, k)
            return ret

        def collate_prefix(batch, k):
            offset = 0
            for bidx, b in enumerate(batch):
                if bidx > 0:
                    b[k] = b[k][1:]
                b[k] = b[k] + offset
                offset = b[k][-1]
            ret = collate_cat(batch, k)
            return ret

        def collate_edges(batch, k, key_k):
            prefix = [0] + [b[key_k].shape[0] for b in batch]
            prefix = np.cumsum(prefix)
            for b, p in zip(batch, prefix):
                b[k] = b[k] + p
            ret = collate_cat(batch, k)
            return ret.transpose(1, 0)

        ret = {}
        keys = list(batch[0].keys())
        for k in keys:
            if k in ["tgt", "src_ims", "src_ind", "tgt_dm", "tgt_ma"]:
                ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch])
            # TODO: does only work for batch size = 1
            elif k in [
                "point_src_edge_bins",
                "point_src_edge_weights",
                "point_tgt_edge_bins",
                "point_tgt_edge_weights",
            ]:
                ret[k] = torch.stack([torch.from_numpy(b[k]) for b in batch])
            elif k in [
                "m2l_src_pos",
                "point_dirs",
                "point_src_dirs",
                "point_tgt_dirs",
                "pixel_nb_dists",
                "pixel_nb_weights",
                "tgt_px",
            ]:
                ret[k] = collate_cat(batch, k)
            elif k in ["m2l_src_idx"]:
                ret[k] = collate_entangle_batch_idx(batch, k)
            elif k in ["pixel_tgt_idx"]:
                height, width = batch[0]["tgt"].shape[-2:]
                ret[k] = collate_entanlge_height_width(batch, k, height, width)
            elif k in [
                "pixel_nb_key",
                "m2l_tgt_idx",  # TODO: m2l_tgt_idx wrong
            ]:
                ret[k] = collate_continue(batch, k)
            elif k in [
                "point_key",
                "pixel_tgt_key",
                "m2l_prefix",  # m2l_prefix is wrong, should be added
            ]:
                ret[k] = collate_prefix(batch, k)
            # TODO: does only work for batch size = 1
            elif k in ["point_edges", 'point_tgt_edges', 'point_src_edges']:
                ret[k] = collate_edges(batch, k, "point_key")
            else:
                raise Exception(f"invalid k (={k}) in batch collate")
        return ret

    def get_train_data_loader(self, dset, iter):
        return torch.utils.data.DataLoader(
            co.mytorch.TrainDataset(
                dset, self.n_train_iters, self.train_batch_size, iter
            ),
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def get_eval_data_loader(self, dset):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def copy_data(self, data, device, train):
        self.data = {}
        for k, v in data.items():
            self.data[k] = v.to(device).requires_grad_(requires_grad=False)

    def net_forward(self, net, train, iter):
        return net(**self.data)

    def loss_forward(self, output, train, iter):
        errs = {}
        est = output["out"]
        tgt = self.data["tgt"]

        # fix size
        if est.shape[-1] > tgt.shape[-1]:
            est = est[..., : tgt.shape[-1]]
        if tgt.shape[-1] > est.shape[-1]:
            tgt = tgt[..., : est.shape[-1]]
        if est.shape[-2] > tgt.shape[-2]:
            est = est[..., : tgt.shape[-2], :]
        if tgt.shape[-2] > est.shape[-2]:
            tgt = tgt[..., : est.shape[-2], :]

        if "tgt_ma" in self.data:
            self.data["tgt_ma"] = self.data["tgt_ma"].to(est.device)
            est = self.data["tgt_ma"] * est
            tgt = self.data["tgt_ma"] * tgt

        if train:
            for lidx, loss in enumerate(self.train_loss(est, tgt)):
                errs[f"rgb{lidx}"] = loss
        else:
            errs["rgb"] = self.eval_loss(est, tgt)

        return errs

    def eval_set(self, iter, net, eval_set_idx, eval_set, epoch="x"):
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.my_eval_set(iter, net, eval_set_idx, eval_set, epoch)
            self.db_logger.commit()

    def my_eval_set(self, iter, net, eval_set_idx, eval_set, epoch="x"):
        logging.info("-" * 80)
        co.mytorch.log_datetime()
        logging.info("Eval iter %d" % iter)

        net = net.to(self.eval_device)
        net.eval()

        self.stopwatch.reset()
        self.stopwatch.start("total")

        # encode all source images
        if eval_set.src_mode in ["image", "image+index"]:
            self.stopwatch.start("encode_srcs")
            logging.info("  preprocess all source images")
            eval_set.mode = "eval_src"
            eval_loader = self.get_eval_data_loader(eval_set)
            feature_paths = []
            feat_tmp_dir = config.get_feature_tmp_dir(
                self.experiments_root,
                f'tmp_srcfeat_{self.experiment_name}_{eval_set.name.replace("/", "_")}',
            )
            logging.info(f"    feat tmp dir: {feat_tmp_dir}")
            for batch_idx, data in enumerate(eval_loader):
                self.copy_data(data, device=self.eval_device, train=False)
                src_ims, src_ind = None, None
                if "src_ims" in self.data:
                    src_ims = self.data["src_ims"]
                if "src_ind" in self.data:
                    src_ind = self.data["src_ind"]
                enc = net.enc_forward(src_ims, src_ind)
                feature_path = feat_tmp_dir / f"tmp_feature_{batch_idx:04}.pt"
                torch.save(enc[0, 0], str(feature_path))
                feature_paths.append(feature_path)
                feature_channels = enc.shape[2]
                del enc
            if "cuda" in self.eval_device:
                torch.cuda.synchronize()
            del eval_loader
            self.stopwatch.stop("encode_srcs")
            feat_map_cache = EvalCache(feature_paths, cache_size=200)
        elif eval_set.src_mode == "index":
            feature_channels = net.params.shape[1]
        else:
            print(eval_set.src_mode)
            raise Exception("invalid src_mode in eval_set")

        # eval targets
        logging.info("  create target images")
        self.stopwatch.start("callback")
        self.callback_eval_start(
            iter=iter, net=net, set_idx=eval_set_idx, eval_set=eval_set
        )
        self.stopwatch.stop("callback")

        eval_set.mode = "eval_tgt"
        eval_set.m2l_mode = "forward"
        eval_loader = self.get_eval_data_loader(eval_set)

        eta = co.utils.ETA(length=len(eval_loader))
        mean_loss = co.utils.CumulativeMovingAverage()
        self.stopwatch.start("data")
        for batch_idx, data in enumerate(eval_loader):
            self.copy_data(data, device="cpu", train=False)
            # set dimensions
            self.stopwatch.stop("data")

            self.stopwatch.start("extract_features")
            self.stopwatch.start("extract_features_init")
            m2l_prefix = self.data["m2l_prefix"].numpy()
            m2l_prefix_cum = np.hstack(
                (np.zeros((1,), dtype=m2l_prefix.dtype), np.cumsum(m2l_prefix))
            )
            output = torch.empty((m2l_prefix_cum[-1], feature_channels))
            self.stopwatch.stop("extract_features_init")
            if eval_set.src_mode in ["image", "image+index"]:
                for vidx in feat_map_cache.get_ind(m2l_prefix):
                    if m2l_prefix[vidx] == 0:
                        continue

                    self.stopwatch.start("extract_features_load")
                    feature = feat_map_cache.load(vidx)
                    self.stopwatch.stop("extract_features_load")

                    self.stopwatch.start("extract_features_map_to_list")
                    m2l_from = m2l_prefix_cum[vidx]
                    m2l_to = m2l_prefix_cum[vidx + 1]
                    m2l_tgt_idx = self.data["m2l_tgt_idx"][m2l_from:m2l_to]
                    m2l_src_pos = self.data["m2l_src_pos"][m2l_from:m2l_to]
                    # print(vidx, self.data["m2l_tgt_idx"].shape, m2l_from, m2l_to)
                    # print("  ", m2l_tgt_idx.min(), m2l_tgt_idx.max())
                    ext.mytorch.map_to_list_bl_seq(
                        feature, m2l_tgt_idx, m2l_src_pos, output
                    )
                    self.stopwatch.stop("extract_features_map_to_list")
            elif eval_set.src_mode == "index":
                for vidx in range(m2l_prefix.shape[0]):
                    if m2l_prefix[vidx] == 0:
                        continue

                    feature = net.params[vidx]

                    self.stopwatch.start("extract_features_map_to_list")
                    m2l_from = m2l_prefix_cum[vidx]
                    m2l_to = m2l_prefix_cum[vidx + 1]
                    m2l_tgt_idx = self.data["m2l_tgt_idx"][m2l_from:m2l_to]
                    m2l_src_pos = self.data["m2l_src_pos"][m2l_from:m2l_to]
                    ext.mytorch.map_to_list_bl_seq(
                        feature, m2l_tgt_idx, m2l_src_pos, output
                    )
                    self.stopwatch.stop("extract_features_map_to_list")
            else:
                raise Exception("invalid src_mode in eval_set")
            self.stopwatch.stop("extract_features")

            self.stopwatch.start("ref_forward")
            output = output.to(self.eval_device)
            for k, v in self.data.items():
                if k.startswith(("point_", "pixel_")):
                    self.data[k] = v.to(output.device)
            net.ref_net.to(output.device)
            height, width = feature.shape[-2:]
            output = net.ref_net_forward(
                output, self.eval_batch_size, height, width, **self.data
            )
            if "cuda" in self.eval_device:
                torch.cuda.synchronize()
            self.stopwatch.stop("ref_forward")

            self.stopwatch.start("loss")
            self.data["tgt"] = self.data["tgt"].to(output["out"].device)
            errs = self.loss_forward(output, train=False, iter=iter)
            err_items = {}
            for k in errs.keys():
                if torch.is_tensor(errs[k]):
                    err_items[k] = errs[k].item()
                else:
                    err_items[k] = [v.item() for v in errs[k]]
            del errs
            mean_loss.append(err_items)
            self.stopwatch.stop("loss")

            eta.update(batch_idx)
            if batch_idx % eval_set.logging_rate == 0:
                err_str = self.format_err_str(err_items)
                logging.info(
                    f"eval {epoch}/{iter}: {batch_idx+1}/{len(eval_loader)}: loss={err_str} ({np.sum(mean_loss.vals_list()):0.4f}) | {eta.get_str(percentage=True, elapsed=True, remaining=True)}"
                )

            self.stopwatch.start("callback")
            self.callback_eval_add(
                iter=iter,
                net=net,
                set_idx=eval_set_idx,
                eval_set=eval_set,
                batch_idx=batch_idx,
                n_batches=len(eval_loader),
                output=output,
            )
            self.stopwatch.stop("callback")

            self.free_copied_data()

            self.stopwatch.start("data")
        self.stopwatch.stop("total")

        if eval_set.src_mode == "image":
            for feature_path in feature_paths:
                if feature_path.exists():
                    feature_path.unlink()
            feat_tmp_dir.rmdir()

        self.stopwatch.start("callback")
        self.callback_eval_stop(
            iter=iter,
            net=net,
            set_idx=eval_set_idx,
            eval_set=eval_set,
            mean_loss=mean_loss.vals,
        )
        self.stopwatch.stop("callback")

        logging.info("timings: %s" % self.stopwatch)

        err_str = self.format_err_str(mean_loss.vals)
        logging.info(f"avg eval_loss={err_str}")

    def callback_eval_start(self, **kwargs):
        self.metric = None

    def im_to2np(self, im):
        im = im.detach().to("cpu").numpy()
        im = (np.clip(im, -1, 1) + 1) / 2
        im = im.transpose(0, 2, 3, 1)
        return im

    def callback_eval_add(self, **kwargs):
        output = kwargs["output"]
        batch_idx = kwargs["batch_idx"]
        eval_set = kwargs["eval_set"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}_nms{self.eval_n_max_sources}{self.eval_rank_mode}"

        ta = self.im_to2np(self.data["tgt"])
        es = self.im_to2np(output["out"])

        # fix size
        if es.shape[1] > ta.shape[1]:
            es = es[:, : ta.shape[1]]
        if ta.shape[1] > es.shape[1]:
            ta = ta[:, : es.shape[1]]
        if es.shape[2] > ta.shape[2]:
            es = es[..., : ta.shape[2], :]
        if ta.shape[2] > es.shape[2]:
            ta = ta[..., : es.shape[2], :]

        if "tgt_ma" in self.data:
            tgt_ma = self.data["tgt_ma"].detach().to("cpu").numpy()
            tgt_ma = tgt_ma.transpose(0, 2, 3, 1)
            es = tgt_ma * es
            ta = tgt_ma * ta

        # record metrics
        if self.metric is None:
            self.metric = {}
            self.metric["rgb"] = co.metric.MultipleMetric(
                metrics=[
                    co.metric.DistanceMetric(p=1, vec_length=3),
                    co.metric.PSNRMetric(),
                    co.metric.SSIMMetric(),
                ]
            )

        self.metric["rgb"].add(es, ta)

        vis_ind = np.arange(len(eval_set))
        out_dir = self.exp_out_root / eval_set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in range(ta.shape[0]):
            bidx = batch_idx * ta.shape[0] + b
            if bidx not in vis_ind:
                continue

            out_im = (255 * es[b]).astype(np.uint8)
            out_path = out_dir / f"s{bidx:04d}.png"
            PIL.Image.fromarray(out_im).save(out_path)

    def callback_eval_stop(self, **kwargs):
        eval_set = kwargs["eval_set"]
        iter = kwargs["iter"]
        mean_loss = kwargs["mean_loss"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"
        method_suffix = f"nms{self.eval_n_max_sources}{self.eval_rank_mode}"
        for key in self.metric:
            self.metric_add_eval(
                iter=iter,
                dataset=eval_set_name,
                metric=f"loss_{key}",
                value=sum(np.asarray(mean_loss[key]).ravel()),
                method_suffix=method_suffix,
            )
            metric = self.metric[key]
            logging.info(f"\n{key}\n{metric}")
            for k, v in metric.items():
                self.metric_add_eval(
                    iter=iter,
                    dataset=eval_set_name,
                    metric=str(k),
                    value=v,
                    method_suffix=method_suffix,
                )


if __name__ == "__main__":
    parser = co.mytorch.get_parser()
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument(
        "--train-dsets", nargs="+", type=str, default=["tat-wo-val"]
    )
    parser.add_argument(
        "--eval-dsets",
        nargs="+",
        type=str,
        default=["tat-scene-subseq-intermediate/Horse"],
    )
    parser.add_argument("--train-n-nbs", type=int, default=3)
    parser.add_argument("--train-scale", type=float, default=0.25)
    parser.add_argument("--eval-scale", type=float, default=0.5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--train-loss", type=str, default="vgg")
    parser.add_argument("--eval-n-max-sources", type=int, default=5)
    parser.add_argument("--train-rank-mode", type=str, default="pointdir")
    parser.add_argument("--eval-rank-mode", type=str, default="")
    parser.add_argument("--log-debug", type=str, nargs="*", default=[])
    parser.add_argument("--frequency", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--init-net-path", type=str, default="")
    parser.add_argument("--n-train-iters", type=int, default=-65536)
    args = parser.parse_args()

    experiment_name = "+".join(
        [dset.replace("/", "_") for dset in args.train_dsets]
    )
    experiment_name = f"{experiment_name}_bs{args.train_batch_size}_nbs{args.train_n_nbs}_r{args.train_rank_mode}_s{args.train_scale}_{args.net}_{args.train_loss}"

    worker = Worker(
        experiments_root=args.experiments_root,
        experiment_name=experiment_name,
        train_dsets=args.train_dsets,
        eval_dsets=args.eval_dsets,
        train_n_nbs=args.train_n_nbs,
        train_rank_mode=args.train_rank_mode,
        train_scale=args.train_scale,
        eval_scale=args.eval_scale,
        train_loss=args.train_loss,
        eval_n_max_sources=args.eval_n_max_sources,
        eval_rank_mode=args.eval_rank_mode,
        n_train_iters=args.n_train_iters,
    )
    worker.log_debug = args.log_debug
    worker.save_frequency = co.mytorch.Frequency(minutes=args.frequency)
    worker.eval_frequency = co.mytorch.Frequency(minutes=args.frequency)
    worker.train_batch_size = args.train_batch_size
    worker.eval_batch_size = args.eval_batch_size
    if args.eval_batch_size > 1:
        raise Exception("not supported atm")
    worker.train_batch_acc_steps = 1

    worker_objects = co.mytorch.WorkerObjects(
        optim_f=lambda net: torch.optim.Adam(
            net.parameters(), lr=args.learning_rate
        )
    )

    if args.net.startswith(("param_", "paramonly_")):
        worker.train_src_mode = "index"
        _, enc_net, ref_net = args.net.split("_")
        worker_objects.net_f = lambda: modules.get_param_net(
            worker=worker,
            enc_net=enc_net,
            ref_net=ref_net,
            init_net_path=args.init_net_path,
        )
        if args.net.startswith("paramonly_"):
            worker_objects.optim_f = lambda net: torch.optim.Adam(
                [net.params], lr=args.learning_rate
            )
    elif args.net.startswith(("image_", "imageonly_")):
        worker.train_src_mode = "image+index"
        _, enc_net, ref_net = args.net.split("_")
        worker_objects.net_f = lambda: modules.get_image_net(
            worker=worker,
            enc_net=enc_net,
            ref_net=ref_net,
            init_net_path=args.init_net_path,
        )
        if args.net.startswith("imageonly_"):
            worker_objects.optim_f = lambda net: torch.optim.Adam(
                [net.params], lr=args.learning_rate
            )
    elif args.net.startswith(("globalso_", "globalsoonly_")):
        worker.train_src_mode = "image+index"
        _, enc_net, ref_net = args.net.split("_")
        worker_objects.net_f = lambda: modules.get_globalso_net(
            worker=worker,
            enc_net=enc_net,
            ref_net=ref_net,
            init_net_path=args.init_net_path,
        )
        if args.net.startswith("globalsoonly_"):
            worker_objects.optim_f = lambda net: torch.optim.Adam(
                [net.scale_params, net.offset_params], lr=args.learning_rate
            )
    elif args.net.startswith(
        ("weight_", "weightonly_", "weightg_", "weightgonly_")
    ):
        worker.train_src_mode = "image+index"
        pixelwise = args.net.startswith(("weight_", "weightonly_"))
        _, enc_net, ref_net = args.net.split("_")
        worker_objects.net_f = lambda: modules.get_weight_net(
            worker=worker,
            enc_net=enc_net,
            ref_net=ref_net,
            init_net_path=args.init_net_path,
            pixelwise=pixelwise,
        )
        if args.net.startswith(("weightonly_", "weightgonly_")):
            worker_objects.optim_f = lambda net: torch.optim.Adam(
                [net.params], lr=args.learning_rate
            )
    else:
        enc_net, ref_net = args.net.split("_")
        worker_objects.net_f = lambda: modules.get_net(
            enc_net=enc_net, ref_net=ref_net, init_net_path=args.init_net_path
        )

    point_edges_mode, point_aux_data = ref_net.split(".")[:2]
    worker.point_edges_mode = point_edges_mode
    worker.point_aux_data = point_aux_data.split("+")

    worker.do(args, worker_objects)
