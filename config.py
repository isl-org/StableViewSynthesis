from pathlib import Path

train_device = "cuda:0"
eval_device = "cuda:0"

tat_train_sets = [
    "training/Barn",
    "training/Caterpillar",
    "training/Church",
    "training/Courthouse",
    "training/Ignatius",
    "training/Meetingroom",
    "intermediate/Family",
    "intermediate/Francis",
    "intermediate/Horse",
    "intermediate/Lighthouse",
    "intermediate/Panther",
    "advanced/Auditorium",
    "advanced/Ballroom",
    "advanced/Museum",
    "advanced/Temple",
    "advanced/Courtroom",
    "advanced/Palace",
]


tat_train_sets_small = [
    "training/Barn",
    "training/Caterpillar",
    "training/Church",
    "training/Courthouse",
    "training/Ignatius",
    "training/Meetingroom",
]

tat_eval_sets = [
    "training/Truck",
    "intermediate/M60",
    "intermediate/Playground",
    "intermediate/Train",
]

tat_val_sets = ["training/Ignatius", "intermediate/Horse"]
tat_train_wo_val_sets = [s for s in tat_train_sets if s not in tat_val_sets]

# fmt: off
tat_tracks = {}
tat_tracks['training/Truck'] = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
tat_tracks['intermediate/M60'] = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
tat_tracks['intermediate/Playground'] = [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
tat_tracks['intermediate/Train'] = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]

tat_tracks['training/Ignatius'] = [71, 132, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]
tat_tracks['intermediate/Horse'] = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

dtu_invalid_sets = [25, 26, 27, 54, 73, 78, 79, 80, 81]
dtu_all_sets = [idx for idx in range(1, 129) if idx not in dtu_invalid_sets]

dtu_mvsnet_val_sets = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
dtu_mvsnet_eval_sets = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
dtu_mvsnet_train_sets = [idx for idx in range(1, 129) if idx not in dtu_mvsnet_val_sets + dtu_mvsnet_eval_sets + dtu_invalid_sets]

dtu_my_eval_sets = [65, 106, 118]
dtu_my_train_sets = [idx for idx in range(1, 129) if idx not in dtu_my_eval_sets + dtu_invalid_sets]

dtu_interpolation_ind = [13, 14, 15, 23, 24, 25]
dtu_extrapolation_ind = [0, 4, 38, 48]
dtu_source_ind = [idx for idx in range(49) if idx not in dtu_interpolation_ind + dtu_extrapolation_ind]
# fmt: on

fvs_sets = ["bike", "flowers", "pirate", "playground", "sandbox", "soccertable"]


def get_feature_tmp_dir(experiments_root, experiment_name):
    out_dir = Path(experiments_root) / experiment_name
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


lpips_root = None
# TODO: change to valid roots; i.e. Path('...')
tat_root = None
fvs_root = None
