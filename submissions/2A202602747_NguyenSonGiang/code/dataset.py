"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509     # Table 1 của bài báo
CROP_PCT = 0.875         # val/test: Resize(img_size / 0.875) rồi CenterCrop(img_size); 224 -> 256 -> 224
AUGS = ("basic", "color", "vflip", "trivial", "randaug")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    File fold thực tế chỉ có cột `Filename, Label` (README ghi thêm `Species`, nhưng file trên GitHub
    của tác giả không có). Trả về ba DataFrame, không sửa, lọc hay chia lại.
    """
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, labels_csv: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20, lệch > 1 điểm % thì dừng)
    2. giao từng cặp tập theo Filename phải rỗng
    3. hợp ba tập đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    5. (nếu có `labels_csv`) mỗi Label ứng với đúng một Species; liệt kê ảnh có Label khác labels.csv
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(df["Filename"]) for k, df in splits.items()}

    for k, df in splits.items():
        assert len(names[k]) == len(df), f"{k}: có Filename trùng lặp trong cùng một tập"
        assert df["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: Label ngoài [0, {NUM_CLASSES - 1}]"

    n = {k: len(df) for k, df in splits.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(frac[k] - target) <= 0.01, f"{k}: tỉ lệ {frac[k]:.4f} lệch > 1 điểm % khỏi {target}"

    overlap = {f"{a}&{b}": len(names[a] & names[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in overlap.values()), f"Giao giữa các tập khác rỗng: {overlap}"

    union = len(names["train"] | names["val"] | names["test"])
    assert union == TOTAL_IMAGES, f"Hợp ba tập có {union} ảnh, kỳ vọng {TOTAL_IMAGES}"

    on_disk = {p.name for p in Path(images_dir).iterdir()}
    missing = {k: sorted(names[k] - on_disk) for k in splits}
    n_missing = sum(len(v) for v in missing.values())
    assert n_missing == 0, f"{n_missing} file trong CSV không có trong {images_dir}: " \
                           f"{ {k: v[:3] for k, v in missing.items() if v} }"

    per_class = pd.DataFrame({k: df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, df in splits.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(axis=1)

    label_mismatch = None
    if labels_csv is not None:
        ref = pd.read_csv(labels_csv)
        assert (ref.groupby("Label")["Species"].nunique() == 1).all(), "Một Label ứng với nhiều Species"
        merged = pd.concat(splits.values()).merge(ref[["Filename", "Label"]], on="Filename",
                                                  how="left", suffixes=("", "_ref"))
        bad = merged[merged["Label"] != merged["Label_ref"]]
        label_mismatch = bad[["Filename", "Label", "Label_ref"]].to_dict("records")
        # Không sửa CSV (S1): chỉ cảnh báo. Fold 0 có đúng 1 ảnh train lệch (20170714-110407-3.jpg:
        # mọi fold ghi 0 = Chinee Apple, labels.csv ghi 1 = Lantana); không ảnh hưởng val/test.
        if label_mismatch and verbose:
            print(f"CẢNH BÁO: {len(label_mismatch)} ảnh có Label khác labels.csv (giữ nhãn của fold): "
                  f"{label_mismatch}")

    report = {
        "n": n, "frac": {k: round(v, 4) for k, v in frac.items()}, "union": union,
        "overlap": overlap, "missing_files": n_missing, "label_mismatch": label_mismatch,
        "per_class": per_class, "imbalance_ratio": float(per_class["total"].max() / per_class["total"].min()),
    }
    if verbose:
        print(f"Số ảnh: {n}  (tỉ lệ {report['frac']}), hợp = {union}")
        print(f"Giao theo Filename: {overlap}; file thiếu trên đĩa: {n_missing}; "
              f"số ảnh nhãn lệch labels.csv: {len(label_mismatch or [])}")
        print(per_class.to_string())
        print(f"Tỉ lệ lớp lớn nhất / nhỏ nhất: {report['imbalance_ratio']:.2f}")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = CROP_PCT,
                     scale_min: float = 0.08):
    """Tạo transform.

    Train, theo `aug` (trục B của GUIDE.md mục 3):
      basic   : RandomResizedCrop(img_size, scale=(scale_min, 1)) + lật ngang
                (scale_min = 0.08 là mặc định của torchvision; tăng lên để giảm lệch tỉ lệ train/test)
      color   : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      vflip   : basic + lật dọc. Ảnh chụp từ trên xuống nên hướng không có nghĩa; bài báo xoay ±360°
      trivial : basic + TrivialAugmentWide
      randaug : basic + RandAugment(num_ops=2, magnitude=9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Val/test: Resize(round(img_size / crop_pct)) + CenterCrop(img_size). Với img_size=224 nghĩa là
    giữ ảnh gốc 256 rồi cắt giữa 224. Không có augmentation ngẫu nhiên. Đổi `crop_pct` hoặc
    `img_size` để dò độ phân giải kiểm tra (I04).
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose([T.Resize(round(img_size / crop_pct)), T.CenterCrop(img_size), *norm])

    if aug not in AUGS:
        raise ValueError(f"aug={aug!r} không hỗ trợ; chọn một trong {AUGS}")
    ops = [T.RandomResizedCrop(img_size, scale=(scale_min, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "vflip":
        ops.append(T.RandomVerticalFlip())
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose([*ops, *norm])


def load_byte_store(images_dir: str | Path) -> dict[str, bytes]:
    """Byte JPEG của mọi ảnh (~490 MB) trong một file pickle cạnh `images_dir`.

    Đọc 17.509 file nhỏ qua ổ mạng mất ~1-2 phút mỗi lần chạy; một file pickle đọc trong vài giây.
    Lần đầu tự tạo `<images_dir>_bytes.pkl`. Chỉ là bản sao byte nguyên vẹn, không đổi ảnh.
    """
    import pickle
    images_dir = Path(images_dir)
    path = images_dir.parent / f"{images_dir.name}_bytes.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    store = {p.name: p.read_bytes() for p in sorted(images_dir.glob("*.jpg"))}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return store


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    `cache`: False (đọc file từ đĩa), True (đọc trước byte JPEG của các ảnh trong df), hoặc một dict
    {filename: bytes} từ load_byte_store dùng chung cho mọi tập. Giải mã vẫn làm trong worker.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).to_numpy()
        if isinstance(cache, dict):
            self._bytes = [cache[f] for f in self.filenames]
        elif cache:
            self._bytes = [(self.images_dir / f).read_bytes() for f in self.filenames]
        else:
            self._bytes = None

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, i: int):
        from PIL import Image
        src = io.BytesIO(self._bytes[i]) if self._bytes is not None else self.images_dir / self.filenames[i]
        with Image.open(src) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed random/numpy trong mỗi worker từ seed torch mà DataLoader cấp (tái lập augmentation)."""
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def class_balanced_weights(labels) -> np.ndarray:
    """Trọng số mỗi mẫu = 1 / (số ảnh của lớp đó)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return 1.0 / counts[labels]


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache=False, dataset: DeepWeedsDataset | None = None,
                persistent: bool = True):
    """Tạo DataLoader.

    - train=True: shuffle, hoặc `sampler="balanced"` (WeightedRandomSampler, trọng số 1/số ảnh của
      lớp, có hoàn lại, số mẫu mỗi epoch = len(df)); drop_last=True để BatchNorm không gặp batch lẻ
    - train=False: không shuffle, giữ đúng thứ tự df để ghép logit với Filename
    - `seed` cố định thứ tự batch, sampler và seed của worker
    - `persistent`: giữ worker giữa các epoch (tắt khi tạo nhiều loader dùng một lần)
    - `dataset` (tuỳ chọn) dùng lại một DeepWeedsDataset đã cache, khi đó bỏ qua df/images_dir/transform
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = dataset if dataset is not None else DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator().manual_seed(seed)

    smp, shuffle = None, False
    if train:
        if sampler == "balanced":
            w = torch.as_tensor(class_balanced_weights(ds.labels), dtype=torch.double)
            smp = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        elif sampler is None:
            shuffle = True
        else:
            raise ValueError(f"sampler={sampler!r} không hỗ trợ; chọn None hoặc 'balanced'")
    elif sampler is not None:
        raise ValueError("Không dùng sampler khi đánh giá")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g, persistent_workers=persistent and num_workers > 0,
    )
