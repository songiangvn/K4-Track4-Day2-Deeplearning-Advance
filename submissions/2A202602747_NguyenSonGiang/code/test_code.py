"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy từ thư mục code/:

    python -m unittest test_code -v

Không cần GPU và không cần tải trọng số (model tạo với pretrained=False).
"""
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import benchmark as B
import inference as I
import losses as L
import model as M
import train as T


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)
        self.ce = F.cross_entropy(self.logits, self.y)

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.logits, self.y)
        self.assertLess(abs(fl.item() - self.ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(gamma=2.0)(self.logits, self.y).item(), self.ce.item())

    def test_focal_alpha_matches_weighted_terms(self):
        alpha = torch.linspace(0.5, 2.0, 9)
        fl = L.FocalLoss(gamma=0.0, alpha=alpha)(self.logits, self.y)
        ref = (F.cross_entropy(self.logits, self.y, reduction="none") * alpha[self.y]).mean()
        self.assertLess(abs(fl.item() - ref.item()), 1e-6)

    def test_label_smoothing(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.logits, self.y).item() - self.ce.item()), 1e-6)
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertLess(abs(L.LabelSmoothingCE(0.1)(self.logits, self.y).item() - ref.item()), 1e-6)

    def test_class_weights(self):
        counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]
        w = L.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.sum().item(), 9.0, places=4)
        self.assertLess(w[8].item(), w[0].item())
        np.testing.assert_allclose((w * torch.tensor(counts, dtype=torch.float32)).numpy(),
                                   (w[0] * counts[0]).item(), rtol=1e-5)  # w_c * n_c là hằng số
        wb = L.class_weights(counts, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertGreater(wb[8].item(), w[8].item())  # beta < 1 làm trọng số ít cực đoan hơn

    def test_cutmix_lambda_equals_kept_area(self):
        rng = np.random.default_rng(0)
        x = torch.zeros(8, 3, 224, 224)
        x[1::2] = 1.0  # ảnh lẻ toàn 1, ảnh chẵn toàn 0
        y = torch.arange(8)
        for _ in range(50):
            xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix", rng=rng)
            self.assertTrue(torch.equal(ya, y))
            for i in range(8):
                if x[i, 0, 0, 0] != x[yb[i], 0, 0, 0]:   # hai ảnh khác nhau: đếm được diện tích dán
                    pasted = (xm[i, 0] != x[i, 0]).float().mean().item()
                    self.assertAlmostEqual(1 - lam, pasted, places=6)

    def test_mixup_and_mixed_loss(self):
        rng = np.random.default_rng(1)
        x = torch.randn(16, 3, 8, 8)
        y = torch.randint(0, 9, (16,))
        xm, (ya, yb, lam) = L.mix_batch(x, y, 0.4, "mixup", rng=rng)
        perm = [int(torch.nonzero((y == b)).flatten()[0]) for b in yb]  # chỉ để kiểm tra nhãn hợp lệ
        self.assertEqual(len(perm), 16)
        logits = torch.randn(16, 9)
        crit = nn.CrossEntropyLoss()
        ref = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        self.assertAlmostEqual(L.mixed_loss(crit, logits, (ya, yb, lam)).item(), ref.item(), places=6)


class TestModel(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = M.build_model("resnet18", pretrained=False)
        groups = {g["name"]: g for g in M.param_groups(m, 1e-4, 1e-3, 0.05)}
        names = {id(p): n for n, p in m.named_parameters()}
        for g in groups.values():
            for p in g["params"]:
                n = names[id(p)]
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0, n)
                if n.startswith("fc."):
                    self.assertEqual(g["lr"], 1e-3, n)
                else:
                    self.assertEqual(g["lr"], 1e-4, n)
        n_total = sum(len(g["params"]) for g in groups.values())
        self.assertEqual(n_total, len(list(m.parameters())))

    def test_frozen_backbone_keeps_bn_eval_and_stats(self):
        m = M.build_model("resnet18", pretrained=False)
        M.freeze_backbone(m)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertEqual(sorted(trainable), ["fc.bias", "fc.weight"])
        self.assertEqual(len(M.param_groups(m, 1e-4, 1e-3, 0.05)), 2)  # chỉ head
        M.set_train_mode(m)
        self.assertTrue(m.fc.training)
        self.assertFalse(m.bn1.training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64) * 5 + 3)
        self.assertTrue(torch.equal(before, m.bn1.running_mean))

    def test_count_params_and_gmacs(self):
        m = M.build_model("resnet50", pretrained=False)
        self.assertAlmostEqual(M.count_params(m), 23.53, delta=0.05)    # head 9 lớp
        self.assertAlmostEqual(M.count_gmacs(m, 224), 4.09, delta=0.05)


class TestTrainHelpers(unittest.TestCase):
    def test_lr_schedule_warmup_cosine(self):
        total, warm = 1000, 100
        f = [T.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[0], 1 / warm)
        self.assertAlmostEqual(f[warm - 1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(f[:warm], f[1:warm])))
        self.assertTrue(all(a >= b for a, b in zip(f[warm:], f[warm + 1:])))
        self.assertLess(f[-1], 1e-4)
        self.assertAlmostEqual(T.lr_factor(warm + (total - warm) // 2, total, warm), 0.5, places=2)

    def test_parse_overrides(self):
        d = T.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "amp=false",
                               "sampler=balanced", "lr_head=2e-3", "drop_path_rate=0.1"])
        self.assertEqual(d, {"seed": 3, "loss": "focal", "ema_decay": None, "amp": False,
                             "sampler": "balanced", "lr_head": 2e-3, "drop_path_rate": 0.1})
        with self.assertRaises(KeyError):
            T.parse_overrides(["nope=1"])

    def test_ema(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        ema = T.EMA(m, decay=0.9)
        w0 = ema.module[0].weight.clone()
        with torch.no_grad():
            m[0].weight.add_(1.0)
            m[1].running_mean.add_(1.0)
        ema.update(m)
        d = min(0.9, 2 / 11)
        torch.testing.assert_close(ema.module[0].weight, d * w0 + (1 - d) * m[0].weight)
        torch.testing.assert_close(ema.module[1].running_mean, (1 - d) * torch.ones(4))
        self.assertEqual(int(ema.module[1].num_batches_tracked), int(m[1].num_batches_tracked))

    def test_kd_loss(self):
        s = torch.randn(8, 9)
        self.assertLess(T.kd_loss(s, s, 4.0).item(), 1e-6)            # cùng phân phối: KL = 0
        t = torch.randn(8, 9)
        ref = F.kl_div(F.log_softmax(s / 2, -1), F.softmax(t / 2, -1), reduction="batchmean") * 4
        self.assertAlmostEqual(T.kd_loss(s, t, 2.0).item(), ref.item(), places=5)
        self.assertGreater(T.kd_loss(s, t, 2.0).item(), 0)

    def test_dinov2_builds_at_224(self):
        m = M.build_model("vit_small_patch14_dinov2", pretrained=False, img_size=224).eval()
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.randn(1, 3, 224, 224)).shape), (1, 9))

    def test_defaults_match_guide(self):
        c = T.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))
        self.assertFalse(c.save_test_predictions)


class TestInference(unittest.TestCase):
    def test_views(self):
        x = torch.arange(2 * 3 * 256 * 256, dtype=torch.float32).reshape(2, 3, 256, 256)
        torch.testing.assert_close(I.view_hflip(I.view_hflip(x)), x)
        self.assertTrue(torch.equal(I.view_hflip(x)[..., 0], x[..., -1]))
        crops = I.views_multicrop(x, 224)
        self.assertEqual(len(crops), 5)
        self.assertTrue(all(c.shape == (2, 3, 224, 224) for c in crops))
        self.assertTrue(torch.equal(crops[3][..., -1, -1], x[..., -1, -1]))      # góc dưới phải
        self.assertTrue(torch.equal(crops[4], x[..., 16:240, 16:240]))           # crop giữa
        self.assertEqual(len(I.views_multicrop(x, 224, flip=True)), 10)
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, (224, 256, 288))], [224, 256, 288])

    def test_aggregate_and_ensemble(self):
        rng = np.random.default_rng(0)
        views = [rng.normal(size=(20, 9)) * 2 for _ in range(3)]
        for space in ("prob", "logit"):
            p = I.aggregate_views(views, space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
        np.testing.assert_allclose(I.aggregate_views(views, "logit"), I.softmax(np.mean(views, 0)))
        np.testing.assert_allclose(I.aggregate_views(views[:1], "prob"), I.softmax(views[0]))
        probs = [I.softmax(v) for v in views]
        np.testing.assert_allclose(I.ensemble_probs(probs), np.mean(probs, 0))

    def test_temperature_recovers_true_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(20000, 9)) * 3
        p_true = I.softmax(z, 2.5)                      # nhãn sinh từ softmax(z / 2.5)
        y = np.array([rng.choice(9, p=p) for p in p_true])
        T_hat = I.fit_temperature(z, y)
        self.assertAlmostEqual(T_hat, 2.5, delta=0.1)
        np.testing.assert_array_equal(I.apply_temperature(z, T_hat).argmax(1), z.argmax(1))
        probs, temps = I.crossfit_temperature(z, y)
        self.assertEqual(len(temps), 2)
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)

    def _randomize_bn(self, m):
        g = torch.Generator().manual_seed(0)
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
                mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
                mod.weight.data.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
                mod.bias.data.copy_(torch.randn(mod.num_features, generator=g) * 0.1)

    def test_fuse_conv_bn(self):
        x = torch.randn(2, 3, 96, 96)
        for name, min_pairs in (("resnet18", 20), ("efficientnet_b0", 40), ("mobilenetv3_large_100", 40),
                                ("convnext_atto", 0)):
            m = M.build_model(name, pretrained=False).eval()
            self._randomize_bn(m)
            fused = I.fuse_conv_bn(m, example=x, atol=1e-3)
            self.assertGreaterEqual(fused.fused_pairs, min_pairs, name)
            self.assertLess(fused.fuse_max_abs_err, 1e-3, name)
            n_bn = sum(isinstance(mod, nn.BatchNorm2d) for mod in fused.modules())
            if name != "convnext_atto":
                self.assertEqual(n_bn, 0, f"{name}: còn {n_bn} BN chưa gộp")


class TestBenchmark(unittest.TestCase):
    def test_bench_percentiles(self):
        r = B.bench(lambda: sum(range(1000)), warmup=10, iters=60)
        self.assertEqual(r["n"], 60)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])
        with self.assertRaises(ValueError):
            B.bench(lambda: None, iters=10)

    def test_latency_report_cpu(self):
        r = B.latency_report(nn.Conv2d(3, 4, 3), 1, 32, "fp32", device="cpu", iters=50)
        self.assertEqual((r["batch"], r["img_size"], r["dtype"]), (1, 32, "fp32"))
        self.assertTrue(math.isfinite(r["p95_ms"]))


if __name__ == "__main__":
    unittest.main()
