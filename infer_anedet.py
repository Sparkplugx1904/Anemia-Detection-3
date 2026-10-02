import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys, time, json

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Install tflite-runtime atau tensorflow:")
        print("  pip install tflite-runtime")
        sys.exit(1)

SEG_MODEL = "AnedetApp/app/src/main/assets/yolo26n_seg_fp16.tflite"
CLS_MODEL = "AnedetApp/app/src/main/assets/yolo26s_cls_fp16.tflite"
DATA_DIR = "Datasets/Test"
SEG_SZ = 320
CLS_SZ = 448
CONF_TH = 0.25
FILL = 114


class Pipeline:
    def __init__(self):
        self.seg = tflite.Interpreter(model_path=SEG_MODEL)
        self.cls = tflite.Interpreter(model_path=CLS_MODEL)
        self.seg.allocate_tensors()
        self.cls.allocate_tensors()
        self.seg_in = self.seg.get_input_details()[0]
        self.seg_out = self.seg.get_output_details()
        self.cls_in = self.cls.get_input_details()[0]
        self.cls_out = self.cls.get_output_details()

        for i, d in enumerate(self.seg_out):
            print(f"  seg out[{i}]: {d['shape']}")
        print(f"  cls out: {self.cls_out[0]['shape']}")

    @staticmethod
    def _lb(img, sz):
        w, h = img.size
        scale = min(sz / w, sz / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        result = Image.new("RGB", (sz, sz), (FILL, FILL, FILL))
        pl = (sz - nw) // 2
        pt = (sz - nh) // 2
        result.paste(resized, (pl, pt))
        return result, scale, pl, pt

    def _seg_preprocess(self, img):
        lb, scale, pl, pt = self._lb(img, SEG_SZ)
        arr = np.array(lb, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0), scale, pl, pt

    def _cls_preprocess(self, img):
        lb, _, _, _ = self._lb(img, CLS_SZ)
        arr = np.array(lb, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _compute_mask(self, proto, coeffs, x1m, y1m, x2m, y2m):
        h, w = proto.shape[:2]
        pl = int(x1m * w / SEG_SZ); pr = int(x2m * w / SEG_SZ) + 1
        pt = int(y1m * h / SEG_SZ); pb = int(y2m * h / SEG_SZ) + 1
        pl = max(0, min(pl, w-1)); pr = max(0, min(pr, w-1))
        pt = max(0, min(pt, h-1)); pb = max(0, min(pb, h-1))

        mask = np.zeros((h, w), dtype=np.float32)
        for y in range(pt, pb + 1):
            for x in range(pl, pr + 1):
                if self._sigmoid(np.dot(proto[y, x], coeffs)) > 0.5:
                    mask[y, x] = 1.0
        return mask

    def _mask_to_orig(self, mask_80, orig_w, orig_h, scale, pl, pt):
        mask_img = Image.fromarray((mask_80 * 255).astype(np.uint8))
        a = scale / 4.0
        transformed = mask_img.transform(
            (orig_w, orig_h), Image.AFFINE,
            (a, 0, pl / 4.0, 0, a, pt / 4.0),
            resample=Image.BILINEAR,
        )
        return np.array(transformed) > 127

    def _mask_boundary(self, mask):
        from scipy.ndimage import binary_dilation
        struct = np.ones((3, 3))
        dilated = binary_dilation(mask, structure=struct)
        return dilated & ~mask

    def run_seg(self, img):
        inp, scale, pl, pt = self._seg_preprocess(img)
        self.seg.set_tensor(self.seg_in["index"], inp)
        self.seg.invoke()

        det = self.seg.get_tensor(self.seg_out[0]["index"])[0]
        proto = self.seg.get_tensor(self.seg_out[1]["index"])[0]

        best_conf, best_idx = 0, -1
        for i in range(det.shape[0]):
            c = float(det[i, 4])
            if c > best_conf and c > CONF_TH:
                best_conf = c
                best_idx = i

        if best_idx < 0:
            return None, None, None

        x1r, y1r, x2r, y2r = det[best_idx, 0:4]
        if x1r <= 1.01 and x2r <= 1.01:
            x1r *= SEG_SZ; x2r *= SEG_SZ
            y1r *= SEG_SZ; y2r *= SEG_SZ

        x1 = max(0, (x1r - pl) / scale)
        y1 = max(0, (y1r - pt) / scale)
        x2 = min(img.width, (x2r - pl) / scale)
        y2 = min(img.height, (y2r - pt) / scale)

        coeffs = det[best_idx, 6:38]
        mask_80 = self._compute_mask(proto, coeffs, x1r, y1r, x2r, y2r)
        mask_orig = self._mask_to_orig(mask_80, img.width, img.height, scale, pl, pt)
        return (x1, y1, x2, y2), mask_orig, best_conf

    def crop_conj(self, img, bbox, mask):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.width, x2); y2 = min(img.height, y2)

        arr = np.array(img, dtype=np.float32)
        mask_3 = np.stack([mask] * 3, axis=-1).astype(np.float32)
        masked = arr * mask_3
        cropped = masked[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return None
        return Image.fromarray(cropped.astype(np.uint8))

    def run_cls(self, cropped):
        inp = self._cls_preprocess(cropped)
        self.cls.set_tensor(self.cls_in["index"], inp)
        self.cls.invoke()
        out = self.cls.get_tensor(self.cls_out[0]["index"])[0]

        if 0.95 < out.sum() < 1.05:
            return float(out[0]), float(out[1])
        mx = out.max()
        ex = np.exp(out - mx)
        probs = ex / ex.sum()
        return float(probs[0]), float(probs[1])

    def draw_overlay(self, img, mask, ap, nap, is_anemic, bbox):
        img_rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        mask_px = np.array(mask, dtype=np.uint8)
        green = Image.new("RGBA", img.size, (76, 175, 80, 160))
        overlay = Image.composite(green, overlay, Image.fromarray((mask_px * 255).astype(np.uint8)))

        try:
            from scipy.ndimage import binary_dilation
            boundary = binary_dilation(mask_px, structure=np.ones((3, 3))) & ~mask_px
            boundary_px = np.argwhere(boundary)
            for y, x in boundary_px[::3]:
                overlay.putpixel((x, y), (255, 255, 0, 255))
        except ImportError:
            pass

        result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        draw = ImageDraw.Draw(result)

        label = "Anemia" if is_anemic else "Non-Anemia"
        prob = ap if is_anemic else nap
        text = f"{label} ({prob:.1%})"

        try:
            font = ImageFont.truetype("arial.ttf", max(20, img.width // 25))
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        pad = 6
        tx = img.width - tw - pad * 2 - 10
        ty = 10

        draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=(0, 0, 0, 200))
        draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

        return result

    MAX_DIM = 2000

    def analyze(self, path, save_result=False):
        img = Image.open(path).convert("RGB")
        if max(img.width, img.height) > self.MAX_DIM:
            img.thumbnail((self.MAX_DIM, self.MAX_DIM), Image.LANCZOS)
        t0 = time.time()
        bbox, mask, seg_conf = self.run_seg(img)
        if bbox is None:
            return {"error": "Konjungtiva tidak terdeteksi", "time_ms": (time.time() - t0) * 1000}

        cropped = self.crop_conj(img, bbox, mask)
        if cropped is None:
            return {"error": "Gagal crop konjungtiva", "time_ms": (time.time() - t0) * 1000}

        ap, nap = self.run_cls(cropped)
        dt = (time.time() - t0) * 1000
        is_anemic = ap > nap

        result_img = self.draw_overlay(img, mask, ap, nap, is_anemic, bbox) if save_result else None

        return {
            "is_anemic": is_anemic,
            "anemic_prob": ap,
            "non_anemic_prob": nap,
            "confidence": max(ap, nap),
            "margin": abs(ap - nap),
            "seg_confidence": float(seg_conf),
            "bbox": bbox,
            "time_ms": dt,
            "error": None,
            "result_img": result_img,
        }


def load_images(data_dir):
    imgs = []
    for label, cls_name in [(1, "Anemia"), (0, "NonAnemia")]:
        d = os.path.join(data_dir, cls_name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append({"path": os.path.join(d, f), "true_label": label, "cls": cls_name})
    return imgs


def main():
    import argparse
    p = argparse.ArgumentParser(description="Anemia Detection Inference")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--output-json", default=None)
    p.add_argument("--save-result", action="store_true", help="Simpan gambar hasil ke folder result/")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    for f, n in [(SEG_MODEL, "Segmentation"), (CLS_MODEL, "Classification")]:
        if not os.path.exists(f):
            print(f"Model {n} tidak ditemukan: {f}")
            sys.exit(1)
    if not os.path.isdir(args.data_dir):
        print(f"Data dir tidak ditemukan: {args.data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Anemia Detection — Inference Test")
    print("=" * 60)
    pipe = Pipeline()

    imgs = load_images(args.data_dir)
    print(f"\nGambar ditemukan: {len(imgs)}")
    a = sum(1 for i in imgs if i["true_label"] == 1)
    n_g = sum(1 for i in imgs if i["true_label"] == 0)
    print(f"  Anemia: {a}, Non-Anemia: {n_g}")

    results = []
    total_time = 0
    for i, info in enumerate(imgs):
        rp = os.path.relpath(info["path"], args.data_dir)
        res = pipe.analyze(info["path"], save_result=args.save_result)
        res["path"] = info["path"]
        res["true_label"] = info["true_label"]
        res["true_class"] = info["cls"]
        results.append(res)
        total_time += res.get("time_ms", 0)

        if args.save_result and res["result_img"] is not None:
            result_dir = os.path.join(os.path.dirname(info["path"]), "result")
            os.makedirs(result_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(info["path"]))[0]
            res["result_img"].save(os.path.join(result_dir, f"{base}_result.jpg"), quality=92)

        if args.verbose:
            ok = res["error"] is None
            pred = "Anemia" if res.get("is_anemic") else "Non-Anemia"
            gt = info["cls"]
            match = "OK" if (ok and res["is_anemic"] == (info["true_label"] == 1)) else "XX"
            err = f" [{res['error']}]" if res["error"] else ""
            ap_s = f"{res['anemic_prob']:.2f}".replace('.', ',')
            nap_s = f"{res['non_anemic_prob']:.2f}".replace('.', ',')
            print(f"  [{i+1}/{len(imgs)}] {rp} -> {pred} (gt={gt}) {match}  A={ap_s} NA={nap_s}{err}")

    tp = sum(1 for r in results if r["error"] is None and r["is_anemic"] and r["true_label"] == 1)
    tn = sum(1 for r in results if r["error"] is None and not r["is_anemic"] and r["true_label"] == 0)
    fp = sum(1 for r in results if r["error"] is None and r["is_anemic"] and r["true_label"] == 0)
    fn = sum(1 for r in results if r["error"] is None and not r["is_anemic"] and r["true_label"] == 1)
    errs = sum(1 for r in results if r["error"] is not None)
    valid = tp + tn + fp + fn
    acc = (tp + tn) / valid if valid else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print("\n" + "=" * 60)
    print("  HASIL")
    print("=" * 60)
    print(f"\nTotal: {len(imgs)}, Berhasil: {valid}, Error: {errs}")
    print(f"\nConfusion Matrix:")
    print(f"                Prediksi")
    print(f"               Anemia  Non-Anemia")
    print(f"Anemia         {tp:5d}   {fn:5d}")
    print(f"Non-Anemia     {fp:5d}   {tn:5d}")
    print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"\nRata-rata waktu inferensi: {total_time/len(imgs):.0f}ms")

    if valid:
        ar = [r for r in results if r["true_label"] == 1]
        nr = [r for r in results if r["true_label"] == 0]
        if ar:
            ac_a = sum(1 for r in ar if r["error"] is None and r["is_anemic"])
            print(f"  Anemia    : {ac_a}/{len(ar)} benar ({ac_a/len(ar)*100:.1f}%)")
        if nr:
            nc_n = sum(1 for r in nr if r["error"] is None and not r["is_anemic"])
            print(f"  Non-Anemia: {nc_n}/{len(nr)} benar ({nc_n/len(nr)*100:.1f}%)")

    if args.output_json:
        out = {
            "summary": {
                "total": len(imgs),
                "successful": valid,
                "errors": errs,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "avg_time_ms": total_time / len(imgs),
            },
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "per_image": [
                {
                    "path": r["path"],
                    "true_label": r["true_label"],
                    "predicted_anemic": r.get("is_anemic"),
                    "anemic_prob": r.get("anemic_prob"),
                    "non_anemic_prob": r.get("non_anemic_prob"),
                    "seg_confidence": r.get("seg_confidence"),
                    "error": r["error"],
                    "time_ms": r.get("time_ms"),
                }
                for r in results
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nHasil disimpan ke: {args.output_json}")


if __name__ == "__main__":
    main()
