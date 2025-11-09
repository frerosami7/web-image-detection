import argparse


def parse_args():
    ap = argparse.ArgumentParser(description="Train an anomalib model on a folder dataset and export Torch model.")
    ap.add_argument("--data-root", required=True, help="Root folder with category data (normal + abnormal).")
    ap.add_argument("--model", default="Patchcore", help="Model class in anomalib.models, e.g., Patchcore, Padim, Stfpm")
    ap.add_argument("--task", default="classification", choices=["segmentation", "classification"], help="Learning task")
    ap.add_argument("--image-size", type=int, default=256, help="Resize for datamodule")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience (if callback available)")
    ap.add_argument("--export", action="store_true", help="Export best model to Torch format (model.pt)")
    return ap.parse_args()


def main():
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.deploy import ExportType
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Anomalib not installed. Run: pip install anomalib") from e

    args = parse_args()

    # Datamodule
    dm = Folder(
        root=args.data_root,
        task=args.task,
        image_size=args.image_size,
    )

    # Model
    mod = __import__("anomalib.models", fromlist=[args.model])
    ModelCls = getattr(mod, args.model)
    model = ModelCls()

    # Callbacks
    callbacks = []
    try:
        ckpt_cb = ModelCheckpoint(monitor="image_AUROC", mode="max", save_last=True, every_n_epochs=1)
        callbacks.append(ckpt_cb)
        es_cb = EarlyStopping(monitor="image_AUROC", mode="max", patience=args.patience)
        callbacks.append(es_cb)
    except Exception:
        pass  # Some models/tasks may not log image_AUROC

    engine = Engine(max_epochs=args.max_epochs, callbacks=callbacks, task=args.task)

    print("[train] fitting model...")
    engine.fit(datamodule=dm, model=model)

    print("[train] testing model...")
    try:
        engine.test(datamodule=dm, model=model)
    except Exception:
        print("[warn] test phase skipped (no test set or metric)")

    if args.export:
        print("[export] exporting Torch model...")
        try:
            path_export = engine.export(export_type=ExportType.TORCH, model=model)
            print(f"[export] Torch model saved: {path_export}")
        except Exception as e:
            print(f"[export] failed: {e}")


if __name__ == "__main__":
    main()
