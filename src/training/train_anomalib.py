import argparse

def main():
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Anomalib not installed. Run: pip install anomalib") from e

    args = parse_args()
    # Instantiate datamodule and model
    dm = Folder(root=args.data_root, task=args.task, image_size=args.image_size)
    mod = __import__("anomalib.models", fromlist=[args.model])
    ModelCls = getattr(mod, args.model)
    model = ModelCls()
    engine = Engine()
    # Train
    engine.fit(datamodule=dm, model=model)


def parse_args():
    ap = argparse.ArgumentParser(description="Train an anomalib model on a folder dataset.")
    ap.add_argument("--data-root", required=True, help="Root folder with train/test structure.")
    ap.add_argument("--model", default="Patchcore", help="Model class in anomalib.models, e.g., Patchcore, Padim, Stfpm")
    ap.add_argument("--task", default="segmentation", choices=["segmentation", "classification"], help="Learning task")
    ap.add_argument("--image-size", type=int, default=256, help="Resize for datamodule")
    return ap.parse_args()


if __name__ == "__main__":
    main()
