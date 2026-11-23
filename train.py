"""Train the baseline:  python train.py --name baseline_x2

Quick smoke test first:
  python train.py --name smoke --iterations 2000 --val-every 500
Val PSNR on Set5 should pass bicubic (33.67 dB) within that run.

Outputs in runs/<name>/:
  best.pt   - weights with the best val PSNR (use this in evaluate.py)
  last.pt   - latest weights + optimizer/scheduler state (for --resume)
  log.csv   - iteration, avg train loss, lr, val PSNR, val SSIM
"""
import argparse
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from sr import metrics
from sr.data import SREvalDataset, SRTrainDataset
from sr.metrics import psnr, ssim
from sr.models import SuperResolutionNet, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-dir", default="data/DIV2K_train_HR")
    p.add_argument("--val-dir", default="data/Set5_HR")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=16)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--batch", type=int, default=16, help="patches per step")
    p.add_argument("--crops_per_image", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_patch", type=int, default=48)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    p.add_argument("--run-id", type=str, default=None, help="W&B run ID to continue (default: the one saved in last.pt)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p.parse_args()



def draw_PSNR(iterations, PSNR, outpath: Path):
    """Draw PSNR curves from the CSV log."""
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, PSNR, label="PSNR")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR (Validation)")
    plt.legend()
    plt.grid()
    plt.savefig(outpath)
    plt.close()



def draw_SSIM(iterations, SSIM, outpath: Path):
    """Draw SSIM curves from the CSV log."""
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, SSIM, label="SSIM")
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("SSIM (Validation)")
    plt.legend()
    plt.grid()
    plt.savefig(outpath)
    plt.close()



def draw_losses(iterations, train_loss, val_loss, outpath: Path):
    """Draw training loss curve from the CSV log."""
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, train_loss, label="Train Loss")
    plt.plot(iterations, val_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.savefig(outpath)
    plt.close()


def analyze_training_log(log_path: Path):
    """Draw loss and accuracy curves from the CSV log."""
    iterations, train_loss, val_psnr, val_ssim, val_loss = [], [], [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row["iteration"]))
            train_loss.append(float(row["train_loss"]))
            val_psnr.append(float(row["val_psnr"]))
            val_ssim.append(float(row["val_ssim"]))
            val_loss.append(float(row["val_loss"]))
    out_dir = log_path.parent
    draw_PSNR(iterations, val_psnr, out_dir / "PSNR.png")
    draw_SSIM(iterations, val_ssim, out_dir / "SSIM.png")
    draw_losses(iterations, train_loss, val_loss, out_dir / "loss.png")


@torch.no_grad()
def validate(model, dataset, device, loss_function) -> tuple[float, float, float, torch.Tensor]:
    """Returns avg PSNR, SSIM, loss and the SR output [3,H,W] of the first image (for logging)."""
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    loss_sum = 0.0
    sample = None
    for lr, hr, _ in dataset:
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)
        sr = model(lr)
        if sample is None:
            sample = sr[0]
        psnr_sum += psnr(sr, hr, crop=dataset.scale).item()
        ssim_sum += ssim(sr, hr, crop=dataset.scale).item()
        loss_sum += loss_function(sr, hr).item()
    model.train()
    return psnr_sum / len(dataset), ssim_sum / len(dataset), loss_sum / len(dataset), sample


def main():
    args = parse_args()
    assert args.batch % args.crops_per_image == 0, "--batch must be divisible by --crops_per_image"
    torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN pick the fastest conv algorithm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    model = SuperResolutionNet(scale=args.scale, channels=args.channels, n_blocks=args.n_blocks).to(device)
    print(f"Model: {count_parameters(model):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # Halve the LR at 50% and 75% of training, whatever --iterations is.
    milestones = [args.iterations // 2, args.iterations * 3 // 4]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    iteration, best_psnr, run_id = 0, 0.0, None
    if args.resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iteration, best_psnr = ckpt["iteration"], ckpt["best_psnr"]
        run_id = ckpt.get("run_id")  # .get: older checkpoints have no run_id
        print(f"Resumed at iteration {iteration}, best PSNR {best_psnr:.3f}")
    if args.run_id:  # manual override
        run_id = args.run_id

    # After the checkpoint load, so a resumed run continues the same W&B run.
    run = None
    if not args.wandb:
        wandb.init(mode="disabled")
    else:
        run = wandb.init(
            project="super-resolution",
            name=args.name,
            config=vars(args),
            dir=out,  # W&B's local files go to runs/<name>/wandb/ (ignored by git)
            id=run_id,  # None -> new run
            resume="allow",
        )
        print(f"W&B run id: {run.id}")

    train_dataset = SRTrainDataset(hr_dir=args.train_dir, scale=args.scale, lr_patch=args.lr_patch,
                                   crops_per_image=args.crops_per_image)
    # Each dataset item is crops_per_image patches, so fewer items per batch.
    train_loader = DataLoader(train_dataset, batch_size=args.batch // args.crops_per_image, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_dataset = SREvalDataset(hr_dir=args.val_dir, scale=args.scale)

    log_file = open(out / "log.csv", "a" if args.resume else "w", newline="")
    log = csv.writer(log_file)
    if not args.resume:
        log.writerow(["iteration", "train_loss", "lr", "val_psnr", "val_ssim", "val_loss"])

    def save(path):
        torch.save({
            "iteration": iteration,
            "best_psnr": best_psnr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "run_id": run.id if run else None,
            "args": vars(args),
        }, path)

    loss_sum, t0 = 0.0, time.time()
    while iteration < args.iterations:  # iteration-based: re-iterate the loader as often as needed
        for lr, hr in train_loader:
            lr = lr.flatten(0, 1).to(device, non_blocking=True)  # [B,k,3,p,p] -> [B*k,3,p,p]
            hr = hr.flatten(0, 1).to(device, non_blocking=True)
            sr = model(lr)
            loss = F.l1_loss(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            loss_sum += loss.item()

            if iteration % args.val_every == 0 or iteration == args.iterations:
                n = args.val_every if iteration % args.val_every == 0 else iteration % args.val_every
                avg_loss = loss_sum / n
                val_psnr, val_ssim, val_loss, sample = validate(model, val_dataset, device, F.l1_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                speed = n / (time.time() - t0)
                print(f"Iteration {iteration}/{args.iterations}  loss {avg_loss:.5f}  lr {current_lr:.1e}  "
                      f"val PSNR {val_psnr:.3f}  SSIM {val_ssim:.4f}  Loss {val_loss:.5f}  ({speed:.1f} it/s)")
                log.writerow([iteration, avg_loss, current_lr, val_psnr, val_ssim, val_loss])
                log_file.flush()
                # wandb.Image expects uint8 in [0, 255] - it does NOT rescale [0, 1] floats (just casts -> black)
                img_hwc = sample.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                wandb.log({
                    "train/loss": avg_loss,
                    "val/psnr": val_psnr,
                    "val/ssim": val_ssim,
                    "val/loss": val_loss,
                    "lr": current_lr,
                    "val/sample": wandb.Image(img_hwc, caption=f"SR, iteration {iteration}"),
                }, step=iteration)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save(out / "best.pt")
                save(out / "last.pt")
                loss_sum, t0 = 0.0, time.time()

            if iteration >= args.iterations:
                break

    log_file.close()
    wandb.finish()
    print(f"Done. Best val PSNR {best_psnr:.3f} dB -> {out / 'best.pt'}")

    analyze_training_log(out / "log.csv")


if __name__ == "__main__":
    main()
