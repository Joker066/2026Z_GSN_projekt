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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    return p.parse_args()


@torch.no_grad()
def validate(model, dataset, device) -> tuple[float, float]:
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    for lr, hr, _ in dataset:
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)
        sr = model(lr)
        psnr_sum += psnr(sr, hr, crop=dataset.scale).item()
        ssim_sum += ssim(sr, hr, crop=dataset.scale).item()
    model.train()
    return psnr_sum / len(dataset), ssim_sum / len(dataset)


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

    iteration, best_psnr = 0, 0.0
    if args.resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iteration, best_psnr = ckpt["iteration"], ckpt["best_psnr"]
        print(f"Resumed at iteration {iteration}, best PSNR {best_psnr:.3f}")

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
        log.writerow(["iteration", "train_loss", "lr", "val_psnr", "val_ssim"])

    def save(path):
        torch.save({
            "iteration": iteration,
            "best_psnr": best_psnr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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
                val_psnr, val_ssim = validate(model, val_dataset, device)
                current_lr = optimizer.param_groups[0]["lr"]
                speed = n / (time.time() - t0)
                print(f"Iteration {iteration}/{args.iterations}  loss {avg_loss:.5f}  lr {current_lr:.1e}  "
                      f"val PSNR {val_psnr:.3f}  SSIM {val_ssim:.4f}  ({speed:.1f} it/s)")
                log.writerow([iteration, avg_loss, current_lr, val_psnr, val_ssim])
                log_file.flush()
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save(out / "best.pt")
                save(out / "last.pt")
                loss_sum, t0 = 0.0, time.time()

            if iteration >= args.iterations:
                break

    log_file.close()
    print(f"Done. Best val PSNR {best_psnr:.3f} dB -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
