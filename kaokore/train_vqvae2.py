import os, sys
import typer
import comet_ml
import torch

from tqdm import tqdm
from rich.progress import track
from contextlib import nullcontext
from typing import Optional
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from models.vae import VQ_VAE2
from scheduler import CycleScheduler
import distributed as dist
from data import Kaokore

app = typer.Typer()
port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)

def save_reconstruction(model, img, sample_size, epoch, experiment: comet_ml.Experiment = None):
    model.eval()
    sample = img[:sample_size]
    with torch.no_grad(): out, _ = model(sample)
    image_grid = make_grid(torch.cat([sample.mul(0.5).add(0.5), out.mul(0.5).add(0.5)], 0))
    name = f"train-{epoch+1:05}"
    save_image(
        torch.cat([sample, out], 0),
        f"sample/{name}.png",
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )
    if experiment:
        # breakpoint()
        experiment.log_image(image_data=image_grid.to('cpu'), name=name, image_channels='first', overwrite=True)

def train(epoch: int, loader: DataLoader, model: VQ_VAE2, optimizer, scheduler, device, experiment: comet_ml.Experiment = None):
    if dist.is_primary(): loader = tqdm(loader)

    sample_size = 8

    loss_sum = 0
    mse_sum = 0
    commit_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.train()
        model.zero_grad()

        img = img.to(device)
        out, diff = model(img)
        loss, recon_loss, commit_loss = model.loss(img, out, diff, return_parts=True)
        loss.backward()

        if scheduler is not None: scheduler.step()
        optimizer.step()

        part_loss_sum = loss.item() * img.shape[0]
        part_mse_sum = recon_loss.item() * img.shape[0]
        part_commit_sum = commit_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"loss_sum": part_loss_sum,
                "mse_sum": part_mse_sum,
                "commit_sum": part_commit_sum,
                "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            loss_sum += part['loss_sum']
            mse_sum += part["mse_sum"]
            commit_sum += part['commit_sum']
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description('; '.join([
                f"epoch: {epoch + 1}",
                f"mse: {recon_loss.item():.5f}",
                f"commitment: {commit_loss.item():.5f}",
                f"avg loss: {loss_sum / mse_n:.5f}",
                f"avg mse: {mse_sum / mse_n:.5f}",
                f"lr: {lr:.5f}"
            ]))

    save_reconstruction(model, img, sample_size, epoch, experiment)
    if experiment:
        experiment.log_metrics(dict(
            loss=loss_sum / mse_n,
            mse=mse_sum / mse_n,
            commitment=commit_sum / mse_n
        ))

# def test(epoch: int, loader: DataLoader, model: VQ_VAE2, device)

def main(path: str, category: str = 'gender', n_gpu: int = 1, dist_url: str = f"tcp://127.0.0.1:{port}",
          size: int = 256, batchsize: int = 128, epoch: int = 560, lr: float = 0.001, sched: Optional[str] = None,
          comet=False):
    device = "cuda"

    distributed = dist.get_world_size() > 1

    # DATA
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # dataset = datasets.ImageFolder(path, transform=transform)
    dataset = Kaokore(path, split='train', category=category, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=distributed)
    loader = DataLoader(dataset, batch_size=batchsize // n_gpu, sampler=sampler, num_workers=2)

    # MODEL
    model = VQ_VAE2(commit_coef=0.1).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    # OPTIMIZER, SCHEDULER
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            lr,
            n_iter=len(loader) * epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    # COMET
    experiment = None
    if comet:
        print("COMET_ML_API_KEY: ", os.getenv("COMET_ML_API_KEY"))
        experiment = comet_ml.Experiment(api_key=os.getenv("COMET_ML_API_KEY"),
                                        project_name='cs274e')
        experiment.set_name(f'kaokore[{category}]/VQVAE2')
        experiment.add_tags(['kaokore', category, 'VQVAE2',
                            f'lr={lr}',
                            f'batch={batchsize}'])


    # TRAIN LOOP
    for i in range(epoch):
        with experiment.train() if comet else nullcontext():
            train(i, loader, model, optimizer, scheduler, device, experiment=experiment)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")

@app.command()
def run(path: str, category: str = 'gender', n_gpu: int = 1, dist_url: str = f"tcp://127.0.0.1:{port}",
          size: int = 256, batchsize: int = 128, epoch: int = 560, lr: float = 0.001, sched: Optional[str] = None,
          comet: bool = False):
    print(locals())
    dist.launch(main, n_gpu_per_machine=n_gpu, dist_url=dist_url,
                args=(path, category, n_gpu, dist_url, size, batchsize, epoch, lr, sched, comet))

if __name__ == "__main__":
    app()
