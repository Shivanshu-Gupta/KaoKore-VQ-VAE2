import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from PIL import Image
import os


def graph_test_samples(vqvae_path, vqvae2_path, padding=25, epochs=[1, 12, 14, 16, 18, 100]):
    with Image.open(os.path.join(vqvae_path, "test-1.png")) as test_im:
        test_im = test_im.crop((0, 0, test_im.width, test_im.height / 2))
    width, height = test_im.size

    samples_im = Image.new("RGB", (width * 2 + padding, height * (len(epochs) + 1) + padding))

    for i, epoch in enumerate(epochs):

        with Image.open(os.path.join(vqvae_path, "test-{}.png".format(epoch))) as im:
            im = im.crop((0, im.height / 2, im.width, im.height))
        samples_im.paste(im, (0, i * height))

        with Image.open(os.path.join(vqvae2_path, "samples/test-{0:05d}.png".format(epoch))) as im:
            im = im.crop((0, im.height / 2, im.width, im.height))
        samples_im.paste(im, (width + padding, i * height))

    samples_im.paste(test_im, (0, height * len(epochs) + padding))
    samples_im.paste(test_im, (width + padding, height * len(epochs) + padding))

    samples_im.save("samples.png")


def plot_test_metrics(vqvae_paths, vqvae2_paths, epochs=100):
    vq_commit = np.zeros((len(vqvae_paths), epochs))
    vq_mse = np.zeros((len(vqvae_paths), epochs))
    vq2_loss = np.zeros((len(vqvae2_paths), epochs))
    vq2_mse = np.zeros((len(vqvae2_paths), epochs))

    for i, path in enumerate(vqvae_paths):
        loss_path = os.path.join(path, "commitment/test")
        loss_events_path = os.path.join(loss_path, os.listdir(loss_path)[0])
        loss_ea = event_accumulator.EventAccumulator(loss_events_path)
        loss_ea.Reload()
        events = loss_ea.Scalars("commitment")
        for j, event in enumerate(events):
            vq_commit[i, j] = event.value
        
        mse_path = os.path.join(path, "mse/test")
        mse_events_path = os.path.join(mse_path, os.listdir(mse_path)[0])
        mse_ea = event_accumulator.EventAccumulator(mse_events_path)
        mse_ea.Reload()
        events = mse_ea.Scalars("mse")
        for j, event in enumerate(events):
            vq_mse[i, j] = event.value

    for path in vqvae2_paths:
        j = 0
        with open(os.path.join(path, "test_metrics.txt"), "r") as f:
            for line in f.readlines():
                metrics = line.split("; ")
                if len(metrics) == 3:
                    vq2_loss[i][j] = float(metrics[1][-7:])
                    vq2_mse[i][j] = float(metrics[2][-7:])
                    j += 1

    vq2_commit = (vq2_loss - vq2_mse) / 0.1  # TODO log commitment loss for vaevq2

    plt.plot(np.mean(vq_commit, axis=0), label="VQ-VAE", color="orange")
    plt.fill_between(list(range(epochs)), np.min(vq_commit, axis=0), np.max(vq_commit, axis=0), color="orange", alpha=.2)
    plt.plot(np.mean(vq2_commit, axis=0), label="VQ-VAE2", color="blue")
    plt.fill_between(list(range(epochs)), np.min(vq2_commit, axis=0), np.max(vq2_commit, axis=0), color="blue", alpha=.2)
    plt.legend()
    plt.title("Test Commitment Loss")
    plt.savefig("commit.png")

    plt.clf()
    plt.plot(np.mean(vq_mse, axis=0), label="VQ-VAE", color="orange")
    plt.fill_between(list(range(epochs)), np.min(vq_mse, axis=0), np.max(vq_mse, axis=0), color="orange", alpha=.2)
    plt.plot(np.mean(vq2_mse, axis=0), label="VQ-VAE2", color="blue")
    plt.fill_between(list(range(epochs)), np.min(vq2_mse, axis=0), np.max(vq2_mse, axis=0), color="blue", alpha=.2)
    plt.legend()
    plt.title("Test Reconstruction Loss")
    plt.savefig("recon.png")

    print("Best VQ-VAE:", vqvae_paths[np.argmin(vq_mse[:, -1])])
    print("Best VQ-VAE2:", vqvae2_paths[np.argmin(vq2_mse[:, -1])])
    print("VQ-VAE Commitment Loss:", np.mean(vq_commit, axis=0)[-1], np.std(vq_commit, axis=0)[-1])
    print("VQ-VAE Reconstruction Loss:", np.mean(vq_mse, axis=0)[-1], np.std(vq_mse, axis=0)[-1])
    print("VQ-VAE2 Commitment Loss:", np.mean(vq2_commit, axis=0)[-1], np.std(vq2_commit, axis=0)[-1])
    print("VQ-VAE2 Reconstruction Loss:", np.mean(vq2_mse, axis=0)[-1], np.std(vq2_mse, axis=0)[-1])


if __name__ == "__main__":
    plot_test_metrics(
        [
            "results/2021-12-04_23-51-34",
            "results/2021-12-05_01-06-50",
            "results/2021-12-05_02-22-10",
        ],
        [
            "results/2021-12-04_22-21-52_vqvae2",
            "results/2021-12-04_22-51-43_vqvae2",
            "results/2021-12-04_23-21-40_vqvae2",
        ],
    )
    graph_test_samples(
        "results/2021-12-05_02-22-10",
        "results/2021-12-04_22-21-52_vqvae2",
    )
