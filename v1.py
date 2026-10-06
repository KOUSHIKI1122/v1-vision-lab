"""V1 Vision Lab: what does the first stage of visual cortex "see"?

Simple cells in V1 act like Gabor filters tuned to edge orientation (Hubel & Wiesel).
We run a bank of 8 filters over a doodle and colour each pixel by its winning orientation.

Run:  python v1.py   ->  assets/v1_vision_lab.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve

BG, PANEL, INK, MUTE = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE,
    "ytick.color": MUTE, "axes.edgecolor": "#30363d", "font.size": 10,
})

N_ORI = 8


def gabor(theta, psi=0.0, size=25, lam=9.0, sigma=4.5, gamma=0.6):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam + psi)
    return g - g.mean()


def doodle(size=192):
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 100, 100], outline=255, width=4)
    d.rectangle([110, 30, 175, 90], outline=255, width=4)
    d.polygon([(30, 175), (65, 115), (100, 175)], outline=255, width=4)
    d.line([(115, 175), (175, 110)], fill=255, width=4)
    d.line([(125, 110), (125, 170)], fill=255, width=4)
    return np.asarray(img, dtype=float) / 255.0


def main():
    os.makedirs("assets", exist_ok=True)
    img = doodle()
    thetas = np.arange(N_ORI) * np.pi / N_ORI

    energy = []
    for th in thetas:  # quadrature pair -> phase-invariant "complex cell" energy
        e = fftconvolve(img, gabor(th, 0), mode="same") ** 2 + fftconvolve(img, gabor(th, np.pi / 2), mode="same") ** 2
        energy.append(np.sqrt(e))
    energy = np.array(energy)

    winner = energy.argmax(0)
    strength = energy.max(0)
    strength = (strength / strength.max()) ** 0.6
    # Filter axis is perpendicular to the edge it likes, so shift hue by 90 degrees.
    hue = ((thetas[winner] + np.pi / 2) % np.pi) / np.pi
    rgb = hsv_to_rgb(np.dstack([hue, np.full_like(hue, 0.85), strength]))

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.12)

    a0 = fig.add_subplot(gs[0])
    a0.imshow(img, cmap="gray")
    a0.set_title("what the eye sends in", loc="left")
    a0.axis("off")

    inner = gs[1].subgridspec(2, N_ORI // 2, hspace=0.05, wspace=0.05)
    for k, th in enumerate(thetas):
        a = fig.add_subplot(inner[k // (N_ORI // 2), k % (N_ORI // 2)])
        a.imshow(gabor(th), cmap="PuOr")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{int(np.degrees(th))}°", fontsize=8, color=MUTE)
    fig.text(0.52, 0.95, "a bank of 8 orientation-tuned 'neurons'", ha="center", color=INK)

    a2 = fig.add_subplot(gs[2])
    a2.imshow(rgb)
    a2.set_title("which orientation wins, per pixel", loc="left")
    a2.axis("off")

    bar = fig.add_axes([0.70, 0.07, 0.2, 0.03])
    bar.imshow(hsv_to_rgb(np.dstack([np.tile(np.linspace(0, 1, 256), (4, 1)), np.ones((4, 256)), np.ones((4, 256))])), aspect="auto")
    bar.set_yticks([])
    bar.set_xticks([0, 128, 255])
    bar.set_xticklabels(["0°", "90°", "180°"])

    fig.savefig("assets/v1_vision_lab.png", dpi=130, bbox_inches="tight")
    print("saved assets/v1_vision_lab.png")


if __name__ == "__main__":
    main()
