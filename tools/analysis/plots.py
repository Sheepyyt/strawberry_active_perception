"""Three readable figures from audited single-run measurements; no ROS imports."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


BLUE = "#2468a0"
GREEN = "#17866c"
ORANGE = "#cc792b"
RED = "#b14b4b"
INK = "#1e3345"


def _style() -> None:
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "sans-serif"
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False,
        "text.color": INK, "axes.labelcolor": INK,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def _labels(views: list[dict]) -> list[str]:
    return ["初始" if row["index"] == 0 else f"运动后{row['index']}" for row in views]


def _finish(fig, path: Path, note: str) -> Path:
    fig.text(.03, .025, note, color="#536579", fontsize=9, va="bottom")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def information_growth(analysis: dict, output: Path) -> Path:
    views = analysis["views"]
    x = np.arange(len(views))
    labels = _labels(views)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))
    fig.subplots_adjust(left=.065, right=.975, top=.81, bottom=.19, wspace=.35)
    fig.suptitle("多视角观察为地图增加了多少信息", fontsize=18, fontweight="bold", y=.96)
    coverage = np.array([row["coverage"] * 100 for row in views])
    axes[0].plot(x, coverage, "o-", color=BLUE, lw=2.3)
    target = analysis["summary"].get("coverage_target")
    if target is not None:
        axes[0].axhline(target * 100, color="#9aa8b4", ls="--", lw=1)
        axes[0].text(.03, target * 100 + 2, f"本轮停止目标 {target * 100:.0f}%", fontsize=8, color="#697987")
    axes[0].set_ylim(0, max(coverage.max() + 12, (target or 0) * 100 + 12))
    axes[0].set_title("ROI 射线覆盖率")
    axes[0].set_ylabel("覆盖率（%）")
    for i, value in enumerate(coverage):
        axes[0].annotate(f"{value:.2f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)

    counts = [row["target_voxels"] for row in views]
    axes[1].plot(x, counts, "s-", color=GREEN, lw=2.3)
    axes[1].set_ylim(0, max(counts) * 1.19)
    axes[1].set_title("地图内的目标支持体素")
    axes[1].set_ylabel("体素数")
    for i, value in enumerate(counts):
        axes[1].annotate(str(value), (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    added = np.array([row["target_added"] for row in views])
    removed = np.array([row["target_removed"] for row in views])
    axes[2].bar(x, added, width=.55, color=GREEN, label="新增为目标")
    axes[2].bar(x, -removed, width=.55, color=ORANGE, label="不再标作目标")
    axes[2].axhline(0, color="#748493", lw=.8)
    axes[2].set_title("相邻地图的目标体素变化")
    axes[2].set_ylabel("体素数（集合逐项比较）")
    axes[2].legend(fontsize=8, loc="upper right", frameon=False)
    axes[2].set_ylim(-max(removed.max() * 1.5, 15), max(added.max() * 1.45, 20))
    for i in range(1, len(views)):
        axes[2].text(i, added[i] + 2, f"+{added[i]}", ha="center", va="bottom", fontsize=8)
        label = f"−{removed[i]}" if removed[i] else "0"
        axes[2].text(i, -removed[i] - 2, label, ha="center", va="top", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    return _finish(fig, output, "初始地图作为比较基线；体素新增/消失反映地图分类变化，不能直接解释为草莓表面完整度。")


def _observation_crop(view: dict, depth_range: tuple[float, float]):
    with np.load(view["observation_path"], allow_pickle=False) as data:
        rgb = data["rgb"].copy()
        mask = data["mask"] > 0
        depth = data["depth_m"]
        valid = mask & np.isfinite(depth) & (depth >= depth_range[0]) & (depth <= depth_range[1])
        invalid = mask & ~valid
    y, x = np.nonzero(mask)
    if len(x):
        margin = max(12, int(.55 * max(np.ptp(x) + 1, np.ptp(y) + 1)))
        bounds = (slice(max(0, y.min() - margin), min(rgb.shape[0], y.max() + margin + 1)),
                  slice(max(0, x.min() - margin), min(rgb.shape[1], x.max() + margin + 1)))
    else:
        bounds = (slice(None), slice(None))
    crop = rgb[bounds].astype(float)
    for selected, color in ((valid[bounds], (42, 190, 135)), (invalid[bounds], (245, 134, 40))):
        crop[selected] = .58 * crop[selected] + .42 * np.array(color)
    return np.clip(crop, 0, 255).astype(np.uint8)


def depth_quality(analysis: dict, output: Path) -> Path:
    views = analysis["views"]
    labels = _labels(views)
    fig = plt.figure(figsize=(12, 6.1))
    grid = fig.add_gridspec(2, 2, left=.07, right=.97, top=.83, bottom=.17, hspace=.58, wspace=.25,
                          height_ratios=(1, 1.1))
    crops = grid[0, :].subgridspec(1, len(views), wspace=.18)
    fig.suptitle("草莓分割与深度质量随视角变化", fontsize=18, fontweight="bold", y=.97)
    depth_range = tuple(analysis.get("depth_range_m", (.2, 2.5)))
    for index, view in enumerate(views):
        axis = fig.add_subplot(crops[0, index])
        axis.imshow(_observation_crop(view, depth_range), interpolation="nearest")
        axis.set_title(f"{labels[index]} · {view['mask_pixels']} 像素", fontsize=10)
        axis.set_axis_off()
    fig.legend(handles=[Patch(facecolor="#2abe87", label="掩膜内有效深度"),
                        Patch(facecolor="#f58628", label="掩膜内无效深度")],
               loc="upper center", bbox_to_anchor=(.5, .93), ncol=2, frameon=False, fontsize=9)
    ratio_axis, depth_axis = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    x = np.arange(len(views))
    ratios = [row["valid_depth_fraction"] * 100 for row in views]
    ratio_axis.plot(x, ratios, "o-", color=BLUE, lw=2)
    ratio_axis.set_ylim(max(0, min(ratios) - 12), 102)
    ratio_axis.set_title("掩膜内有效深度比例")
    ratio_axis.set_ylabel("比例（%）")
    for i, value in enumerate(ratios):
        ratio_axis.annotate(f"{value:.2f}%", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    depths = [row["median_depth_mm"] for row in views]
    mads = [row["depth_mad_mm"] for row in views]
    depth_axis.errorbar(x, depths, yerr=mads, fmt="o-", color=ORANGE, lw=2, capsize=4)
    depth_axis.set_ylim(min(depths) - 18, max(depths) + 28)
    depth_axis.set_title("定位所用近层深度：中位数 ± MAD")
    depth_axis.set_ylabel("光轴深度（mm）")
    for i, value in enumerate(depths):
        depth_axis.annotate(f"{value:.0f}", (i, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    for axis in (ratio_axis, depth_axis):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=.18)
    return _finish(fig, output, "上方为各视角的目标局部裁剪；有效深度比例不是分割准确率。MAD 为中位绝对偏差，不是置信区间。")


def candidate_analysis(analysis: dict, output: Path) -> Path:
    rounds = analysis["rounds"]
    radii = [5, 10, 25, 50, 75, 100]
    labels = ["首步\n冻结预演" if row["round_index"] == 1 else
              ("终止\n选点" if row["selected_radius_mm"] is None else f"第{row['round_index']}步\n选点") for row in rounds]
    positive = np.array([[next(r["positive"] for r in item["radii"] if abs(r["radius_mm"]-radius) < 1e-6)
                          for item in rounds] for radius in radii])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": (1, 1.2)})
    fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.22, wspace=.4)
    fig.suptitle("冻结状态下的候选半径与距离门限分析", fontsize=18, fontweight="bold", y=.96)
    fig.text(.5, .875, "每一列对应已记录的选点状态；参数筛选不代表另一条实际执行轨迹", ha="center", fontsize=10, color="#536579")
    image = axes[0].imshow(positive, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, positive.max()))
    for i in range(len(radii)):
        for j in range(len(rounds)):
            axes[0].text(j, i, str(positive[i, j]), ha="center", va="center", fontsize=11,
                         color="white" if positive[i, j] > positive.max() * .6 else INK)
    axes[0].set_xticks(range(len(rounds)), labels)
    axes[0].set_yticks(range(len(radii)), radii)
    axes[0].set_ylabel("候选半径（mm）")
    axes[0].set_title("原门限下各半径的正收益候选数")
    fig.colorbar(image, ax=axes[0], fraction=.046, pad=.04).set_label("候选数")
    thresholds = [.42, .43, .45]
    width = .24
    x = np.arange(len(rounds))
    for index, (threshold, color) in enumerate(zip(thresholds, (BLUE, GREEN, ORANGE))):
        values = [next(r["positive"] for r in row["thresholds"] if abs(r["minimum_depth_m"]-threshold) < 1e-8) for row in rounds]
        bars = axes[1].bar(x + (index-1)*width, values, width, color=color, label=f"≥ {threshold:.2f} m")
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + width/2, value + .35, str(value), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("保留的正收益候选数")
    axes[1].set_title("仅收紧目标光轴深度门限")
    ymax = max(row["positive"] for item in rounds for row in item["thresholds"])
    axes[1].set_ylim(0, max(5, ymax*1.3))
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=.15)
    axes[1].set_axisbelow(True)
    return _finish(fig, output, "正收益按原程序容差判断；未评分候选不视为零收益。首步来自绑定预演，其余来自实际执行与终止记录。")


def render_figures(analysis: dict, output: Path) -> list[Path]:
    _style()
    output.mkdir(parents=True, exist_ok=True)
    return [information_growth(analysis, output / "01_多视角信息增量.png"),
            depth_quality(analysis, output / "02_分割与深度质量.png"),
            candidate_analysis(analysis, output / "03_候选选点分析.png")]
