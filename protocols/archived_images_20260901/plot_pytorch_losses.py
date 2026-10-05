"""Aggregate and plot the current global-model validation loss."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import statistics

from PIL import Image, ImageColor, ImageDraw, ImageFont


METHOD_LABELS = {
    "critical": "Critical E",
    "critical-r1": "Closed-form optimal E (R=1)",
    "corrected-fixed": "Grid Search best E",
    "fedavg": "FedAvg",
    "original-radfed": "Original RADFed",
}


def _dataset_label(root: Path) -> str:
    normalized = root.name.strip().lower().replace("_", "").replace("-", "")
    labels = {
        "mnist": "MNIST",
        "cifar": "CIFAR-10",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "cifar100datacenter": "CIFAR-100 Datacenter",
        "cifar100edge": "CIFAR-100 Edge",
        "mnistdatacenter": "MNIST Datacenter",
        "mnistedge": "MNIST Edge",
        "cifar10datacenter": "CIFAR-10 Datacenter",
        "cifar10edge": "CIFAR-10 Edge",
        "covclsg": "COVCLS-G",
        "covclsl": "COVCLS-L",
        "covfeatg": "COVFEAT-G",
        "covfeatl": "COVFEAT-L",
        "shakespeare": "Shakespeare",
        "covclsgdatacenter": "COVCLS-G Datacenter",
        "covclsldatacenter": "COVCLS-L Datacenter",
        "covfeatgdatacenter": "COVFEAT-G Datacenter",
        "covfeatldatacenter": "COVFEAT-L Datacenter",
        "shakespearedatacenter": "Shakespeare Datacenter",
        "covclsgedge": "COVCLS-G Edge",
        "covclsledge": "COVCLS-L Edge",
        "covfeatgedge": "COVFEAT-G Edge",
        "covfeatledge": "COVFEAT-L Edge",
        "shakespeareedge": "Shakespeare Edge",
    }
    if normalized in labels:
        return labels[normalized]
    return root.name


def _dataset_prefix(root: Path) -> str:
    label = _dataset_label(root)
    prefixes = {
        "CIFAR-10": "cifar10",
        "CIFAR-100": "cifar100",
        "CIFAR-100 Datacenter": "cifar100_datacenter",
        "CIFAR-100 Edge": "cifar100_edge",
        "MNIST Datacenter": "mnist_datacenter",
        "MNIST Edge": "mnist_edge",
        "CIFAR-10 Datacenter": "cifar10_datacenter",
        "CIFAR-10 Edge": "cifar10_edge",
        "COVCLS-G": "covcls_g",
        "COVCLS-L": "covcls_l",
        "COVFEAT-G": "covfeat_g",
        "COVFEAT-L": "covfeat_l",
        "COVCLS-G Datacenter": "covcls_g_datacenter",
        "COVCLS-L Datacenter": "covcls_l_datacenter",
        "COVFEAT-G Datacenter": "covfeat_g_datacenter",
        "COVFEAT-L Datacenter": "covfeat_l_datacenter",
        "Shakespeare Datacenter": "shakespeare_datacenter",
        "COVCLS-G Edge": "covcls_g_edge",
        "COVCLS-L Edge": "covcls_l_edge",
        "COVFEAT-G Edge": "covfeat_g_edge",
        "COVFEAT-L Edge": "covfeat_l_edge",
        "Shakespeare Edge": "shakespeare_edge",
    }
    return prefixes.get(label, label.lower().replace(" ", "_"))


def _is_aggregation(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _initial_profile_seconds(metrics_path: Path) -> float:
    profile_path = metrics_path.with_name("profile_metrics.csv")
    if not profile_path.is_file():
        return 0.0
    with profile_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("round", "")).strip() == "0":
                raw_seconds = row.get("profile_seconds", "")
                return float(raw_seconds) if raw_seconds not in {None, ""} else 0.0
    return 0.0


def aggregate_validation_losses(dataset_root: str | Path) -> list[dict[str, object]]:
    root = Path(dataset_root)
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    # Paper-matrix outputs use method/fold_N/seed_N/round_metrics.csv, while a
    # single Nomad allocation extracts round_metrics.csv directly under root.
    # Search recursively and read the method from the CSV so both layouts work.
    for metrics_path in sorted(root.rglob("round_metrics.csv")):
        relative_parts = metrics_path.relative_to(root).parts
        if (
            "grid_search" in relative_parts
            or "critical_epsilon_search" in relative_parts
        ):
            continue
        legacy_initial_profile_seconds = _initial_profile_seconds(metrics_path)
        modeled_total_seconds = 0.0
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw_modeled_round = row.get("modeled_critical_round_seconds", "")
                if raw_modeled_round not in {None, ""}:
                    modeled_total_seconds += float(raw_modeled_round)
                if not _is_aggregation(row.get("is_aggregation", "")):
                    continue
                raw_loss = row.get("validation_loss", "")
                if raw_loss in {None, ""}:
                    continue
                if row.get("method"):
                    method = str(row["method"])
                elif (
                    metrics_path.parent.name.startswith("seed_")
                    and metrics_path.parent.parent.name.startswith("fold_")
                ):
                    method = metrics_path.parent.parent.parent.name
                else:
                    method = metrics_path.parent.name
                round_number = int(row.get("outer_round") or row["round"])
                loss = float(raw_loss)
                raw_time = row.get("wall_total_seconds", "")
                wall_time = float(raw_time) if raw_time not in {None, ""} else float(round_number)
                if not _is_aggregation(
                    row.get("wall_time_includes_initial_profile", "")
                ):
                    wall_time += legacy_initial_profile_seconds
                modeled_time = (
                    modeled_total_seconds
                    if raw_modeled_round not in {None, ""}
                    else wall_time
                )
                values = grouped.setdefault(
                    (method, round_number),
                    {"loss": [], "wall_time": [], "modeled_time": []},
                )
                values["loss"].append(loss)
                values["wall_time"].append(wall_time)
                values["modeled_time"].append(modeled_time)

    rows: list[dict[str, object]] = []
    for (method, round_number), values_by_measure in sorted(grouped.items()):
        losses = values_by_measure["loss"]
        wall_times = values_by_measure["wall_time"]
        modeled_times = values_by_measure["modeled_time"]
        rows.append(
            {
                "method": method,
                "aggregation_round": round_number,
                "runs": len(losses),
                "wall_total_seconds_mean": statistics.fmean(wall_times),
                "wall_total_seconds_sample_std": (
                    statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
                ),
                "modeled_total_seconds_mean": statistics.fmean(modeled_times),
                "modeled_total_seconds_sample_std": (
                    statistics.stdev(modeled_times)
                    if len(modeled_times) > 1
                    else 0.0
                ),
                "validation_loss_mean": statistics.fmean(losses),
                "validation_loss_sample_std": (
                    statistics.stdev(losses) if len(losses) > 1 else 0.0
                ),
            }
        )
    return rows


def write_loss_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no aggregated validation-loss rows were found")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_loss_svg(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    x_field: str = "aggregation_round",
    y_field: str = "validation_loss_mean",
    y_std_field: str = "validation_loss_sample_std",
    title: str = "Aggregated-model validation loss",
    x_label: str = "Aggregation round",
    y_label: str = "Client-balanced cross-entropy loss",
) -> Path:
    if not rows:
        raise ValueError("no aggregated validation-loss rows were found")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 620
    left, right, top, bottom = 82, 30, 52, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row[x_field]) for row in rows]
    upper_values = [
        float(row[y_field]) + float(row[y_std_field])
        for row in rows
    ]
    lower_values = [
        max(0.0, float(row[y_field]) - float(row[y_std_field]))
        for row in rows
    ]
    min_x = 1.0 if x_field == "aggregation_round" else 0.0
    max_x = max(x_values)
    raw_y_min = min(lower_values)
    raw_y_max = max(upper_values)
    raw_y_span = raw_y_max - raw_y_min
    y_padding = max(0.01, raw_y_span * 0.08, raw_y_max * 0.005)
    y_min = max(0.0, raw_y_min - y_padding)
    y_max = raw_y_max + y_padding

    def x_value(raw_x: float) -> float:
        if max_x <= min_x:
            return left + plot_width / 2
        return left + (raw_x - min_x) * plot_width / (max_x - min_x)

    def y_value(loss: float) -> float:
        bounded = min(y_max, max(y_min, loss))
        return top + plot_height * (1.0 - (bounded - y_min) / (y_max - y_min))

    methods = sorted({str(row["method"]) for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        loss = y_min + (y_max - y_min) * tick / 5
        y = y_value(loss)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>'
        )
        svg.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{loss:.2f}</text>'
        )
    tick_count = min(7, max(2, int(max_x - min_x + 1)))
    for index in range(tick_count):
        tick_value = min_x + index * (max_x - min_x) / (tick_count - 1)
        x = x_value(tick_value)
        tick_text = (
            str(round(tick_value))
            if x_field == "aggregation_round"
            else f"{tick_value:.0f}"
        )
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#eeeeee"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick_text}</text>'
        )
    svg.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#777777"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{top+plot_height/2:.2f}" transform="rotate(-90 20 {top+plot_height/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )

    for method_index, method in enumerate(methods):
        color = colors[method_index % len(colors)]
        display_method = METHOD_LABELS.get(method, method)
        selected = [row for row in rows if row["method"] == method]
        selected.sort(key=lambda row: float(row[x_field]))
        upper = [
            (x_value(float(row[x_field])), y_value(float(row[y_field]) + float(row[y_std_field])))
            for row in selected
        ]
        lower = [
            (x_value(float(row[x_field])), y_value(max(0.0, float(row[y_field]) - float(row[y_std_field]))))
            for row in reversed(selected)
        ]
        band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        points = " ".join(
            f'{x_value(float(row[x_field])):.2f},{y_value(float(row[y_field])):.2f}'
            for row in selected
        )
        svg.append(f'<polygon points="{band}" fill="{color}" opacity="0.14"/>')
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left + method_index * 205
        legend_y = height - 42
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{legend_x+32}" y="{legend_y+4}" font-family="sans-serif" font-size="12">{html.escape(display_method)}</text>'
        )
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")
    return output


def _plot_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        filename,
        str(Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")),
        str(
            Path("/usr/share/fonts/truetype/dejavu")
            / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
        str(
            Path("/usr/share/fonts/dejavu")
            / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def write_loss_jpg(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    x_field: str = "aggregation_round",
    y_field: str = "validation_loss_mean",
    y_std_field: str = "validation_loss_sample_std",
    title: str = "Aggregated-model validation loss",
    x_label: str = "Aggregation round",
    y_label: str = "Client-balanced cross-entropy loss",
) -> Path:
    """Write a high-resolution, white-background JPEG of a loss curve."""
    if not rows:
        raise ValueError("no aggregated validation-loss rows were found")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    scale = 2
    width, height = 1000, 700
    left, right, top, bottom = 108, 35, 76, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row[x_field]) for row in rows]
    upper_values = [
        float(row[y_field]) + float(row[y_std_field]) for row in rows
    ]
    lower_values = [
        max(0.0, float(row[y_field]) - float(row[y_std_field])) for row in rows
    ]
    min_x = 1.0 if x_field == "aggregation_round" else 0.0
    max_x = max(x_values)
    raw_y_min = min(lower_values)
    raw_y_max = max(upper_values)
    raw_y_span = raw_y_max - raw_y_min
    y_padding = max(0.01, raw_y_span * 0.08, raw_y_max * 0.005)
    y_min = max(0.0, raw_y_min - y_padding)
    y_max = raw_y_max + y_padding

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    def x_value(raw_x: float) -> float:
        if max_x <= min_x:
            return left + plot_width / 2
        return left + (raw_x - min_x) * plot_width / (max_x - min_x)

    def y_value(loss: float) -> float:
        bounded = min(y_max, max(y_min, loss))
        return top + plot_height * (1.0 - (bounded - y_min) / (y_max - y_min))

    image = Image.new("RGB", point(width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _plot_font(28 * scale, bold=True)
    axis_font = _plot_font(20 * scale)
    tick_font = _plot_font(16 * scale)
    legend_font = _plot_font(17 * scale)
    draw.text(point(width / 2, 36), title, fill="#111111", font=title_font, anchor="mm")

    for tick in range(6):
        loss = y_min + (y_max - y_min) * tick / 5
        y = y_value(loss)
        draw.line([point(left, y), point(width - right, y)], fill="#dddddd", width=scale)
        draw.text(
            point(left - 12, y),
            f"{loss:.2f}",
            fill="#222222",
            font=tick_font,
            anchor="rm",
        )
    tick_count = min(7, max(2, int(max_x - min_x + 1)))
    for index in range(tick_count):
        tick_value = min_x + index * (max_x - min_x) / (tick_count - 1)
        x = x_value(tick_value)
        tick_text = str(round(tick_value)) if x_field == "aggregation_round" else f"{tick_value:.0f}"
        draw.line([point(x, top), point(x, height - bottom)], fill="#eeeeee", width=scale)
        draw.text(
            point(x, height - bottom + 27),
            tick_text,
            fill="#222222",
            font=tick_font,
            anchor="mm",
        )

    draw.rectangle(
        [point(left, top), point(width - right, height - bottom)],
        outline="#777777",
        width=scale,
    )
    draw.text(
        point(left + plot_width / 2, height - bottom + 62),
        x_label,
        fill="#111111",
        font=axis_font,
        anchor="mm",
    )
    label_box = draw.textbbox((0, 0), y_label, font=axis_font)
    label_image = Image.new(
        "RGBA",
        (label_box[2] - label_box[0] + 12 * scale, label_box[3] - label_box[1] + 12 * scale),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(label_image).text(
        (6 * scale, 6 * scale), y_label, fill="#111111", font=axis_font
    )
    rotated_label = label_image.rotate(90, expand=True)
    image.paste(
        rotated_label,
        (
            round(28 * scale - rotated_label.width / 2),
            round((top + plot_height / 2) * scale - rotated_label.height / 2),
        ),
        rotated_label,
    )

    methods = sorted({str(row["method"]) for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    draw = ImageDraw.Draw(image, "RGBA")
    for method_index, method in enumerate(methods):
        color = colors[method_index % len(colors)]
        rgb = ImageColor.getrgb(color)
        display_method = METHOD_LABELS.get(method, method)
        selected = [row for row in rows if row["method"] == method]
        selected.sort(key=lambda row: float(row[x_field]))
        upper = [
            point(x_value(float(row[x_field])), y_value(float(row[y_field]) + float(row[y_std_field])))
            for row in selected
        ]
        lower = [
            point(
                x_value(float(row[x_field])),
                y_value(max(0.0, float(row[y_field]) - float(row[y_std_field]))),
            )
            for row in reversed(selected)
        ]
        if len(upper) + len(lower) >= 3:
            draw.polygon(upper + lower, fill=rgb + (36,))
        line_points = [
            point(x_value(float(row[x_field])), y_value(float(row[y_field])))
            for row in selected
        ]
        if len(line_points) == 1:
            x, y = line_points[0]
            radius = 3 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        else:
            draw.line(line_points, fill=color, width=3 * scale, joint="curve")
        legend_columns = min(3, len(methods))
        legend_column_width = plot_width / legend_columns
        legend_x = left + (method_index % legend_columns) * legend_column_width
        legend_y = height - 46 + (method_index // legend_columns) * 27
        draw.line(
            [point(legend_x, legend_y), point(legend_x + 25, legend_y)],
            fill=color,
            width=3 * scale,
        )
        draw.text(
            point(legend_x + 32, legend_y),
            display_method,
            fill="#111111",
            font=legend_font,
            anchor="lm",
        )

    image.save(output, format="JPEG", quality=95, subsampling=0, dpi=(200, 200))
    return output


def create_loss_outputs(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root)
    dataset_label = _dataset_label(root)
    dataset_prefix = _dataset_prefix(root)
    comparison_prefix = f"{dataset_prefix}_critical_e_vs_grid_search"
    rows = aggregate_validation_losses(root)
    csv_path = write_loss_csv(root / "aggregated_validation_loss.csv", rows)
    plot_specs = [
        (
            f"{comparison_prefix}_validation_loss_by_round",
            {"title": f"{dataset_label}: Validation loss versus aggregation round"},
        ),
        (
            f"{comparison_prefix}_validation_loss_by_time",
            {
                "x_field": "wall_total_seconds_mean",
                "title": f"{dataset_label}: Validation loss versus elapsed time",
                "x_label": "Elapsed wall time (seconds)",
            },
        ),
        (
            f"{comparison_prefix}_validation_loss_by_modeled_time",
            {
                "x_field": "modeled_total_seconds_mean",
                "title": f"{dataset_label}: Validation loss versus modeled time",
                "x_label": "Modeled elapsed time (seconds)",
            },
        ),
    ]
    for stem, options in plot_specs:
        write_loss_jpg(root / f"{stem}.jpg", rows, **options)
    primary_jpg_path = root / f"{comparison_prefix}_validation_loss_by_round.jpg"
    return csv_path, primary_jpg_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot global validation loss over folds and seeds."
    )
    parser.add_argument("--root", required=True, help="one dataset's experiment output root")
    args = parser.parse_args()
    csv_path, jpg_path = create_loss_outputs(args.root)
    dataset_prefix = _dataset_prefix(Path(args.root))
    comparison_prefix = f"{dataset_prefix}_critical_e_vs_grid_search"
    print(f"loss_csv={csv_path}")
    print(f"loss_jpg={jpg_path}")
    print(f"loss_by_round_jpg={Path(args.root) / f'{comparison_prefix}_validation_loss_by_round.jpg'}")
    print(f"loss_by_time_jpg={Path(args.root) / f'{comparison_prefix}_validation_loss_by_time.jpg'}")


if __name__ == "__main__":
    main()
