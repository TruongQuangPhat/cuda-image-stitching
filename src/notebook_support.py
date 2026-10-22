# Tác giả: HUỲNH TẤN PHƯỚC (23120334), tích hợp notebook; nhóm 23120318/23120329/23120334.
# Điều phối thực nghiệm và thống kê; S1–S6 được cài đặt trong C/C++.
"""Biên dịch, đo và trình bày kết quả thực nghiệm V0 trong notebook."""
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import stat
import statistics
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from IPython import get_ipython
from IPython.display import display, HTML, Image as DisplayImage
from IPython.utils.capture import capture_output
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path.cwd().resolve()
GROUP_ID = "Group09"
DATASET_NAME = "ImageStitching_Team_Dataset.zip"
DATASET_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=11Nbg77zkOizdNcY5vEmlybP60KTv5NUk&export=download&confirm=t"
)
DATASET_SHA256 = "2097dc6eeb3d339731541f94df0d3f44694b447bd6f5b9c66699559ac7269827"
RUNS, WARMUPS = 3, 1
MODES = ["feather", "seam"]
STAGES = [f"s{i}_ms" for i in range(1,7)]
TIMINGS = ["preprocess_ms", *STAGES, "total_ms"]
SIZES = ["small", "medium", "large"]
SIZE_NAMES = {"small":"Nhỏ (400 px)", "medium":"Vừa (800 px)", "large":"Lớn (1.200 px)"}
PAIR_NAMES = {"cmu0_standard":"CMU0 — chuẩn", "cmu1_standard":"CMU1 — chuẩn",
              "cmu2_standard":"CMU2 — chuẩn", "cmu0_challenging":"CMU0 — khó"}
def extract_safe(archive, destination, allowed=None):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP bị lỗi CRC")
        for entry in bundle.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Đường dẫn không an toàn trong ZIP")
            if allowed is not None and not allowed(entry.filename):
                raise ValueError("ZIP chứa tệp ngoài cấu trúc dự án: " + entry.filename)
        bundle.extractall(destination, members=[name for name in bundle.namelist()
            if name.startswith(("src/", "dataset/")) or name.endswith(".ipynb")])

def source_hashes(root):
    paths = sorted(root.joinpath("src").glob("*.*"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}

class Runner:
    def __init__(self, root=ROOT, backend="auto", distro=""):
        self.root = Path(root).resolve()
        self.backend = ("wsl" if os.name == "nt" else "native") if backend == "auto" else backend
        self.prefix = []
        if self.backend == "wsl":
            drive = self.root.drive.rstrip(":").lower()
            if not drive or self.root.drive.startswith("\\\\"):
                raise ValueError("WSL runner requires a local Windows drive")
            linux_root = "/mnt/" + drive + self.root.as_posix()[2:]
            self.prefix = ["wsl"] + (["-d", distro] if distro else []) + ["--cd", linux_root, "--"]
            self.compiler = "g++"
        else:
            self.compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
            if not self.compiler:
                raise RuntimeError("Install a C++17 compiler (g++ or clang++)")

    def command(self, args, log=None):
        result = subprocess.run(self.prefix + list(map(str, args)), cwd=self.root,
                                text=True, encoding="utf-8", errors="replace", capture_output=True)
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {args}\n"
                               + result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def build(self):
        (self.root / "build").mkdir(exist_ok=True)
        flags = ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-O2", "-Isrc"]
        self.command([self.compiler, *flags, "-DSTITCHING_NO_MAIN", "-x", "c++",
                      "src/main.cu", "src/stb_impl.cpp", "src/v0_stitching.cpp", "src/orb.cpp",
                      "src/matching.cpp", "src/homography.cpp", "src/warping.cpp", "src/blending.cpp",
                      "-o", "build/v0_stitching"], self.root / "build/v0_build.log")

    def pair(self, a, b, folder, mode="feather", threshold=3.0, truth=None, name="metrics"):
        folder.mkdir(parents=True, exist_ok=True)
        metrics = folder / (name + ".json")
        metrics.unlink(missing_ok=True)
        args = ["./build/v0_stitching", self.relative(a), self.relative(b),
                self.relative(folder / "panorama.ppm"), self.relative(metrics), mode, str(threshold)]
        if truth:
            args.append(self.relative(truth))
        self.command(args, folder / (name + ".log"))
        data = json.loads(metrics.read_text(encoding="utf-8"))
        if not data.get("success") or any(not isinstance(data.get(key), (float, int))
                or not math.isfinite(data[key]) or data[key] < 0 for key in TIMINGS):
            raise RuntimeError("Missing timing/correctness fields in C++ output")
        return data

def synthetic(runner, output):
    folder = Path(output) / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(14120)
    size = (400, 300)
    a = Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(400*300*3)))
    a = a.filter(ImageFilter.GaussianBlur(0.8))
    draw = ImageDraw.Draw(a)
    for _ in range(100):
        x, y = rng.randrange(380), rng.randrange(280)
        draw.rectangle((x,y,x+12,y+12), fill=tuple(rng.randrange(256) for _ in range(3)))
    a.save(folder / "a.ppm")
    a.crop((30,20,430,320)).save(folder / "b.ppm")
    truth = folder / "H_ground_truth.txt"
    truth.write_text("1 0 30\n0 1 20\n0 0 1\n", encoding="ascii")
    data = runner.pair(folder / "a.ppm", folder / "b.ppm", folder, truth=truth)
    if not data["holdout_mean_error_px"] < 3.0:
        raise AssertionError(f"Synthetic mean holdout error exceeds 3 px: {data}")
    with Image.open(folder / "panorama.ppm") as image:
        image.save(folder / "panorama.png")
    print(f"Ảnh tổng hợp: mean={data['holdout_mean_error_px']:.4f} px, "
          f"max={data['holdout_max_error_px']:.4f} px", flush=True)
    cases = {"translation": data}
    angle = math.radians(5)
    transforms = {
        "rotation": [math.cos(angle),-math.sin(angle),18, math.sin(angle),math.cos(angle),-12, 0,0,1],
        "scale": [0.92,0,18, 0,0.92,12, 0,0,1],
        "perspective": [1,0.025,12, -0.015,1,8, 0.00008,-0.00005,1],
    }
    for name, matrix in transforms.items():
        case = folder/name
        case.mkdir(exist_ok=True)
        a.save(case/"a.ppm")
        # Pillow's output->input transform is exactly the known B->A homography.
        a.transform(size,Image.Transform.PERSPECTIVE,matrix[:8],
                    resample=Image.Resampling.BICUBIC).save(case/"b.ppm")
        truth = case/"H_ground_truth.txt"
        truth.write_text(" ".join(map(str,matrix)),encoding="ascii")
        result = runner.pair(case/"a.ppm",case/"b.ppm",case,truth=truth)
        if not result["holdout_mean_error_px"] < 3 or not result["holdout_max_error_px"] < 6:
            raise AssertionError(f"Synthetic {name} projection error exceeds mean 3 / max 6 px")
        with Image.open(case/"panorama.ppm") as image:
            image.save(case/"panorama.png")
        cases[name] = result
        print(f"Ảnh tổng hợp {name}: mean={result['holdout_mean_error_px']:.4f}, "
              f"max={result['holdout_max_error_px']:.4f} px",flush=True)
    (folder/"suite.json").write_text(json.dumps(cases,indent=2),encoding="utf-8")
    return data

def environment(runner, output, runs, warmups, modes):
    with (runner.root/"dataset/metadata.csv").open(encoding="utf-8",newline="") as stream:
        dataset_paths = {runner.root/"dataset"/row[key] for row in csv.DictReader(stream)
                         for key in ("prepared_a","prepared_b")}
    data = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": sys.version, "host_processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "backend": runner.backend, "compiler": runner.command([runner.compiler,"--version"]).splitlines()[0],
            "runs": runs, "warmups": warmups, "modes": modes, "max_features": 2000,
            "ransac_iterations": 2000, "reference_width": 800, "reference_threshold_px": 3.0,
            "compile_flags": "-std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 -Isrc -DSTITCHING_NO_MAIN -x c++",
            "starter_status": "team-authored placeholder adapter; not the official starter",
            "dataset_course_approval_verified": False,
            "seam_comparison": {"feather_radius_px":8,
                                "exposure_gains":"RGB overlap-mean intensity odds ratio, clamped [0.5,2]; preserves black/white"},
            "timing_scope": "RGB-to-gray preprocessing + S1-S6; excludes file I/O, image decoding, process startup",
            "s1_s2_scope": "sum of work for both input images", "clock": "std::chrono::steady_clock",
            "source_sha256": source_hashes(runner.root),
            "dataset_sha256": {f.relative_to(runner.root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(dataset_paths)}}
    try:
        data["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=runner.root,text=True,
                                                     stderr=subprocess.DEVNULL).strip()
        data["git_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"],cwd=runner.root,text=True,
                                                         stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        data["git_commit"] = "bundle without .git; use source_sha256"
    if runner.backend == "wsl" or sys.platform.startswith("linux"):
        data["cpu_model"] = runner.command(["sh","-c","lscpu | head -25"])
    (Path(output) / "environment.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    return data

def profile(runner, output, runs=3, warmups=1, modes=("feather", "seam")):
    if runs < 1 or warmups < 0 or not modes or len(set(modes)) != len(modes) or any(m not in ("feather","seam") for m in modes):
        raise ValueError("Invalid repetition counts or modes")
    output = Path(output).resolve()
    runner.relative(output)
    output.mkdir(parents=True, exist_ok=True)
    environment(runner,output,runs,warmups,list(modes))
    with (runner.root / "dataset/metadata.csv").open(encoding="utf-8",newline="") as stream:
        rows = list(csv.DictReader(stream))
    records, summary = [], []
    for row in rows:
        pair_id, resolution = row["pair_id"], row["resolution"]
        fixture = output / "inputs" / pair_id / resolution
        fixture.mkdir(parents=True,exist_ok=True)
        for side in ("a","b"):
            with Image.open(runner.root / "dataset" / row["prepared_"+side]) as image:
                image.convert("RGB").save(fixture / (side + ".ppm"))
        # A 3-pixel threshold at width 800 is scaled with the actual resize ratio.
        threshold = 3.0 * int(row["target_width"]) / 800.0
        for mode in modes:
            folder = output / pair_id / resolution / mode
            measured = []
            for iteration in range(warmups+runs):
                name = f"warmup_{iteration+1}" if iteration < warmups else f"run_{iteration-warmups+1}"
                data = runner.pair(fixture / "a.ppm",fixture / "b.ppm",folder,mode,threshold,name=name)
                if iteration < warmups:
                    continue
                base = {key: row[key] for key in ("pair_id","site","pair_type","resolution")}
                record = {**base, "mode": mode, "run": iteration-warmups+1,
                          **{k:v for k,v in data.items() if k not in ("H_B_to_A","exposure_gains")}}
                records.append(record); measured.append(record)
            with Image.open(folder / "panorama.ppm") as image:
                image.save(folder / "panorama.png")
            (folder / "panorama.ppm").unlink()
            item = {**base, "mode": mode, "runs": runs,
                    "seam_applied": all(r["seam_applied"] for r in measured)}
            for key in TIMINGS:
                values = [r[key] for r in measured]
                item[key] = statistics.mean(values)
                item[key.replace("_ms","_std_ms")] = statistics.stdev(values) if len(values)>1 else 0.0
            item["bottleneck"] = max(STAGES,key=lambda k:item[k])
            item["gpu_candidate_fraction"] = sum(item[k] for k in ("s3_ms","s5_ms","s6_ms"))/item["total_ms"]
            item["amdahl_limit"] = 1.0/(1.0-item["gpu_candidate_fraction"])
            summary.append(item)
            print(f"{pair_id}/{resolution}/{mode}: {item['total_ms']:.2f} ms "
                  f"(std {item['total_std_ms']:.2f}), đã dùng seam={item['seam_applied']}",flush=True)
        # Preserve incremental results if a later pair fails.
        for name, data in (("timings.csv",records),("summary.csv",summary)):
            with (output/name).open("w",encoding="utf-8",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    return summary

def _record(cell_id, captured):
    shell = get_ipython()
    cell = next(c for c in _report["cells"] if c["id"] == cell_id)
    source = shell.history_manager.input_hist_raw[-1]
    if cell_id not in source:
        raise RuntimeError("Không xác định được cell vừa chạy")
    outputs = [{"output_type":"stream", "name":name, "text":value}
               for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)] if value]
    for out in captured.outputs:
        data = {k:v for k,v in out.data.items() if k != "application/vnd.jupyter.widget-view+json"}
        if data:
            outputs.append({"output_type":"display_data", "data":data, "metadata":out.metadata})
    cell.update(source=source, execution_count=shell.execution_count, outputs=outputs)
    json.dumps(cell, ensure_ascii=False, allow_nan=False)

def _export_bundle():
    if not all(c["execution_count"] is not None for c in _report["cells"] if c["cell_type"] == "code"):
        raise RuntimeError("Chưa chạy đủ cell; hãy Run all")
    _report["metadata"]["lan_chay"] = {
        "utc": datetime.now(timezone.utc).isoformat(), "google_colab": IN_COLAB,
        "nguon_sha256": source_hashes(ROOT), "so_luot_do": len(timings)}
    report_path = ROOT / "Group09_Midterm.ipynb"
    report_path.write_text(json.dumps(_report, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    archive = ROOT / (GROUP_ID + "_Midterm.zip")
    temporary = archive.with_suffix(".tmp")
    paths = [report_path, *sorted((ROOT / "src").glob("*.*"))]
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.is_file():
                bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {p.relative_to(ROOT).as_posix() for p in paths if p.is_file()}
    temporary.replace(archive)
    print("Đã tạo ZIP kết quả:", archive.name)
    if IN_COLAB:
        files.download(str(archive))
    return archive

def configure(root):
    global ROOT, OUT, runner, _report, IN_COLAB, files
    ROOT = Path(root).resolve()
    os.chdir(ROOT)
    try:
        from google.colab import files
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    OUT = ROOT / "output/v0"
    OUT.mkdir(parents=True, exist_ok=True)
    runner = Runner(ROOT)
    _report = json.loads((ROOT/"Group09_Midterm.ipynb").read_text(encoding="utf-8"))
    for cell in _report["cells"]:
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])

def run(cell_id, action):
    # Widget tải dataset cần hiển thị ngay, không chờ hàm chuẩn bị kết thúc.
    with capture_output(display=(cell_id != "setup")) as captured:
        try:
            action()
        except Exception:
            captured.show()
            raise
    captured.show()
    _record(cell_id, captured)
    if cell_id == "export":
        _export_bundle()

def unpack_dataset(archive):
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("SHA-256 dataset không khớp; cần dùng đúng phiên bản ZIP ở mục Dữ liệu")
    extract_safe(archive,ROOT,lambda name:name.startswith("dataset/") or name in ("README.txt","DATASET_SHA256.json"))
    if not (ROOT/"dataset/metadata.csv").is_file():
        raise ValueError("Dataset thiếu dataset/metadata.csv")
    print("Dataset: SHA-256 của ZIP hợp lệ.")

def prepare():
    candidates = [ROOT/DATASET_NAME, ROOT.parent/DATASET_NAME]
    if not (ROOT/"dataset/metadata.csv").is_file():
        archive = next((p for p in candidates if p.is_file()),None)
        if archive is not None:
            unpack_dataset(archive)
        else:
            downloaded = ROOT/"dataset_download.zip"
            try:
                with urllib.request.urlopen(DATASET_URL,timeout=45) as response, downloaded.open("wb") as target:
                    shutil.copyfileobj(response,target)
                unpack_dataset(downloaded)
            except (OSError,ValueError,zipfile.BadZipFile):
                print("Chưa tải được dataset tự động; cần chọn ZIP ở phần Dữ liệu.")
            finally:
                downloaded.unlink(missing_ok=True)
    runner.build()
    compiler = runner.command([runner.compiler,"--version"]).splitlines()[0]
    print("V0: biên dịch C++17 hoàn tất, sử dụng một luồng CPU.")
    print(f"Môi trường: {platform.system()} / {runner.backend}; {compiler}.")

def dataset():
    global metadata
    import pandas as pd
    metadata = pd.read_csv(ROOT/"dataset/metadata.csv")
    assert len(metadata)==12 and metadata.site.nunique()==3
    assert set(metadata.resolution)==set(SIZES)
    for column in ["prepared_a","prepared_b"]:
        assert all((ROOT/"dataset"/p).is_file() for p in metadata[column])
    rows = [{"Cặp ảnh":PAIR_NAMES[row.pair_id],
             "Ảnh A":"dataset/"+row.source_a,"Ảnh B":"dataset/"+row.source_b,
             "Kích thước gốc A/B":row.source_size_a+" / "+row.source_size_b}
            for row in metadata.drop_duplicates("pair_id").itertuples()]
    show_table(pd.DataFrame(rows))
    assert (metadata.prepared_size_a == metadata.prepared_size_b).all()
    sizes = metadata.drop_duplicates(["site","resolution"]).pivot(
        index="site",columns="resolution",values="prepared_size_a").reindex(columns=SIZES)
    show_table(sizes.reset_index().rename(columns={"site":"Bộ ảnh",**SIZE_NAMES}))

def ensure_dataset():
    if (ROOT/"dataset/metadata.csv").is_file():
        return
    candidates = [ROOT/DATASET_NAME,ROOT.parent/DATASET_NAME]
    archive = next((path for path in candidates if path.is_file()),None)
    if archive is None and IN_COLAB:
        print("Cần dữ liệu đầu vào: chọn ImageStitching_Team_Dataset.zip từ liên kết ở mục Dữ liệu.")
        uploaded = files.upload()
        choices = [Path(name) for name in uploaded if name.lower().endswith(".zip")]
        if len(choices)!=1:
            raise ValueError("Chọn đúng một ZIP dataset")
        archive = choices[0]
    if archive is None:
        raise FileNotFoundError("Thiếu ImageStitching_Team_Dataset.zip trong thư mục dự án hoặc thư mục cha")
    unpack_dataset(archive)

def show_table(frame, digits=2):
    table = frame.to_html(index=False, border=0, float_format=lambda value:f"{value:.{digits}f}")
    display(HTML('<div style="overflow-x:auto">'+table+'</div>'))

def correctness():
    global synthetic_suite
    import pandas as pd
    with capture_output() as logs:
        synthetic(runner,OUT)
    (OUT/"synthetic/run.log").write_text(logs.stdout+logs.stderr,encoding="utf-8")
    synthetic_suite = json.loads((OUT/"synthetic/suite.json").read_text())
    names = {"translation":"Tịnh tiến", "rotation":"Xoay", "scale":"Đổi tỷ lệ", "perspective":"Phối cảnh"}
    assert set(synthetic_suite)==set(names)
    assert all(r["holdout_mean_error_px"]<3 and r["holdout_max_error_px"]<6 for r in synthetic_suite.values())
    frame = pd.DataFrame([{"Phép biến đổi":names[name],"Sai số TB (px)":r["holdout_mean_error_px"],
                          "Sai số lớn nhất (px)":r["holdout_max_error_px"]} for name,r in synthetic_suite.items()])
    show_table(frame,3)
    worst = frame["Sai số lớn nhất (px)"].max()
    print(f"Đạt 4/4 phép thử. Sai số lớn nhất: {worst:.2f} px; chưa kết luận độ đúng của mọi ảnh thực tế.")

def profiling():
    global timings, summary, baseline, stage_means, amdahl
    import pandas as pd
    with capture_output() as logs:
        profile(runner,OUT,RUNS,WARMUPS,MODES)
    (OUT/"profiling.log").write_text(logs.stdout+logs.stderr,encoding="utf-8")
    timings = pd.read_csv(OUT/"timings.csv")
    summary = pd.read_csv(OUT/"summary.csv")
    env = json.loads((OUT/"environment.json").read_text(encoding="utf-8"))
    cpu = next((line.split(":",1)[1].strip() for line in env.get("cpu_model","").splitlines()
                if line.strip().lower().startswith("model name:")),env["host_processor"])
    show_table(pd.DataFrame([
        {"Thông tin môi trường":"CPU","Giá trị":cpu},
        {"Thông tin môi trường":"Nền tảng / backend","Giá trị":env["platform"]+" / "+env["backend"]},
        {"Thông tin môi trường":"Trình biên dịch","Giá trị":env["compiler"]},
        {"Thông tin môi trường":"Cờ biên dịch","Giá trị":env["compile_flags"]},
        {"Thông tin môi trường":"Lặp đo / làm nóng mỗi cấu hình","Giá trị":f'{env["runs"]} / {env["warmups"]}'},
    ]))
    assert len(timings)==72 and len(summary)==24 and timings.success.all()
    assert (timings[["preprocess_ms",*STAGES]].sum(axis=1)-timings.total_ms).abs().max()<0.01
    baseline = summary[summary["mode"]=="feather"]
    stage_means = baseline.groupby("resolution")[[*STAGES,"total_ms"]].mean().reindex(SIZES)
    table = stage_means.rename(columns={**{f"s{i}_ms":f"S{i}" for i in range(1,7)},"total_ms":"Tổng"}).copy()
    table.insert(0,"Kích thước",[SIZE_NAMES[s] for s in table.index])
    table["Chậm nhất"] = stage_means[STAGES].idxmax(axis=1).str[:2].str.upper()
    show_table(table,1)
    print("Đơn vị: ms. Mỗi dòng là trung bình của bốn cặp ảnh; baseline feather.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,3.5))
    width = 0.24
    for offset,size in zip((-width,0,width),SIZES):
        ax.bar([i+offset for i in range(6)],stage_means.loc[size,STAGES],width,
               label=SIZE_NAMES[size].split(" (")[0])
    ax.set_xticks(range(6),[f"S{i}" for i in range(1,7)])
    ax.set_ylabel("Thời gian (ms)")
    ax.set_title("Thời gian trung bình từng giai đoạn — feather")
    ax.grid(axis="y",alpha=0.2)
    ax.legend(frameon=False,ncols=3)
    fig.tight_layout()
    fig.savefig(OUT/"stage_timing.png",dpi=150)
    display(fig)
    plt.close(fig)
    for size,row in stage_means.iterrows():
        hot = row[STAGES].idxmax()
        print(f"• {SIZE_NAMES[size]}: {hot[:2].upper()} chiếm {100*row[hot]/row.total_ms:.1f}% tổng thời gian.")
    display(HTML('<details><summary>Chi tiết từng cặp, từng chế độ (TB ± độ lệch chuẩn)</summary>'+
        summary[["pair_id","resolution","mode",*STAGES,"total_ms","total_std_ms"]].round(2).rename(columns={
            "pair_id":"Cặp ảnh","resolution":"Kích thước","mode":"Cách trộn","total_ms":"Tổng (ms)",
            "total_std_ms":"ĐLC tổng (ms)"}).to_html(index=False)+'<p>Mỗi cấu hình: 3 lần đo sau 1 lần làm nóng.</p></details>'))

def panoramas():
    import pandas as pd
    pairs = metadata.pair_id.drop_duplicates().tolist()
    sheet = Image.new("RGB",(1200,680),"white")
    draw = ImageDraw.Draw(sheet)
    rows = []
    # Bảng ảnh 2×2: mỗi site có panorama, không lặp lại cả ba độ phân giải.
    for index,pair in enumerate(pairs):
        x,y = (index%2)*600,(index//2)*340
        with Image.open(OUT/pair/"medium/feather/panorama.png") as image:
            image.thumbnail((590,300))
            sheet.paste(image,(x+(600-image.width)//2,y+30+(300-image.height)//2))
        label = pair.replace("_standard"," / standard").replace("_challenging"," / challenging").upper()
        draw.text((x+8,y+8),label,fill="black")
        result = json.loads((OUT/pair/"medium/feather/run_1.json").read_text())
        rows.append({"Cặp ảnh":PAIR_NAMES[pair],"Cặp khớp":result["matches"],"Inlier":result["inliers"],
                     "Tỷ lệ inlier":f'{100*result["inliers"]/result["matches"]:.1f}%',
                     "Sai số inlier (px)":result["mean_reprojection_px"]})
    sheet.save(OUT/"panoramas.png")
    display(DisplayImage(filename=str(OUT/"panoramas.png"),width=1100))
    show_table(pd.DataFrame(rows))
    print("Ảnh ở chiều rộng đầu vào 800 px. Sai số inlier chỉ đo trên các cặp được RANSAC giữ lại, không phải sai số so với ảnh chuẩn.")

def comparison():
    import pandas as pd
    from io import BytesIO
    import base64
    import html
    table = summary.groupby(["resolution","mode"]).total_ms.mean().unstack().reindex(SIZES)
    table.columns = ["Feather (ms)" if c=="feather" else "Seam (ms)" for c in table.columns]
    table.insert(0,"Kích thước",[SIZE_NAMES[s] for s in table.index])
    show_table(table,1)
    pictures = []
    for mode in MODES:
        with Image.open(OUT/"cmu0_challenging/medium"/mode/"panorama.png") as image:
            image.thumbnail((1000,650))
            stream = BytesIO(); image.save(stream,format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        pictures.append(f'<div style="flex:1;min-width:260px"><b>{html.escape(mode)}</b><br><img style="width:100%" src="data:image/png;base64,{encoded}"></div>')
    display(HTML('<details><summary>Xem feather và seam trên cặp khó</summary><div style="display:flex;gap:12px;flex-wrap:wrap">'+''.join(pictures)+'</div></details>'))
    fallbacks = int((~summary[summary["mode"]=="seam"].seam_applied).sum())
    print(f"Seam không áp dụng được ở {fallbacks}/12 cấu hình (các trường hợp đó dùng feather).")
    print("Chọn feather làm mốc CPU/GPU. Seam là phép so sánh bổ sung; vẫn có thể còn bóng đôi do chuyển động hoặc parallax.")

def gpu_plan():
    import pandas as pd
    table = []
    for size,row in stage_means.iterrows():
        p = sum(row[key] for key in ["s3_ms","s5_ms","s6_ms"])/row.total_ms
        table.append({"Kích thước":SIZE_NAMES[size],"p: S3+S5+S6":f"{100*p:.1f}%","Trần Amdahl":f"{1/(1-p):.2f}×"})
    show_table(pd.DataFrame(table))
    print("Giới hạn Amdahl giả định phần được tăng tốc có thời gian tiến về 0 và bỏ qua truyền dữ liệu CPU/GPU. Tỷ trọng S1 tăng ở ảnh lớn là cơ sở để xem xét mở rộng phạm vi tăng tốc.")

def finish():
    expected = {(r.pair_id,r.resolution,mode,run) for r in metadata.itertuples()
                for mode in MODES for run in range(1,RUNS+1)}
    actual = set(zip(timings.pair_id,timings.resolution,timings["mode"],timings.run))
    assert len(timings) == len(expected) and actual == expected
    assert not summary.duplicated(["pair_id","resolution","mode"]).any()
    env = json.loads((OUT / "environment.json").read_text())
    assert env["source_sha256"] == source_hashes(ROOT), "Nguồn thay đổi sau khi đo"
    for path,digest in env["dataset_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    for row in summary.itertuples():
        measured = timings[(timings.pair_id == row.pair_id) & (timings.resolution == row.resolution) & (timings["mode"] == row.mode)]
        assert len(measured) == RUNS
        for key in TIMINGS:
            assert math.isclose(measured[key].mean(), getattr(row,key), rel_tol=1e-8, abs_tol=1e-8)
            assert math.isclose(measured[key].std(ddof=1), getattr(row,key.replace("_ms","_std_ms")), rel_tol=1e-8, abs_tol=1e-8)
        assert row.bottleneck == max(STAGES,key=lambda key:getattr(row,key))
        p = sum(getattr(row,key) for key in ["s3_ms","s5_ms","s6_ms"])/row.total_ms
        assert math.isclose(row.gpu_candidate_fraction,p,rel_tol=1e-8)
        assert math.isclose(row.amdahl_limit,1/(1-p),rel_tol=1e-8)
        folder = OUT/row.pair_id/row.resolution/row.mode
        for run in range(1,RUNS+1):
            result = json.loads((folder/f"run_{run}.json").read_text())
            record = measured[measured.run == run].iloc[0]
            assert result["success"] and all(math.isfinite(result[key]) and result[key] >= 0 for key in TIMINGS)
            assert all(math.isclose(result[key],record[key],rel_tol=1e-8,abs_tol=1e-8) for key in TIMINGS)
        with Image.open(folder/"panorama.png") as image:
            assert image.size == (result["canvas_width"],result["canvas_height"])
            image.load()

    print("Đã kiểm tra: 4 phép thử hình học, 72 lượt đo và 24 panorama.")
    print("Số liệu khớp JSON C++, ảnh đầu ra và phiên bản nguồn/dataset.")
    print("Dữ liệu kiểm chứng: output/v0/summary.csv, output/v0/timings.csv và output/v0/synthetic/suite.json.")
