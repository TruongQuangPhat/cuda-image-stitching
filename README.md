# Ghép ảnh panorama — C++17 baseline và kế hoạch CUDA

<p align="center">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white">
  <img alt="Pipeline" src="https://img.shields.io/badge/pipeline-CPU%20baseline-555555">
  <img alt="CUDA status" src="https://img.shields.io/badge/CUDA-planned%20only-76B900?logo=nvidia&logoColor=white">
  <img alt="Notebook" src="https://img.shields.io/badge/report-Jupyter-F37626?logo=jupyter&logoColor=white">
  <img alt="Runtime" src="https://img.shields.io/badge/runtime-Colab%20%7C%20WSL-4285F4">
</p>

Dự án xây dựng baseline tuần tự để ghép hai ảnh có vùng chồng lấp thành panorama. Notebook đánh giá sai số hình học, chất lượng ảnh ghép và thời gian của sáu giai đoạn xử lý, từ đó đề xuất phạm vi tăng tốc GPU.

> **Trạng thái:** các kết quả hiện tại thuộc V0 chạy trên CPU. CUDA mới là kế hoạch; repo chưa có kernel CUDA và chưa đo speedup GPU.

## Nội dung chính

- [Notebook báo cáo](Group09_Midterm.ipynb): ảnh trước/sau khi ghép, thông số, metrics và thời gian xử lý.
- **src/**: các module C++ của pipeline cùng mã điều phối thực nghiệm.
- [Dataset và nguồn ảnh](#dataset): dữ liệu CMU đã chuẩn bị, xây dựng từ OpenPano.
- [Cách chạy](#cách-chạy): Colab hoặc máy cá nhân có WSL.

## Pipeline

| Bước | Phương pháp | Đầu ra |
|---|---|---|
| S1 — Phát hiện điểm | FAST-9 (ngưỡng 20), Harris, non-maximum suppression; tháp ảnh 8 mức, tỷ lệ 1,2, tối đa 2.000 điểm mỗi ảnh | Keypoints |
| S2 — Mô tả điểm | Hướng intensity-centroid và rotated BRIEF 256 bit | Descriptors |
| S3 — Ghép đặc trưng | Hamming distance vét cạn, Lowe ratio test 0,8, không cross-check | Các cặp khớp |
| S4 — Ước lượng hình học | Normalized DLT, RANSAC 2.000 vòng, seed cố định, ngưỡng 3 px ở ảnh rộng 800 px | Homography B→A và inliers |
| S5 — Warping | Inverse mapping và nội suy song tuyến tính | Hai ảnh trên cùng canvas |
| S6 — Blending | Feather làm baseline; tùy chọn seam với feather bán kính 8 px và bù sáng | Panorama |

Pipeline chạy tuần tự trên CPU. **main.cu** là entry point tương thích với cách tổ chức mã của nhóm, nhưng được biên dịch như C++ bằng **g++**; V0 không cần **nvcc** hay CUDA runtime.

Homography phù hợp nhất khi cảnh gần phẳng hoặc máy ảnh chủ yếu xoay quanh cùng vị trí. Chuyển động, parallax và thay đổi phơi sáng vẫn có thể gây bóng đôi hoặc đường nối không đều.

## Đánh giá

Notebook chạy bốn cặp ảnh ở ba kích thước đầu vào (rộng 400, 800 và 1.200 px), với hai chế độ blending. Mỗi cấu hình có một lượt làm nóng và ba lượt đo, tổng cộng 72 lượt đo cho 24 cấu hình.

Các số liệu gồm thời gian tiền xử lý, thời gian S1–S6, tổng thời gian, số keypoint/điểm khớp/inlier và sai số chiếu. Đọc ảnh, ghi ảnh, giải mã JPEG, biên dịch và khởi tạo tiến trình không nằm trong thời gian pipeline.

Phần hình học kiểm tra bốn homography tổng hợp (tịnh tiến, xoay, đổi tỷ lệ và phối cảnh) trên lưới điểm độc lập. Ngưỡng kiểm tra nội bộ là sai số trung bình dưới 3 px và lớn nhất dưới 6 px. Kết quả này không đại diện cho mọi cảnh thực tế.

Feather là baseline để phân tích CPU/GPU. Chế độ seam tìm đường nối có sai khác màu thấp và bù sáng RGB trong giới hạn 0,5–2; nếu không tìm được đường nối liên tục, chương trình dùng feather dự phòng. Giới hạn Amdahl trong notebook là ước lượng lý thuyết, không phải speedup CUDA đã đo.

Bảng Feather/Seam tổng hợp thời gian toàn pipeline của bốn cặp ảnh ở ba kích thước trên; cột cuối cho biết số cấu hình áp dụng được seam. Hình so sánh trực quan dùng cặp CMU0 khó với ảnh đầu vào rộng 800 px.

## Dataset

Dự án dùng bốn cặp ảnh: CMU0 chuẩn, CMU1 chuẩn, CMU2 chuẩn và CMU0 khó. Mỗi cặp có ba kích thước đã chuẩn bị. Metadata nằm tại **dataset/metadata.csv** sau khi giải nén.

- [Tải ImageStitching_Team_Dataset.zip](https://drive.google.com/file/d/11Nbg77zkOizdNcY5vEmlybP60KTv5NUk/view)
- Nguồn ảnh: [OpenPano CMU, release 0.1](https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1)
- SHA-256 expected đang được cấu hình trong `src/notebook_support.py`:

```text
2097dc6eeb3d339731541f94df0d3f44694b447bd6f5b9c66699559ac7269827
```

Đây là expected value của helper, chưa được đối chiếu độc lập với ZIP gốc trong đợt hardening vì ZIP không có sẵn. Các file ảnh đã giải nén không chứng minh được SHA-256 của archive; không tạo lại ZIP để thay thế bằng chứng đó. Một ZIP cụ thể chỉ được xác nhận khi hash bytes thực tế khớp expected value.

Notebook kiểm tra SHA-256 trước khi giải nén. Nếu không tìm thấy dataset ở thư mục dự án hoặc thư mục cha, notebook sẽ thử tải từ Drive; trên Colab có thể chọn ZIP thủ công khi được nhắc. Thư mục **dataset/** được tạo lúc chạy và không được commit vào Git.

Theo tài liệu Midterm được cung cấp, giảng viên không đưa ra cơ chế đăng ký sites riêng. Các cặp CMU là lựa chọn có ghi nguồn của nhóm; điều này không đồng nghĩa với việc giảng viên đã chính thức chấp thuận dataset.

## Cách chạy

### Google Colab

Notebook xuất **Group09_Midterm.zip** ở cuối lần chạy. Nếu chưa có gói nguồn, chạy notebook một lần trong thư mục dự án đầy đủ trên máy cá nhân (có **src/** cạnh notebook); ZIP vừa tạo dùng làm gói đầu vào cho Colab.

1. Mở **Group09_Midterm.ipynb** bằng Colab.
2. Tải gói nguồn **Group09_Midterm.zip** vào thư mục **/content** qua **Files → Upload**. Gói cần có notebook ở thư mục gốc và thư mục **src/**.
3. Chọn **Runtime → Run all** và chờ các ô chạy xong.
4. Ô cuối kiểm tra tính nhất quán, lưu notebook đã chạy và tạo ZIP kết quả mới để tải xuống.

Gói ZIP kết quả chứa notebook có output, mã nguồn **src/** và ảnh thử **examples/campus.png**. Mục 7 của notebook cắt ảnh này thành hai phần trái–phải có overlap rồi ghép theo chiều ngang bằng V0. Các CSV, JSON và ảnh chi tiết được lưu riêng dưới **output/v0/**; phép thử ảnh cắt nằm trong **output/v0/overlap_demo/**.

### Máy cá nhân

Yêu cầu:

- Python 3.9 trở lên.
- C++17: **g++** 10 trở lên hoặc **clang++** tương thích.
- Python packages: Pillow, pandas và Jupyter.
- Trên Windows, cần WSL với **g++**; đặt repo trên ổ đĩa cục bộ để WSL truy cập được.

Cài package Python và mở notebook:

```bash
python -m pip install pillow pandas jupyter
jupyter notebook Group09_Midterm.ipynb
```

Khi notebook chạy, helper sẽ biên dịch V0 với **-std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror**, chạy thực nghiệm và lưu kết quả dưới **output/v0/**. Không cần cài OpenCV, Eigen hay CUDA Toolkit cho baseline CPU.

### Canonical build và standalone build

Canonical entry point là `main.cu`: `main()` gọi `stitching::runV0()` được khai báo trong `versions.h` và định nghĩa trong `v0_stitching.cpp`. `Runner.build()` trong helper dùng đúng source list và flags sau, biên dịch `.cu` như C++:

```bash
mkdir -p build
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 -Isrc \
  -DSTITCHING_NO_MAIN -x c++ \
  src/main.cu src/stb_impl.cpp src/v0_stitching.cpp src/orb.cpp \
  src/matching.cpp src/homography.cpp src/warping.cpp src/blending.cpp \
  -o build/v0_stitching
```

Ở native backend, helper dùng `CXX` nếu đã đặt, hoặc tìm `g++`/`clang++`; WSL backend dùng `g++`. Macro `STITCHING_NO_MAIN` loại standalone `main()` trong `v0_stitching.cpp`, để binary có đúng một `main()` từ `main.cu`.

Standalone entry point vẫn được hỗ trợ: dùng cùng pipeline/CLI, bỏ `main.cu` khỏi source list và không định nghĩa `STITCHING_NO_MAIN`:

```bash
mkdir -p build
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 -Isrc \
  src/stb_impl.cpp src/v0_stitching.cpp src/orb.cpp src/matching.cpp \
  src/homography.cpp src/warping.cpp src/blending.cpp \
  -o build/v0_standalone
```

Khi compile cả `main.cu` và `v0_stitching.cpp` trong một target, cần macro như canonical command; nếu thiếu sẽ có hai định nghĩa `main()`. Không cần build system hoặc source copy thuật toán khác trong notebook.

Ví dụ CLI cho ảnh PPM đã chuẩn bị:

```bash
./build/v0_stitching a.ppm b.ppm panorama.ppm metrics.json feather 3
```

Threshold của CLI tính bằng pixel ở kích thước ảnh thực tế. Notebook tự scale giá trị tham chiếu 3 px theo chiều rộng đầu vào/800; standalone binary nhận cùng tham số theo cùng quy ước.

### Kết quả sinh ra

```text
output/v0/
├── environment.json
├── summary.csv
├── timings.csv
├── synthetic/
│   └── suite.json
└── <pair>/<resolution>/<mode>/
    ├── panorama.png
    ├── run_1.json
    └── ...
```

**summary.csv** tổng hợp từng cấu hình; **timings.csv** lưu từng lượt đo; **synthetic/suite.json** lưu phép kiểm tra hình học. **build/**, **dataset/**, **output/**, ZIP và cache Python được bỏ qua bởi **.gitignore**.

## Cấu trúc mã nguồn

```text
src/
├── orb.cpp / orb.h                 # S1–S2: phát hiện và mô tả đặc trưng
├── matching.cpp / matching.h       # S3: ghép descriptors
├── homography.cpp / homography.h   # S4: DLT và RANSAC
├── warping.cpp / warping.h         # S5: biến đổi ảnh
├── blending.cpp / blending.h       # S6: feather và seam
├── v0_stitching.cpp                # Điều phối pipeline và ghi metrics
├── main.cu                         # Entry point V0, biên dịch như C++
├── common.h / cuda_utils.cuh       # I/O PPM và nhận diện chế độ biên dịch
├── versions.h / stb_impl.cpp       # Entry point và I/O PPM
├── notebook_setup.py               # Thiết lập môi trường và đọc gói nguồn trên Colab
└── notebook_support.py             # Build, chạy đo và xuất báo cáo
```

**stb_impl.cpp** triển khai I/O binary P6 RGB8, với `maxval=255`. Helper Python giải mã JPEG/PNG và chuyển sang PPM trước khi gọi C++; không có mã stb được đóng gói. Thuật toán S1–S6 nằm trong các file C/C++ của `src/`, không có bản sao độc lập trong notebook.

## Source provenance và kết quả lịch sử

`source_hashes()` tiếp tục trả raw SHA-256 của chính bytes từng file trong manifest `src/*.*`. Helper bổ sung `normalized_source_hashes()`: chỉ thay cặp byte CRLF (`0D 0A`) thành LF (`0A`) trước khi hash. CR đơn, khoảng trắng, tabs, BOM và newline cuối file được giữ nguyên. Không sửa nội dung file nguồn.

| Metadata | Raw SHA-256 | SHA-256 sau quy tắc CRLF→LF |
|---|---|---|
| `output/v0/environment.json` | `source_sha256` | `source_sha256_lf` |
| Notebook, `metadata.lan_chay` của lần export mới | `nguon_sha256` | `nguon_sha256_lf` |

Các record mới thêm `source_hash_normalization = "CRLF-to-LF-only-v1"`. Record cũ không có các field bổ sung vẫn được đọc theo raw hashes. Kiểm tra cuối lần chạy luôn yêu cầu **raw hashes khớp**; normalized hashes không cho phép bỏ qua thay đổi bytes sau khi đo. Normalized equality chỉ giải thích khác biệt line endings theo quy tắc trên, không chứng minh hai implementation tương đương về thuật toán.

Notebook đang lưu trong Git là bằng chứng của lần chạy lịch sử và có thể khác source hiện tại sau hardening. Giữ nguyên outputs/hashes gốc; không chép hashes mới vào kết quả cũ. Khi chốt source để nộp, cần chạy lại notebook top-to-bottom từ đúng `src/` đó để tạo kết quả và provenance mới. Phần diễn đạt về starter/sites đã được chỉnh lại trong notebook. Đợt chỉnh sửa Markdown này không chạy lại benchmark, không tạo số liệu hay bằng chứng thực nghiệm mới.

## Nhóm

| Mã số | Thành viên | Phụ trách |
|---|---|---|
| 23120318 | TRƯƠNG QUANG PHÁT | S1–S2 |
| 23120329 | CHÂU HUỲNH PHÚC | S3–S4 |
| 23120334 | HUỲNH TẤN PHƯỚC | S5–S6 và tích hợp |

## Giới hạn và ghi nhận

- V1/V2 là hướng phát triển; chưa có kernel CUDA hoặc kết quả tăng tốc GPU.
- Giảng viên cung cấp PDF Midterm, không cung cấp Starter Notebook. Các interface **common.h**, **cuda_utils.cuh**, **versions.h**, **main.cu** và **stb_impl.cpp** do nhóm thiết kế; không có interface starter bổ sung để đối chiếu. Đây không phải tuyên bố giảng viên đã phê duyệt implementation của nhóm.
- Sai số homography tổng hợp không bảo đảm chất lượng trên mọi ảnh thực tế; parallax và vật thể chuyển động vẫn có thể tạo bóng đôi.
- Không tính PSNR/SSIM vì không có panorama chuẩn tương ứng cho ảnh thực tế.

## Tài liệu tham khảo

- Rublee et al. (2011), *ORB: An efficient alternative to SIFT*; Rosten & Drummond (2006), *Machine learning for high-speed corner detection*.
- Lowe (2004), *Distinctive Image Features from Scale-Invariant Keypoints*.
- Hartley & Zisserman (2004), *Multiple View Geometry in Computer Vision*.
- Fischler & Bolles (1981), *Random Sample Consensus*.
- Szeliski (2022), *Computer Vision: Algorithms and Applications*, 2nd edition.
- [Mã nguồn nền tảng của nhóm](https://github.com/TruongQuangPhat/cuda-image-stitching) và [OpenPano](https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1).

---
<p align="center"><sub>Group09 · Midterm · V0 CPU</sub></p>
