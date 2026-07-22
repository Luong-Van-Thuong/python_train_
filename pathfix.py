# -*- coding: utf-8 -*-
"""
pathfix.py — Cầu nối đường dẫn WSL2  <->  Windows thuần (Anaconda).
====================================================================
Toàn bộ script trong repo viết đường dẫn kiểu WSL: '/mnt/d/...'.
Khi chạy bằng Anaconda TRÊN WINDOWS (không qua WSL2) thì '/mnt/d/...'
KHÔNG tồn tại — phải là 'D:/...'. Module này TỰ ĐỘNG dịch qua lại:

      /mnt/<ổ>/...   <->   <Ổ>:/...

tuỳ theo môi trường đang chạy, nên cùng một file .py chạy được ở CẢ HAI nơi
mà không phải sửa hằng số đường dẫn.

CÁCH DÙNG trong script:
    from pathfix import P
    DEFAULT_DATA = P("/mnt/d/Projects_/.../dataset.yaml")
    INPUT_DIR    = P("/mnt/d/Images_/SIBV/A27/test")

Bạn viết đường dẫn theo KIỂU NÀO CŨNG ĐƯỢC ('/mnt/d/...' hoặc 'D:/...'),
P() sẽ chuyển về đúng dạng cho môi trường hiện tại.

>>> QUY TẮC VÀNG KHI GÕ ĐƯỜNG DẪN (để KHÔNG bao giờ dính lỗi băm chuỗi):
      DÙNG DẤU GẠCH XUÔI  '/'  —  ví dụ  P("D:/Projects_/.../best.pt")
    KHÔNG dùng dấu gạch ngược '\' trong chuỗi thường, vì Python coi '\t' '\r'
    '\n'... là ký tự đặc biệt và LÀM HỎNG chuỗi TRƯỚC KHI P() kịp xử lý.
    Nếu bắt buộc phải dán path Windows có '\', hãy thêm 'r' ở đầu:  r"D:\a\b".
    P() vẫn tự đổi '\' -> '/' cho bạn, nhưng chỉ cứu được khi chuỗi chưa bị băm.

TỰ CHUYỂN CHẾ ĐỘ (khi cần ép buộc) — đặt biến môi trường VISION_PATH_MODE:
    auto  : (mặc định) Windows -> 'win', còn lại (WSL/Linux) -> 'wsl'
    win   : ép ra dạng Windows  'D:/...'
    wsl   : ép ra dạng WSL/Linux '/mnt/d/...'
    off   : GIỮ NGUYÊN, không đụng gì (bỏ qua chuyển đổi)

Ví dụ:
    Windows PowerShell:   $env:VISION_PATH_MODE = "wsl";  python src\\...\\x.py
    WSL bash:             VISION_PATH_MODE=win python src/.../x.py
"""
import os
import re
import platform
import pathlib as _pl


def _detect_mode() -> str:
    """Xác định chế độ dịch đường dẫn cho môi trường hiện tại."""
    mode = os.environ.get("VISION_PATH_MODE", "auto").strip().lower()
    if mode in ("win", "windows"):
        return "win"
    if mode in ("wsl", "linux", "unix"):
        return "wsl"
    if mode in ("off", "none", "raw", "keep"):
        return "off"
    # auto: Windows -> win ; WSL / Linux / mac -> wsl
    return "win" if platform.system() == "Windows" else "wsl"


MODE = _detect_mode()

# '/mnt/d/....'  -> ổ 'd', phần còn lại
_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")
# 'D:/....' hoặc 'D:\....'  -> ổ 'D', phần còn lại
_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def P(path):
    """Chuẩn hoá 1 đường dẫn về đúng dạng cho môi trường hiện tại.

    - Nhận cả '/mnt/d/...' lẫn 'D:/...' (hoặc 'D:\\...').
    - Tự bỏ khoảng trắng thừa, bỏ dấu nháy lỡ dán vào, đổi '\\' -> '/'.
    - Trả về str đã dịch. None -> None. Đường dẫn tương đối / URL giữ nguyên.
    """
    if path is None:
        return path
    # bỏ khoảng trắng + dấu nháy lỡ copy vào 2 đầu
    s = str(path).strip().strip('"').strip("'")
    if MODE == "off":
        return s

    # Đổi mọi '\' -> '/' (Windows chấp nhận '/'); nhờ vậy dù lỡ dán path kiểu
    # r"D:\a\b" thì vẫn khớp regex và chạy được ở CẢ HAI môi trường.
    s = s.replace("\\", "/")

    if MODE == "win":
        m = _MNT_RE.match(s)
        if m:
            drive = m.group(1).upper()
            rest = (m.group(2) or "").lstrip("/")
            return _clean(f"{drive}:/{rest}")
        # đã là 'D:/...' hoặc đường tương đối -> trả bản đã chuẩn hoá gạch chéo
        return _clean(s)

    # MODE == "wsl"
    m = _DRIVE_RE.match(s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).lstrip("/")
        return _clean(f"/mnt/{drive}/{rest}")
    return _clean(s)


def _clean(path: str) -> str:
    """Gộp các dấu '/' lặp thành 1 (không đụng phần 'D:')."""
    return re.sub(r"(?<!:)/{2,}", "/", path)


def is_windows() -> bool:
    """True nếu đang chạy Anaconda Windows thuần (không WSL)."""
    return MODE == "win"


def Pyaml(yaml_path):
    """Chuẩn hoá 1 file data.yaml / dataset.yaml của YOLO cho môi trường hiện tại.

    Ultralytics ĐỌC TRỰC TIẾP khoá 'path' (và train/val/test) BÊN TRONG yaml, nên
    P() ở tầng Python không với tới được. Hàm này:
      - Đọc yaml, dịch 'path'/'train'/'val'/'test' tuyệt đối qua P().
      - Nếu KHÔNG cần đổi -> trả lại nguyên đường dẫn gốc.
      - Nếu có đổi -> ghi 1 bản cạnh file gốc tên '<ten>.__local__.yaml' và trả path đó.
    Dùng:  results = model.train(data=Pyaml(DEFAULT_DATA), ...)
    """
    import yaml as _yaml
    yaml_path = P(yaml_path)
    src = _pl.Path(yaml_path)
    with open(src, "r", encoding="utf-8") as f:
        doc = _yaml.safe_load(f)
    if not isinstance(doc, dict):
        return str(src)

    changed = False
    for key in ("path", "train", "val", "test"):
        val = doc.get(key)
        if isinstance(val, str) and val:
            new = P(val)
            if new != val:
                doc[key] = new
                changed = True
        elif isinstance(val, list):
            new_list = [P(v) if isinstance(v, str) else v for v in val]
            if new_list != val:
                doc[key] = new_list
                changed = True

    if not changed:
        return str(src)

    out = src.with_suffix(".__local__.yaml")
    with open(out, "w", encoding="utf-8") as f:
        _yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    return str(out)


# Cho phép:  python pathfix.py "/mnt/d/abc"   -> in ra bản đã dịch (để test nhanh)
if __name__ == "__main__":
    import sys
    print(f"[pathfix] MODE = {MODE}  (platform={platform.system()})")
    for arg in sys.argv[1:]:
        print(f"  {arg}  ->  {P(arg)}")
