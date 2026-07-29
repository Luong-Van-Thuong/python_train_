# -*- coding: utf-8 -*-
"""
coilassy_1240s.py — Do khoang cach VIEN DEN (khung nhua) -> DAY DONG (cuon dong)
cho con hang CoilAssy 1240S, bang OpenCV THUAN (khong train).

BOI CANH (doc anh that truoc khi doc code): con hang la 1 khung nhua toi mau (nhin xa
tuong den, nhin gan la xanh-xam dam) hinh vuong bo goc vat, giua co 1 LO TRON duoc
chieu sang nen (backlight) -> vung trong lo rat SANG. Day dong (nhieu soi xoan) chay
thanh 1 RANH doc theo canh trong cua khung, MOI VAO so voi VIEN NGOAI cua khung mot
khoang -- chinh khoang do la thu can do (kiem tra day dong co dat dung vi tri, khong
lo ra ngoai vien).

4 BUOC (dung y tuong Align -> Fixture -> Caliper da hoc o tu_hoc_opencv/):
  1. ALIGN nhe: tim tam LO TRON (vung sang nhat giua anh) lam moc tham chieu.
  2. FIXTURE: tim VIEN NGOAI cua khung (contour vung toi) -> rotated rect bao quanh,
     tinh THEO TUNG CON HANG (khong hard-code toa do co dinh, vi moi anh khung co the
     lech/xoay nhe).
  3. Sinh DAY DIEM CALIPER doc theo vien ngoai (moi diem 1 phap tuyen huong VAO TRONG).
  4. CALIPER: tren moi phap tuyen, do tu vien ngoai vao den khi gap DAY DONG (mau cam
     HOAC trang chay sang do phan chieu/loa sang) -> khoang cach pixel.

SO DO LUONG DU LIEU (anh -> so do), doc tu tren xuong, moi dong la 1 buoc trong
xu_ly_1_anh() o duoi:

    anh mau (BGR, numpy 3 chieu H x W x 3)
      |
      |-- cv2.cvtColor(BGR2GRAY) -----------------> anh xam "gray" (H x W, 1 so/pixel)
      |                                                  |
      |                                     tim_tam_lo_tron(gray) --> tam (cx,cy) + ban kinh lo
      |                                                  |
      |                                     tim_vien_ngoai_khung(gray) --> contour + rotated rect
      |
      |-- xay_mask_day_dong(anh mau) -------------> mask day dong (H x W, moi pixel 0 hoac 255)
      |
      +-- sinh_diem_caliper(contour, tam) ---------> danh sach (diem tren vien, huong vao trong)
                |
                +-- do_1_tia_caliper(mask day dong, diem, huong) --> 1 so: khoang cach (px)

KHONG BIET GI VE OPENCV? Doc phan "KHAI NIEM CO BAN" ngay duoi day 1 lan, roi doc code —
moi ham deu co giai thich rieng, khong can nho het cung duoc.

KHAI NIEM CO BAN (dung xuyen suot file nay):
  - Anh = 1 mang so (numpy array). Anh XAM: moi pixel 1 so 0-255 (0=den, 255=trang).
    Anh MAU: moi pixel 3 so (B,G,R -- OpenCV doc theo thu tu NGUOC, Blue-Green-Red,
    khac PIL/matplotlib doc R,G,B -- de nham nen luu y).
  - "mask" (mat na) = 1 anh xam CHI CO 2 GIA TRI: 0 (khong quan tam / nen) hoac 255
    (co quan tam / vat the). Dung de "khoanh vung" pixel nao thoa 1 dieu kien mau/sang.
  - "threshold" (nguong) = so sanh gia tri pixel > hoac < 1 con so co dinh de tao mask.
    VD: gray > 225 -> pixel do la "vung sang".
  - "contour" = duong bao quanh 1 vung lien thong trong mask (nhu dung but khoanh tron
    1 mieng hinh tren giay). cv2.findContours tra ve DANH SACH DIEM (x,y) noi tiep nhau
    tao thanh duong bao do.
  - "rotated rect" (hinh chu nhat xoay) = hinh chu nhat NHO NHAT bao quanh 1 tap diem,
    duoc phep XOAY (khong bat buoc nam ngang/doc theo truc anh) -- cv2.minAreaRect.
  - HSV = 1 cach khac de mo ta mau, thay vi tron B-G-R kho hinh dung. H (Hue) = mau sac
    (do, cam, vang... OpenCV do 0-179 do, khac chuan 0-360 vi phai vua trong 1 byte).
    S (Saturation) = do "dam" mau (0 = xam/trang, 255 = mau thuan). V (Value) = do sang.
    Muon loc "mau cam" thi loc theo H; muon loc "trang chay sang" thi loc theo V cao +
    S thap (sang nhung nhat mau).
  - "phap tuyen" (normal vector) = 1 vector CHIEU DAI 1, VUONG GOC voi huong duong vien
    tai 1 diem -- dung de "ban 1 tia thang" di VAO TRONG vat the tu diem do.

Anh dau vao hien co: C:/THUONG/Images/V2/CoilAssy/1240S/opencv (1 anh, se co them).
Nguong mau/sang o xay_mask_day_dong() la THAM SO CAN TU CHINH LAI khi co them anh that --
dang xem debug panel de tinh mat, chua chot cung.

=====================================================================================
BANG TRA CUU NHANH -- "TOI MUON SUA X" THI SUA O DAU (doc truoc khi mo code doc tung dong)
=====================================================================================
  Muon...                                          | Sua o day
  --------------------------------------------------|----------------------------------
  Vien ngoai/lo tron bat SAI (lem ra nen, hut vao   | DARK_GRAY_MAX (dong ~53) va/hoac
  trong qua)                                        | HOLE_BRIGHT_MIN (dong ~54) -- xem
                                                     | anh "3.mask toi"/"4.mask sang" o
                                                     | debug panel de biet tang hay giam
  Caliper day dac qua / thua qua (nhieu/it vach     | CALIPER_STEP_PX (dong ~56) -- so
  vang trong anh ket qua)                           | nho hon = day hon
  Caliper khong du dai de cham toi day dong (bi     | CALIPER_LEN_PX (dong ~57) -- tang
  "None", khong do duoc, in ra it tia hon tong so)  | len neu day dong nam xa vien hon
  Wire mask (mau cam/trang) bat thieu hoac bat      | xay_mask_day_dong() -- 2 dong
  nham vung khac (VD nen trang, vit kim loai)       | cv2.inRange(...) -- xem anh
                                                     | "5.wire_mask" o debug panel
  Co 1 vi tri co dinh (vit, nhan, chu...) luon bi   | EXCLUDE_ZONES (dong ~65) -- dung
  do nham thanh day dong                            | `--pick` de tu click lay toa do
  Doi thu muc anh dau vao / noi luu ket qua         | --data / --out khi chay, hoac sua
                                                     | DATA / OUT (dong ~50-51)
  Muon xem anh trung gian tung buoc de "bat benh"   | mo file *_debug.png trong thu muc
                                                     | --out (6 o: goc/gray/mask toi/
                                                     | mask sang/wire mask/ket qua)
=====================================================================================

Chay:
    "C:\\Users\\vanth\\miniconda3\\envs\\vision_ai\\python.exe" src\\measure\\coilassy_1240s.py
    (--data de doi thu muc anh, --out de doi noi luu debug, --step de doi mat do caliper)
"""
import argparse   # doc tham so dong lenh (--data, --out, --step, --pick...)
import glob       # tim file theo mau ten (VD "*.bmp")
import os         # ghep duong dan, tao thu muc, lay ten file tu duong dan day du

import cv2          # OpenCV -- thu vien xu ly anh chinh cua file nay
import numpy as np  # numpy -- anh trong OpenCV luon la 1 numpy array, can numpy de tinh toan tren no

# Doan duoi day CHI la ky thuat de import duoc "pathfix" (file o thu muc goc repo) du
# script nay nam sau trong src/measure/ -- KHONG lien quan gi den thuat toan do, bo qua
# duoc, khong can hieu ky de custom code.
import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents
                         if (_p / "pathfix.py").exists()))
from pathfix import P

# ---------------------------------------------------------------- cau hinh
# Tat ca "hang so" (VIET_HOA) duoi day la THAM SO -- doi so, KHONG doi logic code, la
# cach nhanh nhat de tuy bien ket qua. Xem bang tra cuu o dau file de biet sua cai nao.
DATA = P("C:/THUONG/Images/V2/CoilAssy/1240S/opencv")             # thu muc chua anh .bmp/.jpg/.png dau vao
OUT  = P("C:/THUONG/Images/V2/CoilAssy/1240S/opencv/_ket_qua")    # thu muc se luu anh debug + anh ket qua

WORK_W = 1100                 # anh goc rat to (3648x3648px) -- resize ve chieu rong nay TRUOC KHI xu ly,
                               # de moi nguong pixel (DARK_GRAY_MAX...) on dinh du anh goc to nho khac nhau
DARK_GRAY_MAX = 150           # 1 pixel gray < so nay -> coi la "khung nhua" (mau toi). Nen trang co gray ~255,
                               # nen tang/giam so nay se lam mask "khung" nho/to di theo
HOLE_BRIGHT_MIN = 225         # 1 pixel gray > so nay -> coi la "vung sang trong lo tron" (anh backlight)
CALIPER_STEP_PX = 14          # cu moi 14px doc theo vien ngoai thi dat 1 diem do (1 "caliper") -- xem buoc 3
CALIPER_LEN_PX = 55           # 1 tia caliper do toi da 55px vao trong truoc khi bo cuoc (khong tim thay day dong)
CALIPER_SKIP_PX = 2           # bo qua 2px dau tien ngay tren vien (vung nhieu do net vien, de anh huong)

# Vung LOAI TRU (VD: vit noi day kim loai sang, bi mask wire nham thanh day dong).
# Toa do tinh TRONG ANH DA RESIZE ve WORK_W (khong phai anh goc) -- dung `--pick` de tu lay toa do:
#   python src/measure/coilassy_1240s.py --pick "C:/THUONG/Images/V2/CoilAssy/1240S/opencv/<ten_anh>.bmp"
#   Click chuot trai gan tam vit -> in ra (x, y) len console -> tu dien vao list duoi day.
# Moi diem caliper (buoc 3) nam trong ban_kinh cua 1 zone se bi BO QUA, khong do.
EXCLUDE_ZONES = [
    # {"tam": (123, 45), "ban_kinh": 40},
]


# ---------------------------------------------------------------- buoc 1: tam lo tron
def tim_tam_lo_tron(gray):
    """Nguong vung SANG (backlight xuyen qua lo) -> lay component to nhat GAN GIUA anh
    (tranh nham vao vung trang o RIA anh la nen, khong phai lo).
    Tra ve (cx, cy, ban_kinh_uoc_luong, mask_sang)."""
    # Buoc 1a: tao MASK -- moi pixel gray > HOLE_BRIGHT_MIN thi la True/1, con lai False/0.
    # ".astype(np.uint8) * 255" chi la doi True/False (1/0) thanh 255/0 de giong quy uoc
    # anh mask chuan cua OpenCV (0 hoac 255), khong doi y nghia gi ca.
    mask = (gray > HOLE_BRIGHT_MIN).astype(np.uint8) * 255

    h, w = gray.shape   # gray.shape tra ve (so_hang, so_cot) = (chieu cao, chieu rong) tinh theo pixel
    # Buoc 1b: chi giu vung sang nam trong 70% GIUA anh -- ve 1 hinh chu nhat trang (255)
    # tren nen den (0) roi AND voi mask, coi nhu "cat" mask lai chi con phan giua anh.
    # Ly do: neu nen ngoai con hang cung bi loa sang thanh mau trang o SAT RIA anh thi se
    # bi nham la "lo tron" neu khong chan lai truoc.
    bien = np.zeros_like(mask)
    m = 0.15   # chua 15% ria moi canh -> con lai 70% giua
    cv2.rectangle(bien, (int(w * m), int(h * m)), (int(w * (1 - m)), int(h * (1 - m))), 255, -1)
    mask = cv2.bitwise_and(mask, bien)   # AND: pixel chi con 255 khi CA HAI mask deu 255

    # Buoc 1c: mask co the co NHIEU vung sang rieng le (nhieu/vet sang lat...) --
    # connectedComponentsWithStats "gom nhom" cac pixel 255 dinh lien nhau thanh tung
    # "cum" (component) rieng biet, danh so 0,1,2..., va tinh san dien tich + tam moi cum.
    #   n     = tong so cum tim duoc (ke ca cum so 0 = phan NEN mau den, luon co)
    #   stats = mang [n, 5]: cot cuoi (CC_STAT_AREA) la dien tich (so pixel) cua tung cum
    #   cent  = mang [n, 2]: toa do TAM (cx, cy) cua tung cum
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:   # n==1 nghia la CHI co cum nen (khong tim thay vung sang nao) -> chiu, tra None
        return None, mask

    # Buoc 1d: chon cum TO NHAT trong so cac cum THAT (bo qua index 0 = nen) lam "lo tron".
    # stats[1:, AREA] = dien tich cac cum 1..n-1; argmax tim vi tri cum to nhat trong do;
    # +1 de tro lai dung chi so trong mang goc (vi da bo cum 0 truoc khi argmax).
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cent[idx]                                    # tam cum do, chinh la tam lo tron can tim
    r = np.sqrt(stats[idx, cv2.CC_STAT_AREA] / np.pi)     # suy nguoc ban kinh tu dien tich hinh tron: S=pi*r^2
    return (float(cx), float(cy), float(r)), mask


# ---------------------------------------------------------------- buoc 2: vien ngoai khung
def tim_vien_ngoai_khung(gray):
    """Nguong vung TOI (khung nhua) -> contour NGOAI CUNG lon nhat -> rotated rect bao quanh.
    Tra ve (contour, rotated_rect, mask_toi)."""
    # Giong buoc 1a nhung nguoc dau: lay pixel TOI (khung nhua) thay vi SANG.
    mask = (gray < DARK_GRAY_MAX).astype(np.uint8) * 255

    # morphologyEx(..., MORPH_CLOSE, kernel) = "lap day lo nho + noi cac vet dut gan nhau"
    # trong mask (gian no roi co lai). Vi anh that co nhieu, mask khung co the bi lo cham
    # nho hoac dut net o vai cho -- CLOSE giup mask lien mach hon truoc khi tim contour.
    # np.ones((7,7)) = kich thuoc "ban chai" dung de lap/noi, cang lon cang lap duoc lo to
    # hon nhung cung de lam bien dang chi tiet nho that.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # findContours: quet mask, tra ve DANH SACH duong bao (moi duong bao la 1 mang diem
    # (x,y) noi tiep nhau). RETR_EXTERNAL = chi lay duong bao NGOAI CUNG (bo qua lo/vet
    # den trong long vat the, vi minh chi can hinh dang tong the). CHAIN_APPROX_NONE =
    # giu DAY DU tung diem doc theo duong bao (khong rut gon), can thiet o buoc 3 de
    # "di bo" doc vien tinh phap tuyen.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None, mask

    # Neu mask con dinh nhieu cum toi rieng le (VD nhieu sat mep anh), lay contour co
    # DIEN TICH LON NHAT -- gia dinh do chinh la khung con hang (vat the chinh trong anh).
    contour = max(cnts, key=cv2.contourArea)

    # minAreaRect: tim hinh chu nhat NHO NHAT bao quanh tron tap diem contour, DUOC PHEP
    # XOAY theo huong tu nhien cua tap diem (khac boundingRect luon nam ngang/doc theo
    # truc anh). Ket qua la 1 tuple ((tam_x,tam_y), (rong,cao), goc_xoay_do) -- KHONG phai
    # 4 diem goc, muon ve/dung 4 goc phai qua cv2.boxPoints(rect) (xem o xu_ly_1_anh).
    rect = cv2.minAreaRect(contour)   # TINH THEO CON HANG NAY (tu contour that), khong hard-code toa do
    return contour, rect, mask


# ---------------------------------------------------------------- buoc 3: sinh diem caliper
def trong_vung_loai_tru(p, exclude_zones):
    """True neu diem `p` nam trong ban kinh 1 vung nao do trong `exclude_zones`."""
    for z in exclude_zones:
        tam_z = np.array(z["tam"], dtype=np.float32)
        # np.hypot(dx, dy) = sqrt(dx^2 + dy^2) = KHOANG CACH EUCLID giua 2 diem (dinh ly Pythagoras).
        # "*(p - tam_z)" nghia la unpack mang [dx, dy] thanh 2 tham so rieng cho ham hypot.
        if np.hypot(*(p - tam_z)) <= z["ban_kinh"]:
            return True
    return False


def sinh_diem_caliper(contour, tam, step_px=CALIPER_STEP_PX, exclude_zones=None):
    """Lay moi `step_px` diem tren contour (da xap xi lien tuc nho CHAIN_APPROX_NONE), tinh phap
    tuyen DON VI huong VAO TRONG (ve phia `tam`) tai moi diem. Bo qua diem nam trong `exclude_zones`
    (VD vi tri vit noi day, xem EXCLUDE_ZONES o dau file).
    Tra ve list (point_xy, normal_unit_xy)."""
    # contour tu findContours co shape (N, 1, 2) -- reshape ve (N, 2) cho de thao tac
    # (moi hang la 1 diem [x, y]).
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    tam = np.array(tam, dtype=np.float32)
    exclude_zones = exclude_zones or []
    ket_qua = []
    # can vai diem LAN CAN (khong phai ngay sat) de tinh huong duong vien cho on dinh --
    # neu chi lay 2 diem SAT NHAU thi huong de bi nhieu (vien khung co rang cua/gon song nho).
    k = max(3, step_px // 3)
    for i in range(0, n, step_px):   # "di bo" doc contour, cu step_px diem lai dung 1 lan
        p = pts[i]
        if trong_vung_loai_tru(p, exclude_zones):
            continue   # bo qua diem nay, khong tao caliper (VD dang o vi tri vit)

        # Lay 1 diem TRUOC va 1 diem SAU (theo chi so, vong lai tu dau neu vuot qua so
        # luong diem nho toan tu % n) -- vector noi 2 diem nay xap xi HUONG TIEP TUYEN
        # (huong "chay doc theo" duong vien) tai diem p.
        p_prev = pts[(i - k) % n]
        p_next = pts[(i + k) % n]
        tangent = p_next - p_prev
        norm = np.hypot(*tangent)   # do dai vector tangent (de chuan hoa ve vector don vi)
        if norm < 1e-3:             # vector qua ngan (2 diem trung nhau) -> bo qua, tranh chia cho 0
            continue
        tangent /= norm             # chia cho do dai chinh no -> vector CHIEU DAI DUNG BANG 1 (vector don vi)

        # Phap tuyen = tiep tuyen XOAY 90 DO. Cong thuc xoay 90 do cho vector (x,y):
        # (x,y) -> (-y,x). Day la phep xoay hinh hoc chuan, khong can nho, chi can ap dung.
        phap_tuyen = np.array([-tangent[1], tangent[0]])

        # Xoay 90 do co THE cho ra huong VAO TRONG hoac RA NGOAI (2 kha nang doi xung) --
        # np.dot(phap_tuyen, tam - p) = tich vo huong giua phap_tuyen va vector "tu p toi
        # tam". Neu > 0 nghia la phap_tuyen dang "cung huong" voi huong toi tam (dung, vao
        # trong roi); neu < 0 nghia la dang chi RA NGOAI -> dao dau lai (`-phap_tuyen`).
        if np.dot(phap_tuyen, tam - p) < 0:
            phap_tuyen = -phap_tuyen
        ket_qua.append((p, phap_tuyen))
    return ket_qua


# ---------------------------------------------------------------- buoc 4: mask day dong + do
def xay_mask_day_dong(img_bgr):
    """Day dong: mau CAM (hue am, saturation cao) HOAC trang CHAY SANG (do phan xa/loa sang -
    sang gan bao hoa nhung van con chut sac am vi la kim loai phan quang, khac han nen trang
    thuan tuy hoac khung nhua toi).
    Tra ve mask nhi phan 0/255 CUNG KICH THUOC anh dau vao."""
    # Doi tu BGR sang HSV -- xem giai thich HSV o phan "KHAI NIEM CO BAN" dau file.
    # Loc mau CAM bang B-G-R se rat kho (phai can nhac ca 3 kenh tron voi nhau), loc bang
    # H (hue) trong HSV don gian hon nhieu vi hue dai dien DUY NHAT cho "mau sac".
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # cv2.inRange(anh, (lo1,lo2,lo3), (hi1,hi2,hi3)): tra ve mask 255 tai nhung pixel co
    # CA 3 kenh cung nam trong khoang [lo,hi] tuong ung, 0 o nhung pixel con lai.
    # Dong duoi: H trong [0,30] (do cam), S >= 60 (co mau, khong xam), V >= 90 (khong qua toi).
    cam = cv2.inRange(hsv, (0, 60, 90), (30, 255, 255))          # day mau cam ro
    # Dong duoi: bat vung GAN TRANG nhung V khong toi da 255 va S van con chut (0-90) --
    # day la kieu "kim loai/day loa sang" (gan bao hoa nhung khong trang tinh khiet nhu
    # nen), khac nen trang THUAN (S~0) va khac khung nhua toi (V thap).
    chay_sang = cv2.inRange(hsv, (0, 0, 235), (40, 90, 255))     # vung gan trang nhung con anh am (kim loai loa sang)

    mask = cv2.bitwise_or(cam, chay_sang)   # OR: pixel la day dong neu THOA 1 TRONG 2 dieu kien tren
    # CLOSE lan nua de lap cac lo/khe nho trong mask (soi day mong, de bi dut net do nhieu).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def do_1_tia_caliper(wire_mask, p, phap_tuyen, skip_px=CALIPER_SKIP_PX, max_len=CALIPER_LEN_PX):
    """Di doc tia (p + t*phap_tuyen) tu t=skip_px den max_len, tra ve t DAU TIEN gap pixel
    wire_mask=255 (khoang cach px tu vien ngoai den day dong), hoac None neu khong gap trong
    pham vi do."""
    h, w = wire_mask.shape
    # Day chinh la "CALIPER": xuat phat tu diem p tren vien, di TUNG PIXEL MOT (t = 2, 3,
    # 4...) theo huong phap_tuyen (da chuan hoa dai=1 nen di t buoc = di t pixel), moi buoc
    # kiem tra pixel tai do co nam trong wire_mask (co phai day dong khong).
    for t in range(skip_px, max_len):
        x, y = p + t * phap_tuyen              # toa do (thuc, co le) tai buoc t
        xi, yi = int(round(x)), int(round(y))  # lam tron ve so nguyen de tra cuu vao mang anh (chi so pixel)
        # wire_mask[yi, xi]: LUU Y thu tu [hang, cot] = [y, x] khi tra cuu mang numpy, NGUOC
        # voi thu tu (x, y) thong thuong hay dung khi VE hinh (cv2.circle, cv2.line...) --
        # day la 1 trong nhung diem de nham nhat khi moi hoc OpenCV/numpy.
        if 0 <= xi < w and 0 <= yi < h and wire_mask[yi, xi] > 0:
            return float(t)   # cham day dong roi -> t chinh la khoang cach (px) can tim
    return None   # di het max_len ma khong cham -> khong do duoc (co the do vien qua xa/mask thieu)


# ---------------------------------------------------------------- ve debug + main
def _title(tile, text):
    """Dan 1 thanh tieu de (chu vang tren nen den) len phia tren 1 anh nho -- chi de
    anh debug de nhin/phan biet cac o, khong lien quan gi den thuat toan do."""
    h, w = tile.shape[:2]
    if tile.ndim == 2:   # anh xam (2 chieu) -> doi sang 3 kenh (mau) de ghep chung voi anh mau khac
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((24, w, 3), np.uint8)   # thanh den cao 24px, rong bang anh
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])   # np.vstack = ghep 2 anh CHONG DOC (thanh tieu de o tren, anh o duoi)


def xu_ly_1_anh(path, step_px, out_dir):
    """Ham TRUNG TAM: chay du 4 buoc tren 1 file anh, in ket qua so ra console, va luu 2
    anh (panel debug 6-o + anh ket qua) vao out_dir. Doc ham nay truoc de nam duoc THU TU
    goi cac ham o tren, roi moi doc sau vao tung ham neu can."""
    img0 = cv2.imread(path)   # doc anh tu file -> numpy array BGR. Tra ve None neu duong dan sai/file hong.
    if img0 is None:
        print(f"  [loi] khong doc duoc: {path}")
        return

    # Resize ve chieu rong chuan WORK_W (xem giai thich o phan cau hinh) -- giu nguyen ty
    # le khung hinh (scale ap dung deu cho ca w va h).
    h0, w0 = img0.shape[:2]
    scale = WORK_W / w0
    img = cv2.resize(img0, (WORK_W, int(h0 * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # anh xam, dung cho buoc 1 va 2 (chi can do sang, khong can mau)

    # ---- BUOC 1 + 2: tim tam lo tron va vien ngoai khung (doc lap voi nhau, chay truoc) ----
    tam_info, mask_sang = tim_tam_lo_tron(gray)
    contour, rect, mask_toi = tim_vien_ngoai_khung(gray)

    if tam_info is None or contour is None:
        print(f"  [canh bao] khong tim thay tam lo tron hoac vien khung: {os.path.basename(path)}")
        return
    cx, cy, r_lo = tam_info

    # ---- BUOC 3 + 4: mask day dong tren CA anh, roi sinh caliper va do tung tia ----
    wire_mask = xay_mask_day_dong(img)
    diem_caliper = sinh_diem_caliper(contour, (cx, cy), step_px=step_px, exclude_zones=EXCLUDE_ZONES)

    khoang_cach = []   # gom lai TAT CA khoang cach do duoc (px) de tinh min/max/mean/median cuoi cung
    vis = img.copy()   # anh de VE ket qua len (khong ve truc tiep len img goc, giu img sach cho cac buoc khac)

    # Ve khung xanh la = vien ngoai (rotated rect). boxPoints(rect) doi tu dang
    # ((tam),(rong,cao),goc) sang 4 TOA DO GOC THAT (can de ve duoc bang drawContours/line).
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(vis, [box], -1, (0, 255, 0), 1)          # mau OpenCV la (B,G,R) -> (0,255,0) = xanh la
    cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)   # cham do dac = tam lo tron
    cv2.circle(vis, (int(cx), int(cy)), int(r_lo), (0, 0, 255), 1)  # vien do = duong tron uoc luong cua lo
    for z in EXCLUDE_ZONES:
        cv2.circle(vis, tuple(map(int, z["tam"])), z["ban_kinh"], (255, 0, 255), 1)  # tim = vung da loai tru

    # Voi TUNG diem caliper: do 1 tia, ve 1 duong (vang neu do duoc, xam neu khong) tu
    # diem tren vien toi vi tri cham duoc (hoac toi het do dai toi da neu khong cham).
    for p, phap_tuyen in diem_caliper:
        t = do_1_tia_caliper(wire_mask, p, phap_tuyen)
        p_end = p + (t if t else CALIPER_LEN_PX) * phap_tuyen
        mau = (0, 255, 255) if t is not None else (128, 128, 128)   # vang = do duoc, xam = khong do duoc
        cv2.line(vis, tuple(p.astype(int)), tuple(p_end.astype(int)), mau, 1)
        if t is not None:
            khoang_cach.append(t)

    # ---- Tong hop so lieu + in ra console ----
    khoang_cach = np.array(khoang_cach)
    ten = os.path.basename(path)   # chi lay TEN FILE (bo phan thu muc) de in cho gon
    if len(khoang_cach):
        print(f"  {ten}: {len(khoang_cach)}/{len(diem_caliper)} tia do duoc | "
              f"px min={khoang_cach.min():.1f} max={khoang_cach.max():.1f} "
              f"mean={khoang_cach.mean():.1f} median={np.median(khoang_cach):.1f}")
    else:
        print(f"  {ten}: KHONG tia nao do duoc khoang cach -- kiem tra lai nguong mau wire_mask")

    # ---- Xuat anh debug (6 o, xem tung buoc trung gian) + anh ket qua rieng ----
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(ten)[0]   # ten file KHONG co duoi (.bmp/.jpg...), dung de dat ten file output

    # bitwise_and(img, img, mask=wire_mask): giu nguyen mau tai pixel nao wire_mask=255,
    # to den (0) tai cho con lai -- 1 cach truc quan de "xem thu mask dang bat trung nhung
    # cho nao tren anh that", thay vi chi nhin mask trang-den kho lien tuong.
    wire_vis = cv2.bitwise_and(img, img, mask=wire_mask)
    tiles = [
        _title(img, "1.anh (resize)"),
        _title(gray, "2.gray"),
        _title(mask_toi, "3.mask toi (khung)"),
        _title(mask_sang, "4.mask sang (lo tron)"),
        _title(wire_vis, "5.wire_mask ap len anh"),
        _title(vis, f"6.ket qua: xanh=vien ngoai do  vang=tia do toi day dong"),
    ]
    # Dong nhat kich thuoc 6 o (resize ve cung chieu rong cw, giu ty le) roi ghep 3-cot x 2-hang
    # bang np.hstack (ghep NGANG) + np.vstack (ghep DOC) -- chi la dan hinh, khong anh huong ket qua do.
    cw = 340
    tiles = [cv2.resize(t, (cw, int(t.shape[0] * cw / t.shape[1]))) for t in tiles]
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for t in tiles]
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    panel = np.vstack(rows)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_debug.png"), panel)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_ketqua.png"), vis)


def che_do_pick(path):
    """Mo anh (da resize ve WORK_W) trong 1 cua so, click chuot trai gan vi tri can loai tru
    (VD tam vit noi day) -> in ra console dong the {"tam": (x, y), "ban_kinh": 40}, sẵn dang dan
    thang vao EXCLUDE_ZONES o dau file. Nhan ESC hoac 'q' de dong cua so."""
    img0 = cv2.imread(path)
    h0, w0 = img0.shape[:2]
    scale = WORK_W / w0
    img = cv2.resize(img0, (WORK_W, int(h0 * scale)))   # PHAI resize giong het xu_ly_1_anh(), neu khong
                                                          # toa do click se LECH so voi toa do dung khi do that
    vis = img.copy()

    # OpenCV goi ham nay MOI KHI co su kien chuot tren cua so (di chuyen, bam, tha...).
    # (event, x, y, flags, userdata) la CHU KY BAT BUOC cua OpenCV -- kể cả khong dung
    # het tham so van phai khai bao du, day la quy dinh cua thu vien, khong phai loi.
    def _on_click(event, x, y, flags, userdata):
        nonlocal vis   # cho phep sua bien `vis` cua ham ben ngoai (che_do_pick) tu ben trong ham con nay
        if event == cv2.EVENT_LBUTTONDOWN:   # chi xu ly khi BAM CHUOT TRAI xuong (bo qua di chuyen/tha ra)
            cv2.circle(vis, (x, y), 40, (255, 0, 255), 2)   # ve vong tron tim tai cho vua click, de xem lai
            print(f'    {{"tam": ({x}, {y}), "ban_kinh": 40}},')   # in dong san sang dan vao EXCLUDE_ZONES

    ten_cua_so = "Click gan vi tri can loai tru -- ESC/q de thoat"
    cv2.namedWindow(ten_cua_so)
    cv2.setMouseCallback(ten_cua_so, _on_click)   # gan ham _on_click o tren lam "nguoi nghe" su kien chuot
    print("Click chuot trai gan vi tri can loai tru (VD tam vit). Toa do se in ra day:\n")
    while True:   # vong lap ve lien tuc de cua so hien thi song dong (thay duoc vong tron vua ve)
        cv2.imshow(ten_cua_so, vis)
        key = cv2.waitKey(20) & 0xFF   # doi 20ms xem co phim nao duoc bam khong (can de cua so khong "dong cung")
        if key in (27, ord("q")):      # 27 = ma phim ESC
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--step", type=int, default=CALIPER_STEP_PX)
    ap.add_argument("--pick", default=None,
                     help="duong dan 1 anh -- mo cua so click chuot de lay toa do EXCLUDE_ZONES")
    args = ap.parse_args()

    if args.pick:            # co truyen --pick -> CHI chay che do lay toa do roi thoat, khong do
        che_do_pick(args.pick)
        return

    # glob.glob("*.bmp") = tim tat ca file khop mau ten trong 1 thu muc -- lam voi ca 3
    # duoi anh hay dung (.bmp/.jpg/.png) roi gop lai + sap xep theo ten cho on dinh thu tu.
    files = sorted(sum([glob.glob(os.path.join(args.data, ext))
                        for ext in ("*.bmp", "*.jpg", "*.png")], []))
    print(f"[coilassy_1240s] tim thay {len(files)} anh trong {args.data}")
    for f in files:   # chay tuan tu tung anh mot qua toan bo 4 buoc (xem xu_ly_1_anh o tren)
        xu_ly_1_anh(f, args.step, args.out)
    print(f"\n[debug] panel + anh ket qua da luu vao: {args.out}")


if __name__ == "__main__":   # chi chay main() khi file nay duoc CHAY TRUC TIEP (python coilassy_1240s.py),
    main()                   # khong chay khi file nay bi file KHAC import (VD sau nay ghep vao pipeline lon hon)
