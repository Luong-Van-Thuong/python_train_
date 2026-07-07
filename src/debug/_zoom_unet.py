# -*- coding: utf-8 -*-
"""Zoom vào vùng defo prob cao nhất: gốc | heatmap defo | overlay, full-res crop."""
from pathlib import Path
import numpy as np, cv2, yaml, torch
import segmentation_models_pytorch as smp

CFG="/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/model_cfg.yaml"
W2="/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet2/weights/best.pt"
W1="/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/weights/best.pt"
OUT="/mnt/d/Images_/SIBV/A27/_view_out"
IMG="/mnt/d/Images_/SIBV/A27/test3/Image__2026-06-22__16-06-02_obj_2.bmp"

def imread_u(p):
    d=np.fromfile(str(p),np.uint8); return cv2.imdecode(d,cv2.IMREAD_COLOR)
def imwrite_u(p,img):
    ok,buf=cv2.imencode(".png",img); buf.tofile(str(p))
def origins(L,t,ov):
    if L<=t: return [0]
    s=max(1,int(round(t*(1-ov)))); xs=list(range(0,L-t+1,s))
    if xs[-1]!=L-t: xs.append(L-t)
    return xs
@torch.no_grad()
def pred_full(model,rgb,mean,std,tile,dev,nc):
    H,W=rgb.shape[:2]; mean=np.array(mean,np.float32); std=np.array(std,np.float32)
    ps=np.zeros((nc,H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    for y0 in origins(H,tile,0.2):
        for x0 in origins(W,tile,0.2):
            c=rgb[y0:y0+tile,x0:x0+tile]; th,tw=c.shape[:2]
            if (th,tw)!=(tile,tile):
                pad=np.zeros((tile,tile,3),c.dtype); pad[:th,:tw]=c; c=pad
            x=(c.astype(np.float32)-mean)/std
            x=torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(dev)
            pr=torch.softmax(model(x),1)[0].float().cpu().numpy()
            ps[:,y0:y0+th,x0:x0+tw]+=pr[:,:th,:tw]; cnt[y0:y0+th,x0:x0+tw]+=1
    cnt[cnt==0]=1; pa=ps/cnt[None]; return pa

def load(wp,cfg,nc,dev):
    m=smp.create_model(cfg["arch"],encoder_name=cfg["encoder"],encoder_weights=None,in_channels=3,classes=nc).to(dev)
    st=torch.load(wp,map_location=dev)
    if isinstance(st,dict) and "model_state_dict" in st: st=st["model_state_dict"]
    m.load_state_dict(st); m.eval(); return m

def main():
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg=yaml.safe_load(open(CFG,encoding="utf-8")); nc=int(cfg["num_classes"]); tile=int(cfg["tile"])
    bgr=imread_u(IMG); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    for tag,wp in (("unet1",W1),("unet2",W2)):
        m=load(wp,cfg,nc,dev); pa=pred_full(m,rgb,cfg["mean"],cfg["std"],tile,dev,nc)
        defo=pa[3]  # class 3 = defo
        # toàn bộ lớp lỗi gộp (1..4) prob
        defect=pa[1:].max(0)
        y,x=np.unravel_index(np.argmax(defect),defect.shape)
        print(f"[{tag}] đỉnh lỗi tại (x={x},y={y}) prob={defect[y,x]:.3f} ; max_defo={defo.max():.3f}")
        r=260
        y0,y1=max(0,y-r),min(bgr.shape[0],y+r); x0,x1=max(0,x-r),min(bgr.shape[1],x+r)
        crop=bgr[y0:y1,x0:x1].copy()
        hm=(defect[y0:y1,x0:x1]*255).astype(np.uint8)
        hm=cv2.applyColorMap(hm,cv2.COLORMAP_JET)
        blend=cv2.addWeighted(crop,0.6,hm,0.4,0)
        panel=np.hstack([crop,hm,blend])
        cv2.putText(panel,f"{tag} peak={defect[y,x]:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        imwrite_u(Path(OUT)/f"zoom_{tag}_16-06-02.png",panel)
        print("saved zoom",tag)

if __name__=="__main__": main()
