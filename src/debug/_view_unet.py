# -*- coding: utf-8 -*-
"""Sinh ảnh xem: gốc + overlay dự đoán (downscale) cho dễ quan sát."""
from pathlib import Path
import numpy as np, cv2, yaml, torch
import segmentation_models_pytorch as smp

CFG = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/model_cfg.yaml"
SRC_DIR = "/mnt/d/Images_/SIBV/A27/test3"
OUT = "/mnt/d/Images_/SIBV/A27/_view_out"
MODELS = {
    "unet2": "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet2/weights/best.pt",
    "unet1": "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/weights/best.pt",
}
COLORS = [(0,0,0),(0,0,255),(0,165,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255)]

def imread_u(p):
    d=np.fromfile(str(p),np.uint8); return cv2.imdecode(d,cv2.IMREAD_COLOR) if d.size else None
def imwrite_u(p,img):
    ok,buf=cv2.imencode(".png",img); buf.tofile(str(p)) if ok else None

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
    cnt[cnt==0]=1; pa=ps/cnt[None]
    return pa.argmax(0).astype(np.int32), pa.max(0)

def render(bgr,pred,pmax,conf,min_area):
    vis=bgr.copy(); ov=vis.copy(); ndef=0
    for cid in np.unique(pred):
        if cid==0: continue
        col=COLORS[cid%len(COLORS)]
        m=((pred==cid)&(pmax>=conf)).astype(np.uint8)
        if m.sum()==0: continue
        n,lbl,st,_=cv2.connectedComponentsWithStats(m,8); keep=np.zeros_like(m)
        for k in range(1,n):
            if st[k,cv2.CC_STAT_AREA]<min_area: continue
            keep[lbl==k]=1; ndef+=1
            x,y,w,h=st[k,cv2.CC_STAT_LEFT],st[k,cv2.CC_STAT_TOP],st[k,cv2.CC_STAT_WIDTH],st[k,cv2.CC_STAT_HEIGHT]
            cv2.rectangle(vis,(x,y),(x+w,y+h),col,3)
        ov[keep==1]=col
    vis=cv2.addWeighted(ov,0.45,vis,0.55,0)
    return vis,ndef

def main():
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg=yaml.safe_load(open(CFG,encoding="utf-8")); nc=int(cfg["num_classes"]); tile=int(cfg["tile"])
    Path(OUT).mkdir(parents=True,exist_ok=True)
    models={}
    for tag,wp in MODELS.items():
        m=smp.create_model(cfg["arch"],encoder_name=cfg["encoder"],encoder_weights=None,in_channels=3,classes=nc).to(dev)
        st=torch.load(wp,map_location=dev)
        if isinstance(st,dict) and "model_state_dict" in st: st=st["model_state_dict"]
        m.load_state_dict(st); m.eval(); models[tag]=m
    files=sorted(p for p in Path(SRC_DIR).iterdir() if p.suffix.lower()==".bmp")
    for f in files:
        bgr=imread_u(f); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        panels=[bgr.copy()]
        for tag,m in models.items():
            pred,pmax=pred_full(m,rgb,cfg["mean"],cfg["std"],tile,dev,nc)
            for conf in (0.3,0.5):
                vis,nd=render(bgr,pred,pmax,conf,20)
                cv2.putText(vis,f"{tag} conf{conf} n={nd}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
                panels.append(vis)
        row=np.hstack(panels)
        sc=2400.0/row.shape[1]; row=cv2.resize(row,None,fx=sc,fy=sc)
        imwrite_u(Path(OUT)/f"{f.stem}_cmp.png",row)
        print("saved",f.stem,"panels",len(panels))
    print("OUT:",OUT)

if __name__=="__main__": main()
