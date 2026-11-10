import cv2, numpy as np, json, math, sys
RES=0.1; X0,Y1=-68,189
m=cv2.imread('mos.png'); hsv=cv2.cvtColor(m,cv2.COLOR_BGR2HSV); S=hsv[...,1]; V=hsv[...,2]
def uv(x,y): return int(round((x-X0)/RES)), int(round((Y1-y)/RES))
def xy(u,v): return X0+u*RES, Y1-v*RES
reg=np.zeros(V.shape,np.uint8); a=uv(-64,186); b=uv(63,-62); reg[a[1]:b[1],a[0]:b[0]]=255
def rectof(pts):
    (cx,cy),(a,b),ang=cv2.minAreaRect(pts.astype(np.float32))
    if a<b: a,b=b,a; ang+=90
    return cx,cy,a*RES,b*RES,(-ang)%180
def ok(L,W,fill,hd): return 4.5<=L<=14.5 and 1.9<=W<=3.6 and fill>0.72
units=[]; taken=np.zeros(V.shape,np.uint8); rejects=[]
# multi-threshold: strict first (separates touching units), then looser for dull roofs
for TH,SM in ((225,26),(212,30),(200,34),(188,36)):
    w=((V>TH)&(S<SM)).astype(np.uint8)*255; w&=reg
    w=cv2.morphologyEx(w,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    w=cv2.morphologyEx(w,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,lab,st,_=cv2.connectedComponentsWithStats(w)
    for i in range(1,n):
        A=st[i,4]*RES*RES
        if A<9: continue
        comp=(lab==i)
        if (taken[comp]>0).mean()>0.25: continue
        pts=np.argwhere(comp)[:,::-1]; cx,cy,L,W,hd=rectof(pts); fill=A/(L*W)
        if ok(L,W,fill,hd):
            units.append([cx,cy,L,W,hd,TH]); box=cv2.boxPoints(((cx,cy),(L/RES+6,W/RES+6),-hd)).astype(np.int32); cv2.fillPoly(taken,[box],255)
        elif TH==188: rejects.append((round(xy(cx,cy)[0],1),round(xy(cx,cy)[1],1),round(L,1),round(W,1),round(fill,2)))
n0=len(units)
# pass 2: relaxed shape test at mid threshold
w=((V>205)&(S<32)).astype(np.uint8)*255&reg; w=cv2.morphologyEx(w,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
n,lab,st,_=cv2.connectedComponentsWithStats(w)
for i in range(1,n):
    A=st[i,4]*RES*RES
    if A<7: continue
    comp=(lab==i)
    if (taken[comp]>0).mean()>0.25: continue
    pts=np.argwhere(comp)[:,::-1]; cx,cy,L,W,hd=rectof(pts)
    if 4.5<=L<=14.5 and 1.8<=W<=3.6 and A/(L*W)>0.58:
        units.append([cx,cy,L,W,hd,150]); box=cv2.boxPoints(((cx,cy),(L/RES+6,W/RES+6),-hd)).astype(np.int32); cv2.fillPoly(taken,[box],255)
n1=len(units)
# pass 3: leftovers (roofs broken up by vents / AC units)
left=((V>198)&(S<34)).astype(np.uint8)*255&reg; left[taken>0]=0
left=cv2.morphologyEx(left,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)); left=cv2.morphologyEx(left,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(left)
base=np.array([[u[0],u[1],u[4]] for u in units])
for i in range(1,n):
    A=st[i,4]*RES*RES
    if A<6.5: continue
    pts=np.argwhere(lab==i)[:,::-1]; cx,cy,L,W,hd=rectof(pts)
    if L>14.5 or W>4.2: continue
    if L/W<1.7: hd=base[np.argmin((base[:,0]-cx)**2+(base[:,1]-cy)**2),2]
    units.append([cx,cy,max(L,5.5),min(max(W,2.4),3.0),hd,100]); box=cv2.boxPoints(((cx,cy),(max(L,5.5)/RES+6,3.0/RES+6),-hd)).astype(np.int32); cv2.fillPoly(taken,[box],255)
# drop the bright gravel patch / equipment corner (kept as an open zone, not stalls)
units=[u for u in units if not (xy(u[0],u[1])[0]<-27 and xy(u[0],u[1])[1]<-28)]
print('units pass1',n0,'pass2',n1,'final',len(units))
# grow each box to the loosest mask extent along its axes (strict thresholds under-measure)
out=[dict(x=round(xy(u[0],u[1])[0],2),y=round(xy(u[0],u[1])[1],2),L=round(u[2],1),W=round(u[3],1),hd=round(u[4],1),th=u[5]) for u in units]
json.dump(out,open('units.json','w'))
ov=m.copy()
loose=((V>200)&(S<32)).astype(np.uint8)*255&reg; loose=cv2.morphologyEx(loose,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
miss=(loose>0)&(taken==0); ov[miss]=(0,0,255)
for u in units:
    box=cv2.boxPoints(((u[0],u[1]),(u[2]/RES,u[3]/RES),-u[4])).astype(np.int32); cv2.polylines(ov,[box],True,(0,200,0) if u[5]>150 else (255,120,0) if u[5]==150 else (255,0,255),2)
for y in range(-120,181,20):
    v=uv(0,y)[1]; cv2.line(ov,(0,v),(ov.shape[1],v),(255,255,255),1); cv2.putText(ov,str(y),(4,v-4),0,0.8,(0,255,255),2)
for x in range(-60,61,20):
    u=uv(x,0)[0]; cv2.line(ov,(u,0),(u,ov.shape[0]),(255,255,255),1); cv2.putText(ov,str(x),(u+3,ov.shape[0]-20),0,0.8,(0,255,255),2)
cv2.imwrite('ov.png',ov)
def crop(nm,xa,xb,ya,yb,f):
    a=uv(xa,yb); b=uv(xb,ya); cv2.imwrite(nm,cv2.resize(ov[a[1]:b[1],a[0]:b[0]],None,fx=f,fy=f,interpolation=cv2.INTER_AREA))
crop('c1.jpg',-64,0,-62,92,0.62); crop('c2.jpg',0,63,-62,92,0.62); crop('c3.jpg',-30,63,150,186,0.9)
lf=((V>198)&(S<34)).astype(np.uint8)*255&reg; lf[taken>0]=0; lf=cv2.morphologyEx(lf,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)); lf=cv2.morphologyEx(lf,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
n,lab,st,cen=cv2.connectedComponentsWithStats(lf)
print('leftover blobs >=3.5m2:',[(round(xy(*cen[i])[0],1),round(xy(*cen[i])[1],1),round(st[i,4]*RES*RES,1)) for i in range(1,n) if st[i,4]*RES*RES>=3.5 and not (xy(*cen[i])[0]<-27 and xy(*cen[i])[1]<-28)])
