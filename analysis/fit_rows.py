import cv2, numpy as np, json, math
RES=0.1; X0,Y1=-68,189
U=json.load(open('units.json'))
U+= [dict(x=52.4,y=54.9,L=6.5,W=2.6,hd=None,th=0),dict(x=52.4,y=45.9,L=6.5,W=2.6,hd=None,th=0),dict(x=25.6,y=-20.9,L=6.5,W=2.6,hd=None,th=0),dict(x=30.3,y=168.5,L=6.5,W=2.6,hd=None,th=0)]
seeds={'A':((-57.0,88.5),(-47.5,-8.9)),'B1':((-27.6,86.0),(-26.9,42.8)),'B2':((-23.4,28.6),(-20.8,-32.2)),'C':((-9.7,50.5),(-6.4,-52.0)),
 'D':((11.3,80.7),(15.8,-56.6)),'E':((25.9,79.2),(29.5,-54.4)),'F':((48.8,125.1),(54.2,-53.3)),'N':((-21,177),(31,177)),'M':((-21,168.8),(31,168.8))}
def dseg(p,a,b):
    a,b,p=map(np.array,(a,b,p)); t=np.clip((p-a)@(b-a)/((b-a)@(b-a)),-0.08,1.08); return np.linalg.norm(p-(a+t*(b-a)))
for u in U:
    d={k:dseg((u['x'],u['y']),*v) for k,v in seeds.items()}; k=min(d,key=d.get); u['row']=k if d[k]<7.5 else None
print('unassigned',[(u['x'],u['y']) for u in U if u['row'] is None])
rows=[];slots=[];report=[]
order=['A','B1','B2','C','D','E','F','N','M']
cnt={}
for k in order:
    us=[u for u in U if u['row']==k]
    P=np.array([[u['x'],u['y']] for u in us]); c=P.mean(0)
    d=np.linalg.svd(P-c,full_matrices=False)[2][0]
    ew=k in 'NM'
    if (ew and d[0]<0) or (not ew and d[1]>0): d=-d
    t=(P-c)@d; o=np.argsort(t); t=t[o]; us=[us[i] for i in o]
    nv=np.array([-d[1],d[0]]); lat=((P-c)@nv)[o]
    latm=np.array([np.median(lat[max(0,i-1):i+2]) for i in range(len(lat))])
    hds=[u['hd'] for u in us if u['hd'] is not None and u['L']/u['W']>2.2]; hd=float(np.median(hds))
    gaps=np.diff(t); pitch=float(np.median(gaps[gaps<6.5]))
    slen=float(np.clip(np.percentile([u['L'] for u in us],85)+0.8,7,13.5))
    # perpendicular clearance between neighbours
    ang=math.radians(hd)-math.atan2(d[1],d[0]); perp=abs(pitch*math.sin(ang)); sw=round(min(3.4,max(2.7,perp-0.15)),2)
    # blocks
    blocks=[[t[0]]]
    for g,tt in zip(gaps,t[1:]):
        if g>1.6*pitch: blocks.append([tt])
        else: blocks[-1].append(tt)
    pos=[]; filled=0; aisles=[]
    for bi,b in enumerate(blocks):
        if bi>0:
            g=b[0]-blocks[bi-1][-1]; kfill=int(round(g/pitch))-1
            if kfill>=1 and (ew or g<=3.7*pitch):
                pos+= [(blocks[bi-1][-1]+g*(j+1)/(kfill+1),False) for j in range(kfill)]; filled+=kfill
            else: aisles.append(round(float(g),1))
        n=len(b)
        pos+= [((b[0]+(b[-1]-b[0])*j/(n-1)) if n>1 else b[0],True) for j in range(n)]
    rid=k[0]; base=cnt.get(rid,0)
    for j,(tt,occ) in enumerate(pos):
        p=c+d*tt+nv*float(np.interp(tt,t,latm)); slots.append(dict(id=f"{rid}-{base+j+1:02d}",row=rid,index=base+j+1,x=round(float(p[0]),2),y=round(float(p[1]),2),heading_deg=round(hd,1),width_m=sw,length_m=round(slen,1),occupied_in_imagery=occ))
    cnt[rid]=base+len(pos)
    a=c+d*t[0]; b=c+d*t[-1]
    rows.append(dict(key=k,start=[round(float(a[0]),2),round(float(a[1]),2)],end=[round(float(b[0]),2),round(float(b[1]),2)],heading_deg=round(hd,1),pitch_m=round(pitch,2),stall_length_m=round(slen,1),stall_width_m=sw,units_in_imagery=len(us),stalls=len(pos),aisle_gaps_m=aisles))
    print(k,'units',len(us),'stalls',len(pos),'filled',filled,'pitch %.2f hd %.0f len %.1f w %.2f'%(pitch,hd,slen,sw),'aisles',aisles,'resid %.2f'%np.abs((P-c)@np.array([-d[1],d[0]])).mean())
print('total units',len(U),'total stalls',len(slots))
med=np.median([u['hd'] for u in U if u['hd'] is not None and u['row'] and u['row'] not in 'NM'])
for u in U:
    if u['hd'] is None: u['hd']=95.0 if u['row'] in ('N','M') else float(med)
json.dump(dict(rows=rows,slots=slots,units=[dict(x=u['x'],y=u['y'],l=u['L'],w=u['W'],h=round(u['hd'])) for u in U]),open('stalls.json','w'))
m=cv2.imread('mos.png')
def uv(x,y): return (x-X0)/RES,(Y1-y)/RES
for s in slots:
    box=cv2.boxPoints((uv(s['x'],s['y']),(s['length_m']/RES,s['width_m']/RES),-s['heading_deg'])).astype(np.int32)
    cv2.polylines(m,[box],True,(0,200,0) if s['occupied_in_imagery'] else (0,0,255),2)
def crop(nm,xa,xb,ya,yb,f):
    a=uv(xa,yb); b=uv(xb,ya); cv2.imwrite(nm,cv2.resize(m[int(a[1]):int(b[1]),int(a[0]):int(b[0])],None,fx=f,fy=f,interpolation=cv2.INTER_AREA))
crop('s1.jpg',-64,0,-62,92,0.62); crop('s2.jpg',0,63,-62,130,0.5); crop('s3.jpg',-30,40,155,186,0.9)
