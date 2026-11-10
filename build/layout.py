import json, math, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src=json.load(open(os.path.join(HERE, 'detected_units.json'))); units=src['units']
PITCH=4.0; HD=40.0; WID=2.95
N_TOP, X_N, X_S = 152.0, 104.0, 92.0     # north cross aisle is y 152-165, middle cross aisle is y 92-104
# row centre-lines as (y, x) knots, taken from the lines fitted to the parked units and extended
def line(p,q,ya,yb):
    (x0,y0),(x1,y1)=p,q; f=lambda y: x0+(x1-x0)*(y-y0)/(y1-y0); return [(ya,f(ya)),(yb,f(yb))]
R={
 'A': dict(knots=[(152,-57.8),(88.5,-57.0),(-30,-45.4)], blocks=[(152,-30)], length=11.4),
 'B': dict(knots=[(152,-28.7),(-56,-22.5)], blocks=[(152,X_N),(X_S,-56)], length=11.5),
 'C': dict(knots=line((-9.7,50.5),(-6.4,-52.0),152,-56), blocks=[(152,X_N),(X_S,-56)], length=9.2),
 'D': dict(knots=line((11.3,80.7),(15.8,-56.6),152,-56), blocks=[(152,X_N),(X_S,-56)], length=9.2),
 'E': dict(knots=line((25.9,79.2),(29.5,-54.4),152,-56), blocks=[(152,X_N),(X_S,-56)], length=8.8),
 'F': dict(knots=line((48.8,125.1),(54.2,-53.3),152,-56), blocks=[(152,-56)], length=8.9),
}
def xat(k,y):
    ys=[a for a,_ in k]; xs=[b for _,b in k]; return float(np.interp(-y,[-a for a in ys],xs))
U=np.array([[u['x'],u['y']] for u in units])
def occ(x,y): return bool((np.hypot(U[:,0]-x,U[:,1]-y)<2.6).any())
rows=[];slots=[]
for rid,r in R.items():
    i=0; segs=[]
    for ya,yb in r['blocks']:
        n=int((ya-yb)//PITCH); off=((ya-yb)-n*PITCH)/2
        for j in range(n):
            y=ya-off-PITCH*(j+0.5); x=xat(r['knots'],y); i+=1
            slots.append(dict(id=f"{rid}-{i:02d}",row=rid,index=i,x=round(x,2),y=round(y,2),heading_deg=HD,width_m=WID,length_m=r['length'],occupied_in_imagery=occ(x,y)))
        segs.append([[round(xat(r['knots'],ya),2),ya],[round(xat(r['knots'],yb),2),yb]])
    rows.append(dict(id=rid,name=f"Row {rid}",segments=segs,knots=[[round(x,2),y] for y,x in r['knots']],heading_deg=HD,pitch_m=PITCH,stall_length_m=r['length'],stall_width_m=WID,slot_count=i,numbering="01 = north end"))
# north ranks (units nose-to-tail in two ranks along the north edge)
for rid,y,L in (('N',177.0,7.8),('M',168.8,7.2)):
    xs=np.arange(-56,52.1,3.8); i=0
    for x in xs:
        i+=1; slots.append(dict(id=f"{rid}-{i:02d}",row=rid,index=i,x=round(float(x),2),y=y,heading_deg=95.0,width_m=3.3,length_m=L,occupied_in_imagery=occ(x,y)))
    rows.append(dict(id=rid,name=f"Row {rid}",segments=[[[-56,y],[round(float(xs[-1]),2),y]]],knots=[[-56,y],[round(float(xs[-1]),2),y]],heading_deg=95.0,pitch_m=3.8,stall_length_m=L,stall_width_m=3.3,slot_count=i,numbering="01 = west end"))
for r in rows: r['units_in_imagery']=sum(1 for s in slots if s['row']==r['id'] and s['occupied_in_imagery'])
# drive aisles: centre-lines midway between neighbouring rows
def mid(a,b,ys): return [[round((xat(R[a]['knots'],y)+xat(R[b]['knots'],y))/2,2),y] for y in ys]
ys=[158,88.5,-30,-56]
aisles=[dict(id='AISLE-1',name='Aisle 1 (A-B)',line=mid('A','B',[158,88.5,-30])),
        dict(id='AISLE-2',name='Aisle 2 (B-C)',line=mid('B','C',ys)),
        dict(id='AISLE-3',name='Aisle 3 (C-D)',line=mid('C','D',ys)),
        dict(id='AISLE-4',name='Aisle 4 (E-F)',line=mid('E','F',ys)),
        dict(id='CROSS-N',name='North cross aisle',line=[[-52,158.5],[54,158.5]]),
        dict(id='CROSS-M',name='Middle cross aisle',line=[[-46,98],[42,98]]),
        dict(id='CROSS-S',name='South cross aisle',line=[[-38,-64],[42,-64]])]
json.dump(dict(rows=rows,slots=slots,units=units,aisles=aisles),open(os.path.join(HERE, 'stalls.json'),'w'))
print(len(slots),{r['id']:r['slot_count'] for r in rows},'photo-occupied',sum(s['occupied_in_imagery'] for s in slots))
