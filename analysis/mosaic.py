import cv2, numpy as np, math
D='/tmp/claude-tool-results/6324aa70-4b4b-5b18-aae3-e185c1ce0b63/'
LAT0,LNG0=41.5533,-85.8822; M_LAT=111070.0; M_LNG=111320*math.cos(math.radians(LAT0))
def mpp(lat): return 156543.03392*math.cos(math.radians(lat))/2**20*1.5
shots=[('mcp-remote-devices-blob-1794254110491-rbndfx-349f80fe.jpg',41.552895),('mcp-remote-devices-blob-1794254089864-d0szom-d9f3251f.jpg',41.554245)]
LNGC=-85.882218
RES=0.1; X0,X1,Y0,Y1=-68,65,-128,189
W=int((X1-X0)/RES); H=int((Y1-Y0)/RES); mos=np.zeros((H,W,3),np.uint8)
for f,lat in shots:
    im=cv2.imread(D+f); s=mpp(lat); cx=(LNGC-LNG0)*M_LNG; cy=(lat-LAT0)*M_LAT
    A=np.array([[RES/s,0,(X0-cx)/s+400],[0,RES/s,500-(Y1-cy)/s]],np.float32)
    w=cv2.warpAffine(im,A,(W,H),flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)
    m=w.sum(2)>0; mos[m]=w[m]; print(s,cx,cy,im.shape)
cv2.imwrite('mos.png',mos)
