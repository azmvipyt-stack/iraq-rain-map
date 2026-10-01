#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""خريطة الهطول المطري التراكمي للعراق: ساعي حتى 15 يوماً + يومي حتى 40 يوماً، دمج النماذج وECMWF منفرداً.
python rainmap.py --state state --out site          # تشغيل عادي
python rainmap.py --demo --state /tmp/s --out /tmp/o  # اختبار بلا إنترنت (بيانات وهمية)"""
import argparse, json, os, sys, time, warnings, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
warnings.simplefilter("ignore")
UA = "IraqRainMap/1.0 (personal use)"
LAT0, LAT1, LON0, LON1, ST, CS = 28.0, 38.5, 36.0, 51.5, 0.5, 3      # نطاق الخريطة، خطوة الشبكة (درجة)، معامل الشبكة الخشنة للمدى الطويل
LATS = np.arange(LAT1, LAT0 - 1e-6, -ST); LONS = np.arange(LON0, LON1 + 1e-6, ST); NY, NX = len(LATS), len(LONS)
HOURS, DAYS = 360, 40
KEY = os.environ.get("OPENMETEO_KEY", "").strip()                      # اختياري: مفتاح مدفوع يرفع الحصة ويسمح بشبكة 0.25°
REF_H = float(os.environ.get("REFRESH_H", "3")); LONG_H = float(os.environ.get("LONG_REFRESH_H", "12"))
MOD = {"ecmwf_ifs": "ecmwf", "ecmwf_ifs025": "ecmwf2", "ecmwf_aifs025_single": "aifs", "gfs_seamless": "gfs", "icon_seamless": "icon"}
W_H = {"ecmwf": .40, "aifs": .25, "gfs": .20, "icon": .15}            # أوزان الدمج (ثابتة، تُعاد موازنتها حيث يغيب نموذج)
W_L = (.6, .4)                                                          # EC46 ثم GEFS بعد اليوم 15
DEMO = False


def U(u):
    return u.replace("://", "://customer-", 1) if KEY else u


def gj(url, tries=4):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=240) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = RuntimeError(f"HTTP {e.code} {e.read()[:120]}")
            if e.code == 429 or e.code >= 500:
                time.sleep(25 * (i + 1)); continue
            raise err
        except Exception as e:
            err = e; time.sleep(5 * (i + 1))
    raise err


def multi(base, pts, params, chunk):
    out = []
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        q = dict(params, latitude=",".join("%.3f" % a for a, b in p), longitude=",".join("%.3f" % b for a, b in p))
        if KEY: q["apikey"] = KEY
        d = gj(U(base) + "?" + urllib.parse.urlencode(q)); out += d if isinstance(d, list) else [d]; time.sleep(.5)
    return out


def demo(T, seed, sh=None):
    sh = sh or (NY, NX); r = np.random.RandomState(seed); yy, xx = np.mgrid[0:sh[0], 0:sh[1]]; o = np.zeros((T,) + sh, np.float32)
    for _ in range(7):
        cy, cx, s, t0, du, a = r.uniform(0, sh[0]), r.uniform(0, sh[1]), r.uniform(2, 6) * sh[0] / NY, r.uniform(0, T), r.uniform(T * .05, T * .2), r.uniform(.1, 1.2)
        o += (a * np.exp(-((np.arange(T) - t0) / du) ** 2))[:, None, None] * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))[None]
    return o


# ------------------------------------------------------------- الساعي (0-15 يوماً)
def get_hourly():
    if DEMO:
        T0 = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return {k: demo(HOURS, i) * (1 + .15 * i) for i, k in enumerate(MOD.values())}, T0
    pts = [(a, b) for a in LATS for b in LONS]; F = "https://api.open-meteo.com/v1/forecast"; A = {}; T0 = None
    def run(ms):
        return multi(F, pts, dict(hourly="precipitation", models=",".join(ms), forecast_days=15, timezone="GMT"), 40)
    def fill(res, ms):
        nonlocal T0
        T0 = T0 or datetime.strptime(res[0]["hourly"]["time"][0][:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        for m in ms:
            a = np.full((HOURS, NY, NX), np.nan, np.float32)
            for k, d in enumerate(res):
                v = d["hourly"].get("precipitation_" + m) if len(ms) > 1 else d["hourly"].get("precipitation")
                if v:
                    x = np.array([np.nan if t is None else t for t in v[:HOURS]], np.float32); a[:len(x), k // NX, k % NX] = x
            if np.isfinite(a).any(): A[MOD[m]] = a
    try:
        ms = list(MOD); fill(run(ms), ms)
    except Exception as e:
        print("! الطلب المجمّع فشل، نجرب كل نموذج وحده:", e)
        for m in MOD:
            try: fill(run([m]), [m])
            except Exception as e2: print("! تعذّر", m, e2)
    if not A: raise RuntimeError("no hourly data")
    return A, T0


def hproc(A):
    if "ecmwf2" in A:
        A["ecmwf"] = np.where(np.isnan(A["ecmwf"]), A["ecmwf2"], A["ecmwf"]) if "ecmwf" in A else A["ecmwf2"]
    num = np.zeros((HOURS, NY, NX), np.float32); den = np.zeros_like(num)
    for m, w in W_H.items():
        if m in A:
            ok = np.isfinite(A[m]); num += np.where(ok, A[m], 0) * w; den += ok * w
    H = {"blend": np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)}
    if "ecmwf" in A: H["ecmwf"] = A["ecmwf"]
    return H


# ------------------------------------------------------------- المدى الطويل (حتى 40 يوماً)
def members_mean(b):
    ms = [[np.nan if x is None else x for x in v] for k, v in b.items() if k.startswith("precipitation") and isinstance(v, list)]
    if not ms: return None
    n = min(map(len, ms)); return np.nanmean(np.array([m[:n] for m in ms], np.float32), axis=0)


def to_days(times, v, T0):
    s = np.zeros(DAYS); c = np.zeros(DAYS)
    for t, x in zip(times, v):
        if np.isnan(x): continue
        k = int(((datetime.strptime(t[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc) - T0).total_seconds() - 60) // 86400)
        if 0 <= k < DAYS: s[k] += x; c[k] += 1
    return np.where(c > 0, s, np.nan)


def coarse_pts():
    la, lo = LATS[::CS], LONS[::CS]; return [(a, b) for a in la for b in lo], len(la), len(lo)


def get_long(T0, kind):
    pts, cy, cx = coarse_pts()
    if DEMO:
        o = demo(DAYS, 11 if kind == "ec" else 12, (cy, cx)) * 3
        if kind == "ge": o[35:] = np.nan
        return o
    if kind == "ec":
        base, tries, ch = "https://seasonal-api.open-meteo.com/v1/seasonal", [dict(six_hourly="precipitation", models="ecmwf_ec46", forecast_days=46), dict(hourly="precipitation", models="ecmwf_ec46", forecast_days=46)], 10
    else:
        base, tries, ch = "https://ensemble-api.open-meteo.com/v1/ensemble", [dict(hourly="precipitation", models=m, forecast_days=35) for m in ("gfs05", "gfs_seamless")], 8
    err = None
    for p in tries:
        try:
            res = multi(base, pts, dict(p, timezone="GMT"), ch); nm = "six_hourly" if "six_hourly" in p else "hourly"
            o = np.full((DAYS, cy, cx), np.nan, np.float32)
            for k, d in enumerate(res):
                b = d[nm]; o[:, k // cx, k % cx] = to_days(b["time"], members_mean(b), T0)
            if np.isfinite(o).any(): return o
        except Exception as e:
            err = e; print("!", kind, e)
    raise err or RuntimeError(kind)


def up(c):
    cy, cx = c.shape[1:]; ys = np.minimum(np.arange(NY) / CS, cy - 1); xs = np.minimum(np.arange(NX) / CS, cx - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, cy - 1); wy = (ys - y0)[None, :, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, cx - 1); wx = (xs - x0)[None, None, :]
    r = c[:, y0, :] * (1 - wy) + c[:, y1, :] * wy
    return r[:, :, x0] * (1 - wx) + r[:, :, x1] * wx


def lblend(ec, ge):
    if ec is None or ge is None: return ec if ge is None else ge
    w = np.stack([np.where(np.isnan(ec), 0, W_L[0]), np.where(np.isnan(ge), 0, W_L[1])]); v = np.stack([np.nan_to_num(ec), np.nan_to_num(ge)]); d = w.sum(0)
    return np.where(d > 0, (w * v).sum(0) / np.maximum(d, 1e-9), np.nan)


def to40(hc, vh, inc):
    C = np.full((DAYS + 1, NY, NX), np.nan, np.float32); C[0] = 0; dh = min(15, vh // 24)
    for d in range(1, dh + 1): C[d] = hc[24 * d]
    if inc is not None:
        u = up(inc)
        for d in range(dh + 1, DAYS + 1): C[d] = C[d - 1] + u[d - 1]
    return C


def lastvalid(a):
    return max([i for i in range(a.shape[0]) if np.isfinite(a[i]).mean() > .5] or [0])


def enc(a):
    return np.where(np.isnan(a), 65535, np.clip(np.round(a * 10), 0, 65000)).astype("<u2")


def jl(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return {}


def main():
    global DEMO
    ap = argparse.ArgumentParser(); ap.add_argument("--state", default="state"); ap.add_argument("--out", default="site"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(); DEMO = a.demo
    os.makedirs(a.state, exist_ok=True); os.makedirs(a.out + "/data", exist_ok=True)
    now = datetime.now(timezone.utc); sp = a.state + "/stamp.json"; S = jl(sp)
    age = lambda k: (now - datetime.fromisoformat(S[k])).total_seconds() / 3600 if k in S else 1e9
    hp, lp = a.state + "/h.npz", a.state + "/l.npz"
    if DEMO or age("h") >= REF_H - .25 or not os.path.exists(hp):
        try:
            A, T0 = get_hourly(); np.savez_compressed(hp, **hproc(A)); S["h"] = now.isoformat(); S["t0"] = T0.isoformat()
        except Exception as e:
            print("! الساعي:", e)
            if not os.path.exists(hp): sys.exit("✗ لا بيانات")
    H = dict(np.load(hp)); T0 = datetime.fromisoformat(S["t0"])
    if DEMO or age("l") >= LONG_H or S.get("lt0") != S["t0"] or not os.path.exists(lp):
        prev = dict(np.load(lp)) if os.path.exists(lp) and S.get("lt0") == S["t0"] else {}; L = {}
        for k in ("ec", "ge"):
            try: L[k] = get_long(T0, k)
            except Exception as e:
                print("! المدى الطويل", k, e)
                if k in prev: L[k] = prev[k]
        if L: np.savez_compressed(lp, **L); S["l"] = now.isoformat(); S["lt0"] = S["t0"]
    L = dict(np.load(lp)) if os.path.exists(lp) and S.get("lt0") == S["t0"] else {}
    json.dump(S, open(sp, "w"))
    prods = {}
    for k, lab in (("blend", "دمج أدق النماذج"), ("ecmwf", "ECMWF")):
        if k not in H: continue
        raw = H[k]; vh = lastvalid(raw) + 1; hc = np.concatenate([np.zeros((1, NY, NX), np.float32), np.cumsum(np.nan_to_num(raw), 0)])
        c40 = to40(hc, vh, lblend(L.get("ec"), L.get("ge")) if k == "blend" else L.get("ec")); vd = lastvalid(c40)
        enc(hc[:vh + 1]).tofile(f"{a.out}/data/{k}_h.bin"); enc(c40[:vd + 1]).tofile(f"{a.out}/data/{k}_d.bin")
        prods[k] = {"label": lab, "h": vh, "d": vd}; print("✓", k, "ساعات:", vh, "أيام:", vd)
    meta = {"updated": (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"), "t0": T0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid": {"lat0": float(LAT1), "lon0": float(LON0), "st": ST, "ny": NY, "nx": NX}, "prods": prods}
    json.dump(meta, open(a.out + "/data/meta.json", "w"), ensure_ascii=False)
    open(a.out + "/index.html", "w", encoding="utf-8").write(PAGE)
    print("✓ تم:", a.out)


PAGE = r"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#14202b"><title>خريطة الهطول المطري — العراق</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<link href="https://fonts.googleapis.com/css2?family=Readex+Pro:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{display:flex;flex-direction:column;font:400 13px/1.5 "Readex Pro",Tahoma,sans-serif;background:#14202b;color:#e6eef4;padding-top:env(safe-area-inset-top)}
.top{padding:6px 8px 4px}h1{font-size:15px;margin:0 0 4px;font-weight:600}h1 small{font-weight:400;color:#8fa6b6;font-size:11px}
.row{display:flex;gap:5px;overflow-x:auto;scrollbar-width:none;margin:3px 0}.row::-webkit-scrollbar{display:none}
button{font:inherit;font-size:12px;color:#e6eef4;background:#1e3142;border:1px solid #38566d;border-radius:7px;padding:4px 10px;min-height:34px;white-space:nowrap;cursor:pointer;flex:none}
button.on{background:#4fb6dd;color:#0d1a24;border-color:#4fb6dd;font-weight:600}
#map{flex:1;min-height:200px;background:#e8e8ee}.leaflet-container{font-family:inherit}
.bot{background:#fff;color:#111;padding:6px 8px calc(6px + env(safe-area-inset-bottom));direction:ltr}
.hd{display:flex;justify-content:space-between;align-items:flex-end;gap:8px;font-size:11px;line-height:1.35}.hd b{font-size:13.5px}.hd span{text-align:right;color:#333}
.sl{display:flex;gap:8px;align-items:center;margin:4px 0}.sl input{flex:1;accent-color:#1f6fd1;height:28px}.sl button{background:#e8edf2;color:#111;border-color:#b9c5cf;min-height:30px}
.lg{display:flex;margin-top:2px}.lg i{flex:1;height:12px;position:relative;font-style:normal}.lg b{position:absolute;top:13px;left:-50%;width:200%;text-align:center;font:500 8px/1 "Readex Pro",sans-serif;color:#222}
.lg i:first-child b{left:-5%}.lg{margin-bottom:12px}
@media(max-width:560px){.lg i:nth-child(even) b{display:none}}
.note{color:#51606c;font-size:10px;margin-top:4px;direction:rtl;line-height:1.5}
</style></head><body>
<div class="top"><h1>خريطة الهطول المطري التراكمي — العراق <small id="up"></small></h1>
<div class="row" id="b1"></div><div class="row" id="b2"></div><div class="row" id="b3"></div></div>
<div id="map"></div>
<div class="bot"><div class="hd"><b>Accumulated total precipitation (mm)</b><span id="tt"></span></div>
<div class="sl"><button id="pl">▶</button><input id="rg" type="range" min="1" max="24" value="24"></div>
<div class="lg" id="lg"></div>
<div class="note">المصدر: Open-Meteo (ECMWF IFS/AIFS وGFS وICON حتى 15 يوماً؛ ECMWF EC46 وGEFS بعد ذلك). الشبكة 0.5° مُستوفاة بنعومة. الدمج بأوزان ثابتة غير معايَرة، وما بعد ~10 أيام ميل عام لا تنبؤ موضعي. المس الخريطة لقراءة القيمة. ليس تحذيراً رسمياً.</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const LV=[.1,1,2,3,5,7,10,15,20,25,30,40,50,60,70,80,90,100,125,150,175,200,250,300,400,500];
const CO=['#dcdcf2','#bcd2fb','#8fb8fb','#5f9bfb','#2f7df7','#0a5be0','#0a2fa6','#168a1c','#26b82a','#6de024','#f4f400','#e6d200','#ff7a00','#ff9a52','#ffb079','#ff5a8c','#ff2d6a','#e0002a','#a40018','#6e0014','#7d1b8f','#bf22e0','#d066ff','#e3a0ff','#f3dcff','#cfcfcf'];
const RGB=CO.map(h=>[1,3,5].map(i=>parseInt(h.substr(i,2),16))),$=s=>document.querySelector(s);
const WINS={h:[[1,'1س'],[3,'3س'],[6,'6س'],[12,'12س'],[24,'24س'],[72,'3 أيام'],[168,'7 أيام'],[0,'من البداية']],d:[[1,'يوم'],[3,'3 أيام'],[7,'7 أيام'],[10,'10 أيام'],[15,'15 يوماً'],[0,'من البداية']]};
let M,S={prod:'blend',per:'h',win:24,end:24},cache={},D,Fv,map,cv,cx,timer=0;
const fmt=ms=>new Date(ms).toLocaleString('ar-IQ-u-nu-latn',{timeZone:'Asia/Baghdad',weekday:'short',day:'numeric',month:'numeric',hour:'2-digit',minute:'2-digit',hour12:false});
const unit=()=>S.per=='h'?36e5:864e5,sm=t=>t*t*(3-2*t);
function bar(id,items,key){$(id).innerHTML=items.map(([v,l])=>`<button data-k="${key}" data-v="${v}" class="${S[key]==v?'on':''}">${l}</button>`).join('')}
function ui(){bar('#b1',[['h','ساعي · حتى 15 يوماً'],['d','تراكمي يومي · حتى 40 يوماً']],'per');
bar('#b2',Object.entries(M.prods).map(([k,p])=>[k,p.label]),'prod');bar('#b3',WINS[S.per],'win')}
async function load(){const k=S.prod+S.per;if(!cache[k]){const r=await fetch(`data/${S.prod}_${S.per}.bin?v=${encodeURIComponent(M.updated)}`);cache[k]=new Uint16Array(await r.arrayBuffer())}D=cache[k]}
function field(){const g=M.grid,n=g.ny*g.nx,e=S.end,f=S.win?Math.max(0,e-S.win):0,o=new Float32Array(n);
for(let i=0;i<n;i++){const x=D[e*n+i],y=D[f*n+i];o[i]=(x==65535||y==65535)?-1:(x-y)/10}return o}
function samp(lat,lng){const g=M.grid,fy=(g.lat0-lat)/g.st,fx=(lng-g.lon0)/g.st;if(fy<0||fx<0||fy>g.ny-1||fx>g.nx-1)return -1;
const y=Math.min(fy|0,g.ny-2),x=Math.min(fx|0,g.nx-2),ty=sm(fy-y),tx=sm(fx-x),i=y*g.nx+x,a=Fv[i],b=Fv[i+1],c=Fv[i+g.nx],e=Fv[i+g.nx+1];
if(a<0||b<0||c<0||e<0)return -1;return(a*(1-tx)+b*tx)*(1-ty)+(c*(1-tx)+e*tx)*ty}
function draw(){if(!Fv)return;const z=map.getSize(),W=z.x,H=z.y,g=M.grid;if(cv.width!=W||cv.height!=H){cv.width=W;cv.height=H}
L.DomUtil.setPosition(cv,map.containerPointToLayerPoint([0,0]));
const a=map.containerPointToLatLng([0,0]),b=map.containerPointToLatLng([W,H]),img=cx.createImageData(W,H),d=img.data;
const fxs=new Float32Array(W);for(let x=0;x<W;x++)fxs[x]=(a.lng+(b.lng-a.lng)*x/W-g.lon0)/g.st;
for(let y=0;y<H;y++){const fy=(g.lat0-map.containerPointToLatLng([0,y]).lat)/g.st;if(fy<0||fy>g.ny-1)continue;
const y0=Math.min(fy|0,g.ny-2),ty=sm(fy-y0);
for(let x=0;x<W;x++){const fx=fxs[x];if(fx<0||fx>g.nx-1)continue;const x0=Math.min(fx|0,g.nx-2),tx=sm(fx-x0),i=y0*g.nx+x0;
const p=Fv[i],q=Fv[i+1],r=Fv[i+g.nx],s=Fv[i+g.nx+1];if(p<0||q<0||r<0||s<0)continue;
const v=(p*(1-tx)+q*tx)*(1-ty)+(r*(1-tx)+s*tx)*ty;if(v<.1)continue;let k=0;while(k<25&&v>=LV[k+1])k++;
const j=(y*W+x)*4,c=RGB[k];d[j]=c[0];d[j+1]=c[1];d[j+2]=c[2];d[j+3]=215}}
cx.putImageData(img,0,0)}
function upd(){const mx=M.prods[S.prod][S.per];S.end=Math.max(1,Math.min(S.end,mx));const r=$('#rg');r.max=mx;r.value=S.end;
Fv=field();const t0=Date.parse(M.t0),u=unit(),f=S.win?Math.max(0,S.end-S.win):0;
$('#tt').innerHTML=`From ${fmt(t0+f*u)}<br>to ${fmt(t0+S.end*u)} · GMT+3`;draw()}
async function sw(){ui();await load();upd()}
document.addEventListener('click',async e=>{const t=e.target.closest('button[data-k]');if(!t)return;const k=t.dataset.k,v=t.dataset.v;
if(k=='per'){S.per=v;S.win=v=='h'?24:7;S.end=v=='h'?24:7}else if(k=='prod')S.prod=v;else S.win=+v;await sw()});
$('#rg').addEventListener('input',e=>{S.end=+e.target.value;upd()});
$('#pl').onclick=()=>{if(timer){clearInterval(timer);timer=0;$('#pl').textContent='▶';return}$('#pl').textContent='❚❚';
timer=setInterval(()=>{const mx=M.prods[S.prod][S.per];S.end=S.end>=mx?1:S.end+(S.per=='h'?3:1);upd()},350)};
(async()=>{M=await (await fetch('data/meta.json?'+Date.now())).json();$('#up').textContent='· آخر تحديث '+M.updated+' (توقيت العراق)';
$('#lg').innerHTML=LV.map((v,i)=>`<i style="background:${CO[i]}"><b>${v}</b></i>`).join('');
map=L.map('map',{zoomAnimation:false,minZoom:5,maxZoom:10,zoomSnap:.5}).fitBounds([[28,36],[38.5,51.5]]);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',{subdomains:'abcd',attribution:'© OpenStreetMap © CARTO'}).addTo(map);
map.createPane('rain').style.zIndex=250;map.getPane('rain').style.pointerEvents='none';
cv=L.DomUtil.create('canvas','',map.getPane('rain'));cv.style.cssText='position:absolute;left:0;top:0';cx=cv.getContext('2d');
map.createPane('lab').style.zIndex=300;map.getPane('lab').style.pointerEvents='none';
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png',{subdomains:'abcd',pane:'lab'}).addTo(map);
map.on('moveend zoomend resize',draw);
map.on('click',e=>{const v=samp(e.latlng.lat,e.latlng.lng);if(v<0)return;L.popup().setLatLng(e.latlng).setContent(`<b>${v.toFixed(1)}</b> مم`).openOn(map)});
await sw()})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
