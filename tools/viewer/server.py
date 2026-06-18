"""Interactive scene inspector — VAD-viewer feel (stdlib http.server, no deps).

  python tools/viewer/server.py 8077   ->   http://localhost:8077

Left: 6 individual camera images. Right: client-side HTML5 Canvas BEV (dark, with
mouse zoom/pan and instant layer toggles). Top nav lets you filter scenes by the
analysis strata (object density / maneuver / map complexity) and ranking.
"""
import os, sys, json
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import lib_io
import geom

TASK2VAR = {'det': 'no_det', 'map': 'no_map', 'motion': 'no_motion'}
_TET = pd.read_csv(os.path.join(lib_io.CACHE_DIR, 'task_epoch_table.csv'))
_CT = pd.read_csv(os.path.join(lib_io.CACHE_DIR, 'collision_table.csv'))
DF = _TET.merge(_CT[['token'] + [c for c in _CT.columns if c.startswith('col_e')]], on='token', how='left')


def scene_rows(epoch, task, rank, density, maneuver, mapcx, limit=400):
    var = TASK2VAR[task]; d = DF
    if density == 'dense':
        d = d[d['n_near15'] >= 6]
    elif density == 'sparse':
        d = d[d['n_near15'] < 6]
    if maneuver == 'straight':
        d = d[d['is_turn'] == 0]
    elif maneuver == 'turn':
        d = d[d['is_turn'] == 1]
    if mapcx == 'complex':
        d = d[d['map_curv'] >= 15]
    elif mapcx == 'simple':
        d = d[d['map_curv'] < 15]
    if rank == 'showcase':
        d = d[(d[f'col_e{epoch}_{var}'] > 0) & (d[f'col_e{epoch}_full'] == 0)].sort_values('n_near15', ascending=False)
    elif rank == 'safety':
        d = d.sort_values(f'Ccol_{task}_e{epoch}', ascending=False)
    elif rank == 'accuracy':
        d = d.sort_values(f'CL2_{task}_e{epoch}', ascending=False)
    else:
        d = d.reindex(d[f'CL2_{task}_e{epoch}'].abs().sort_values(ascending=False).index)
    return [{'token': r['token'], 'n_near15': int(r['n_near15']), 'cmd': int(r['is_turn']),
             'CL2': round(float(r[f'CL2_{task}_e{epoch}']), 3),
             'Ccol': round(float(r[f'Ccol_{task}_e{epoch}']) * 100, 2)} for _, r in d.head(limit).iterrows()]


def img_bytes(token, cam):
    infos, t2i = lib_io.load_val_infos()
    p = infos[t2i[token]]['cams'][cam]['data_path']
    p = p[2:] if p.startswith('./') else p
    with open(os.path.join(lib_io.REPO, p), 'rb') as f:
        return f.read()


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>HiP-AD Inspector</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:system-ui,sans-serif;background:#0c0c0e;color:#e7e7ea}
.bar{padding:5px 12px;background:#16161a;border-bottom:1px solid #26262c;display:flex;gap:11px;align-items:center;flex-wrap:wrap}
select,button{font-size:13px;padding:3px 7px;background:#1f1f24;color:#e7e7ea;border:1px solid #34343c;border-radius:5px}
button{cursor:pointer}button:hover{background:#2b2b32}
label{font-size:11px;color:#8e8e98;display:flex;gap:3px;align-items:center}
.lay{display:flex;gap:8px;flex-wrap:wrap}.lay label{color:#cfcfd6;font-size:12px}
#meta{font-family:monospace;font-size:12px;color:#7dd3fc;margin-left:auto}
.wrap{display:flex;gap:6px;padding:6px;height:calc(100vh - 80px)}
.cams{width:32%;display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;align-content:start}
.camcell{position:relative;aspect-ratio:16/9;background:#000;border:1px solid #26262c;border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.camcell img{width:100%;height:100%;object-fit:contain;display:block}
.camcell span{position:absolute;top:2px;left:4px;font-size:10px;color:#9cf;background:rgba(0,0,0,.5);padding:0 3px;border-radius:3px}
.bevwrap{flex:1;display:flex;gap:6px}
.canvbox{flex:1;position:relative;background:#111;border:1px solid #26262c;border-radius:6px;overflow:hidden}
.canvbox canvas{width:100%;height:100%;display:block;cursor:grab}
.canvbox .tag{position:absolute;top:4px;left:8px;font:12px monospace;color:#cdd}
.hint{position:absolute;bottom:4px;right:8px;font:10px monospace;color:#667}
</style></head><body>
<div class=bar>
 <label>epoch<select id=epoch><option>1</option><option>3</option><option selected>9</option></select></label>
 <label>ablate<select id=task><option>det</option><option>map</option><option>motion</option></select></label>
 <label>density<select id=density><option value=all>all</option><option value=dense>dense(near≥6)</option><option value=sparse>sparse</option></select></label>
 <label>maneuver<select id=maneuver><option value=all>all</option><option value=straight>straight</option><option value=turn>turn</option></select></label>
 <label>map<select id=mapcx><option value=all>all</option><option value=complex>complex</option><option value=simple>simple</option></select></label>
 <label>rank<select id=rank><option value=showcase>ablated collides,full safe</option><option value=safety>top C_col</option><option value=accuracy>top C_L2</option><option value=divergence>top |C_L2|</option></select></label>
 <button onclick=step(-1)>&larr;</button><button onclick=step(1)>&rarr;</button>
 <span id=meta></span>
</div>
<div class=bar>
 <label>view<select id=mode><option value=side selected>side-by-side</option><option value=overlay>overlay</option></select></label>
 <span class=lay id=layers></span>
 <button onclick=resetView()>reset view</button>
</div>
<div class=wrap>
 <div class=cams id=cams></div>
 <div class=bevwrap id=bevwrap>
   <div class=canvbox><div class=tag id=tagA></div><canvas id=cvA></canvas><div class=hint>wheel=zoom · drag=pan</div></div>
   <div class=canvbox id=boxB><div class=tag id=tagB></div><canvas id=cvB></canvas></div>
 </div>
</div>
<script>
const CAMS=[['CAM_FRONT_LEFT','F-L'],['CAM_FRONT','FRONT'],['CAM_FRONT_RIGHT','F-R'],
 ['CAM_BACK_LEFT','B-L'],['CAM_BACK','BACK'],['CAM_BACK_RIGHT','B-R']];
const LAYERS=[['lidar','LiDAR',1],['gt_boxes','GT box',1],['det','pred det',1],
 ['map','pred map',0],['motion','pred motion',0],['gt_traj','GT traj',1],['plan','plan traj',1]];
const C={bg:'#111',lidar:'rgba(150,162,175,0.40)',ego:'#34d399',gt:'rgba(52,211,153,0.55)',
 a_det:'#22d3ee',b_det:'#f472b6',gt_traj:'#34d399',a_traj:'#f59e0b',b_traj:'#ef4444',
 a_mot:'#a78bfa',b_mot:'#fb923c',coll:'#ef4444',map:{0:'rgba(96,165,250,.85)',1:'rgba(245,158,11,.85)',2:'rgba(16,185,129,.85)'}};
let scenes=[],idx=0,G=null,P=null;
let view={range:40,zoom:1,panX:0,panY:0};
const el=id=>document.getElementById(id),q=id=>el(id).value;
el('layers').innerHTML=LAYERS.map(([k,n,d])=>`<label><input type=checkbox id=L_${k} ${d?'checked':''}>${n}</label>`).join('');
LAYERS.forEach(([k])=>el('L_'+k).onchange=onLayer);
function camGrid(){el('cams').innerHTML=CAMS.map(([c,n])=>`<div class=camcell><span>${n}</span><img id=im_${c}></div>`).join('')}
camGrid();

function onLayer(){ if((el('L_map').checked||el('L_motion').checked)&&!P){loadPercep().then(draw)} else draw() }
async function loadScenes(){
 const u=`/api/scenes?epoch=${q('epoch')}&task=${q('task')}&rank=${q('rank')}&density=${q('density')}&maneuver=${q('maneuver')}&mapcx=${q('mapcx')}`;
 scenes=await (await fetch(u)).json();idx=0;await showScene();
}
async function showScene(){
 if(!scenes.length){el('meta').textContent='(no scenes match)';G=null;clearAll();return}
 const s=scenes[idx];
 CAMS.forEach(([c])=>el('im_'+c).src=`/img?token=${s.token}&cam=${c}`);
 G=await (await fetch(`/api/geom?token=${s.token}&epoch=${q('epoch')}&task=${q('task')}`)).json();
 P=null; resetView();
 if(el('L_map').checked||el('L_motion').checked) await loadPercep();
 el('meta').textContent=`${idx+1}/${scenes.length}  ${s.token.slice(0,10)}  ${G.meta.cmd}  near15=${G.meta.n_near15}  L2 full=${G.meta.L2a} / ${G.meta.vb}=${G.meta.L2b}  coll ${G.colA.length}/${G.colB.length}`;
 draw();
}
async function loadPercep(){ const s=scenes[idx];
 P=await (await fetch(`/api/percep?token=${s.token}&epoch=${q('epoch')}&task=${q('task')}`)).json(); }
function step(d){if(scenes.length){idx=(idx+d+scenes.length)%scenes.length;showScene()}}
['epoch','task','rank','density','maneuver','mapcx'].forEach(id=>el(id).onchange=loadScenes);
el('mode').onchange=()=>{layout();draw()};

function layout(){ el('boxB').style.display = q('mode')==='side' ? '' : 'none'; sizeCanvases(); }
function sizeCanvases(){ ['cvA','cvB'].forEach(id=>{const c=el(id);c.width=c.clientWidth;c.height=c.clientHeight}); }
function resetView(){view={range:40,zoom:1,panX:0,panY:0};}
function W2S(x,y,W,H){const s=Math.min(W,H)/(2*view.range)*view.zoom;
 return [W/2-(y-view.panY)*s, H/2-(x-view.panX)*s];}

function clearAll(){['cvA','cvB'].forEach(id=>{const c=el(id),x=c.getContext('2d');x.fillStyle=C.bg;x.fillRect(0,0,c.width,c.height)})}
function draw(){ if(!G){clearAll();return} layout();
 if(q('mode')==='side'){ drawCanvas('cvA','a'); drawCanvas('cvB','b'); el('tagA').textContent=`full  coll=${G.colA.length}`; el('tagB').textContent=`${G.meta.vb}  coll=${G.colB.length}`; }
 else { drawCanvas('cvA','both'); el('tagA').textContent=`overlay — full=cyan/orange, ${G.meta.vb}=pink/red`; }
}
function box(x,b,col,W,H){const[cx,cy,yaw,dx,dy]=b;const c=Math.cos(yaw),s=Math.sin(yaw);
 const cor=[[dx/2,dy/2],[dx/2,-dy/2],[-dx/2,-dy/2],[-dx/2,dy/2]].map(([ox,oy])=>{
  const wx=cx+ox*c-oy*s, wy=cy+ox*s+oy*c; return W2S(wx,wy,W,H);});
 x.beginPath();x.moveTo(cor[0][0],cor[0][1]);for(let i=1;i<4;i++)x.lineTo(cor[i][0],cor[i][1]);x.closePath();
 x.strokeStyle=col;x.lineWidth=1.3;x.stroke();
 // heading tick (front face midpoint)
 const fm=W2S(cx+ (dx/2)*c, cy+(dx/2)*s, W,H); x.beginPath();x.arc(fm[0],fm[1],1.6,0,7);x.fillStyle=col;x.fill();}
function poly(x,pts,col,W,H,lw){x.beginPath();pts.forEach((p,i)=>{const[sx,sy]=W2S(p[0],p[1],W,H);i?x.lineTo(sx,sy):x.moveTo(sx,sy)});x.strokeStyle=col;x.lineWidth=lw||1.4;x.stroke();}
function trajLine(x,pts,col,W,H,coll){poly(x,pts,col,W,H,2.2);
 pts.forEach((p,i)=>{const[sx,sy]=W2S(p[0],p[1],W,H);x.beginPath();x.arc(sx,sy,1.8,0,7);x.fillStyle=col;x.fill();});
 (coll||[]).forEach(t=>{const[sx,sy]=W2S(pts[t+1][0],pts[t+1][1],W,H);
   x.strokeStyle=C.coll;x.lineWidth=2;x.beginPath();x.moveTo(sx-5,sy-5);x.lineTo(sx+5,sy+5);x.moveTo(sx+5,sy-5);x.lineTo(sx-5,sy+5);x.stroke();});}
function drawCanvas(id,which){const c=el(id),x=c.getContext('2d'),W=c.width,H=c.height;
 x.fillStyle=C.bg;x.fillRect(0,0,W,H);
 const on=k=>el('L_'+k).checked, A=which!=='b', B=which!=='a';
 if(on('lidar')&&G.lidar){x.fillStyle=C.lidar;for(let i=0;i<G.lidar.length;i+=2){const[sx,sy]=W2S(G.lidar[i],G.lidar[i+1],W,H);x.fillRect(sx,sy,1.1,1.1);}}
 if(on('map')&&P){ if(A)P.a_map.forEach(m=>poly(x,m.pts,C.map[m.label]||'#888',W,H,1.4)); if(B&&which==='b')P.b_map.forEach(m=>poly(x,m.pts,C.map[m.label]||'#888',W,H,1.4)); if(which==='both')P.b_map.forEach(m=>poly(x,m.pts,'rgba(244,114,182,.6)',W,H,1.2)); }
 if(on('gt_boxes'))G.gt_boxes.forEach(b=>box(x,b,C.gt,W,H));
 if(on('motion')&&P){ if(A)P.a_motion.forEach(m=>poly(x,m,C.a_mot,W,H,1)); if(which==='b')P.b_motion.forEach(m=>poly(x,m,C.b_mot,W,H,1)); if(which==='both')P.b_motion.forEach(m=>poly(x,m,C.b_mot,W,H,1)); }
 if(on('det')){ if(A)G.a_det.forEach(b=>box(x,b,C.a_det,W,H)); if(which==='b')G.b_det.forEach(b=>box(x,b,C.b_det,W,H)); if(which==='both')G.b_det.forEach(b=>box(x,b,C.b_det,W,H)); }
 if(on('gt_traj'))trajLine(x,G.gt_traj,C.gt_traj,W,H,[]);
 if(on('plan')){ if(A)trajLine(x,G.a_traj,C.a_traj,W,H,G.colA); if(which==='b')trajLine(x,G.b_traj,C.b_traj,W,H,G.colB); if(which==='both')trajLine(x,G.b_traj,C.b_traj,W,H,G.colB); }
 // ego
 const[ex,ey]=W2S(0,0,W,H);x.fillStyle=C.ego;x.beginPath();x.moveTo(ex,ey-7);x.lineTo(ex-5,ey+5);x.lineTo(ex+5,ey+5);x.closePath();x.fill();
}
// interaction
['cvA','cvB'].forEach(id=>{const c=el(id);
 c.addEventListener('wheel',e=>{e.preventDefault();view.zoom*=e.deltaY<0?1.12:0.89;draw();},{passive:false});
 let drag=false,lx,ly;
 c.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;c.style.cursor='grabbing'});
 window.addEventListener('mouseup',()=>{drag=false;c.style.cursor='grab'});
 c.addEventListener('mousemove',e=>{if(!drag)return;const s=Math.min(c.width,c.height)/(2*view.range)*view.zoom;
  view.panX+= (e.clientY-ly)/s; view.panY+= (e.clientX-lx)/s; lx=e.clientX;ly=e.clientY;draw();});
});
window.addEventListener('resize',()=>{sizeCanvases();draw()});
loadScenes();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _s(self, code, ct, body):
        self.send_response(code); self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); p = parse_qs(u.query)
        g = lambda k, d='': p.get(k, [d])[0]
        try:
            if u.path == '/':
                self._s(200, 'text/html; charset=utf-8', PAGE.encode())
            elif u.path == '/api/scenes':
                self._s(200, 'application/json', json.dumps(scene_rows(
                    int(g('epoch', '9')), g('task', 'det'), g('rank', 'showcase'),
                    g('density', 'all'), g('maneuver', 'all'), g('mapcx', 'all'))).encode())
            elif u.path == '/api/geom':
                self._s(200, 'application/json', json.dumps(geom.scene_geom(g('token'), int(g('epoch')), g('task'))).encode())
            elif u.path == '/api/percep':
                self._s(200, 'application/json', json.dumps(geom.scene_percep(g('token'), int(g('epoch')), g('task'))).encode())
            elif u.path == '/img':
                self._s(200, 'image/jpeg', img_bytes(g('token'), g('cam')))
            else:
                self._s(404, 'text/plain', b'nf')
        except Exception as e:
            import traceback; traceback.print_exc(); self._s(500, 'text/plain', str(e).encode())


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    print('pre-loading val infos ...', flush=True); lib_io.load_val_infos()
    print(f'serving on http://localhost:{port}  (Ctrl-C to stop)', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()
