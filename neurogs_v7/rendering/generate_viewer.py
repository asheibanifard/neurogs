#!/usr/bin/env python3
"""
Generate a pre-rendered HTML orbit viewer from splat_clean_best.pt.

Renders 72 azimuth × 72 elevation = 5184 frames at 512×512 using
tiled CUDA splatting, encodes as JPEG base64, and embeds in a
self-contained HTML file with smooth drag-to-orbit interactivity.
"""
import os, sys, io, time, math, base64
import torch
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from rendering.rendering import Camera, orbit_camera_pose, compute_aspect_scales
from splat_cuda_wrapper import (
    CUDASplattingRenderer, build_covariances, apply_aspect_correction
)

device = torch.device("cuda")

# ── Load checkpoint ──────────────────────────────────────────────
ckpt_path = os.path.join(BASE_DIR, "checkpoints", "splat_clean_best.pt")
print(f"Loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
means = ckpt["means"].to(device)
log_scales = ckpt["log_scales"].to(device)
quaternions = ckpt["quaternions"].to(device)
log_amplitudes = ckpt["log_amplitudes"].to(device)
step = ckpt.get("step", "?")
K = means.shape[0]
print(f"  {K} Gaussians, step {step}")

# Build covariances & opacities
covariances = build_covariances(quaternions, log_scales)
amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))
opacities = amplitudes.clamp(0.0, 1.0)

# Aspect correction
aspect_scales = compute_aspect_scales((100, 647, 813)).to(device)
means_c, cov_c = apply_aspect_correction(means, covariances, aspect_scales)

# ── Camera setup ─────────────────────────────────────────────────
H, W = 512, 512
camera = Camera.from_fov(fov_x_deg=50.0, width=W, height=H, near=0.01, far=10.0)
renderer = CUDASplattingRenderer(near=0.01, far=10.0, radius_mult=3.0)

# Orbit grid — dense for smooth dragging
N_AZ = 72   # 5° steps, full 360°
N_EL = 72   # 5° steps, full 360°
azimuths = [i * (360.0 / N_AZ) for i in range(N_AZ)]
elevations = [round(-180 + i * (360.0 / N_EL), 1) for i in range(N_EL)]
radius = 3.5

# ── Render full-frame using tiles ────────────────────────────────
def render_full_frame(R, T, tile_size=64):
    """Render a full H×W frame by tiling to avoid OOM."""
    img = torch.zeros(H, W, device=device)
    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
            xs = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

            with torch.no_grad():
                rendered = renderer.render_at_pixels(
                    means_c, cov_c, opacities, opacities,
                    R.to(device), T.to(device),
                    camera.fx, camera.fy, camera.cx, camera.cy,
                    W, H, pixels,
                )
            img[y0:y1, x0:x1] = rendered.reshape(y1 - y0, x1 - x0)
    return img

def frame_to_jpeg_b64(frame_tensor, quality=85):
    """Convert a (H,W) float tensor [0,1] to JPEG base64 string."""
    from PIL import Image
    arr = (frame_tensor.cpu().clamp(0.0, 1.0) * 255).byte().numpy()
    pil = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')

# ── Pre-render all frames ────────────────────────────────────────
print(f"Pre-rendering {N_AZ}×{N_EL} = {N_AZ*N_EL} frames at {W}×{H}...")
t0 = time.time()
frames = []  # frames[az_idx * N_EL + el_idx]
count = 0
for ai, az in enumerate(azimuths):
    row = []
    for ei, el in enumerate(elevations):
        R, T = orbit_camera_pose(el, az, radius)
        img = render_full_frame(R, T, tile_size=128)
        b64 = frame_to_jpeg_b64(img, quality=85)
        row.append(b64)
        count += 1
        if count % 72 == 0 or count == N_AZ * N_EL:
            elapsed = time.time() - t0
            print(f"  {count}/{N_AZ*N_EL} frames ({elapsed:.1f}s)")
    frames.append(row)

elapsed = time.time() - t0
print(f"Rendering done: {elapsed:.1f}s total")

# ── Build HTML ───────────────────────────────────────────────────
print("Building HTML...")

# Flatten frames into JS array: F[az][el]
js_frames = "const F=[\n"
for ai in range(N_AZ):
    row_strs = [f'"{frames[ai][ei]}"' for ei in range(N_EL)]
    js_frames += "[" + ",".join(row_strs) + "],\n"
js_frames += "];\n"

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>NeuroGS Clean Viewer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111;color:#ccc;font-family:'Segoe UI',system-ui,sans-serif;user-select:none}}
.hdr{{padding:16px 24px;background:#1a1a2e;border-bottom:1px solid #333}}
.hdr h1{{font-size:18px;color:#fff;margin-bottom:4px}}
.hdr p{{font-size:12px;color:#888}}
.main{{display:flex;flex-direction:column;align-items:center;padding:20px}}
.vw{{position:relative;width:512px;height:512px;background:#000;border:1px solid #333;border-radius:8px;overflow:hidden;cursor:grab}}
.vw img{{width:100%;height:100%;object-fit:contain;image-rendering:auto;pointer-events:none}}
.ctl{{margin-top:16px;width:512px}}
.sr{{display:flex;align-items:center;gap:12px;margin:6px 0}}
.sr label{{width:80px;font-size:13px;color:#aaa;text-align:right}}
.sr input[type=range]{{flex:1;accent-color:#4a9eff}}
.sr .v{{width:60px;font-size:13px;color:#fff}}
.b{{padding:6px 16px;background:#4a9eff;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px}}
.b:hover{{background:#3a8eef}}
.b.on{{background:#e44}}
.pc{{display:flex;gap:8px;align-items:center;justify-content:center;margin-top:12px}}
</style>
</head><body>
<div class="hdr"><h1>NeuroGS Clean Gaussian Splat Viewer</h1>
<p>{K} Gaussians &middot; Step {step} &middot; {N_AZ*N_EL} frames ({N_AZ} az &times; {N_EL} el) &middot; {W}&times;{H} &middot; Drag to orbit &middot; <span id="fps" style="color:#4a9eff">-- FPS</span></p></div>
<div class="main">
<div class="vw" id="vw"><img id="d" /></div>
<div class="ctl">
<div class="sr">
<label>Azimuth:</label>
<input type="range" id="azR" min="0" max="{N_AZ-1}" value="0" />
<span class="v" id="azV">0&deg;</span>
</div>
<div class="sr">
<label>Elevation:</label>
<input type="range" id="elR" min="0" max="{N_EL-1}" value="{N_EL//2}" />
<span class="v" id="elV">{elevations[N_EL//2]}&deg;</span>
</div>
<div class="pc">
<button class="b" id="pb" onclick="tp()">&#9654; Play</button>
<button class="b" onclick="spd(0.5)">0.5&times;</button>
<button class="b" onclick="spd(1)">1&times;</button>
<button class="b" onclick="spd(2)">2&times;</button>
</div>
</div></div>
<script>
const NAZ={N_AZ},NEL={N_EL};
const ELS={elevations};
{js_frames}
let az=0,el={N_EL//2},playing=false,speed=1,animId=null;
let faz=0.0,fel={N_EL//2}.0; // fractional accumulator for smooth drag
let frameCount=0,fpsLast=performance.now();
const fpsEl=document.getElementById('fps');
function updateFps(){{frameCount++;let now=performance.now();if(now-fpsLast>=500){{let fps=Math.round(frameCount*1000/(now-fpsLast));fpsEl.textContent=fps+' FPS';frameCount=0;fpsLast=now;}}}}
const img=document.getElementById('d');
const azR=document.getElementById('azR'),elR=document.getElementById('elR');
const azV=document.getElementById('azV'),elV=document.getElementById('elV');
function show(){{img.src="data:image/jpeg;base64,"+F[az][el];azV.innerHTML=Math.round(az*360/NAZ)+"&deg;";elV.innerHTML=ELS[el]+"&deg;";azR.value=az;elR.value=el;updateFps();}}
show();
azR.addEventListener('input',e=>{{az=+e.target.value;show();}});
elR.addEventListener('input',e=>{{el=+e.target.value;show();}});
const vw=document.getElementById('vw');
let drag=false,sx,sy;
vw.addEventListener('pointerdown',e=>{{drag=true;sx=e.clientX;sy=e.clientY;vw.setPointerCapture(e.pointerId);vw.style.cursor='grabbing';}});
vw.addEventListener('pointerup',e=>{{drag=false;vw.style.cursor='grab';}});
vw.addEventListener('pointermove',e=>{{if(!drag)return;let dx=e.clientX-sx,dy=e.clientY-sy;sx=e.clientX;sy=e.clientY;faz+=dx*0.15;fel+=dy*0.15;az=((Math.round(faz)%NAZ)+NAZ)%NAZ;el=((Math.round(fel)%NEL)+NEL)%NEL;show();}});
function tp(){{playing=!playing;document.getElementById('pb').className=playing?'b on':'b';document.getElementById('pb').innerHTML=playing?'&#9724; Stop':'&#9654; Play';if(playing)anim();else if(animId)cancelAnimationFrame(animId);}}
function spd(s){{speed=s;}}
let last=0;function anim(t){{if(!playing)return;animId=requestAnimationFrame(anim);if(t-last>80/speed){{last=t;az=(az+1)%NAZ;show();}}}}
</script>
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), "viewer_clean.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Wrote {len(html)//1024}KB → {out_path}")
print("Done!")
