// 交互教程的公共工具：加载数据、色图、热力图叠加、动画播放器
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

async function loadJSON(url) { const r = await fetch(url); return r.json(); }
async function loadBin(url) { const r = await fetch(url); return new Uint8Array(await r.arrayBuffer()); }
function loadImage(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
}

// 近似 matplotlib 的 magma / viridis 色图（几个控制点线性插值）
const CMAPS = {
  magma: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,253,191]],
  viridis: [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],[39,173,129],[92,200,99],[170,220,50],[253,231,37]],
  jet: [[0,0,143],[0,0,255],[0,127,255],[0,255,255],[127,255,127],[255,255,0],[255,127,0],[255,0,0],[127,0,0]],
};
function cmap(v, name = "magma") {
  const s = CMAPS[name]; v = Math.max(0, Math.min(1, v)) * (s.length - 1);
  const i = Math.min(Math.floor(v), s.length - 2), t = v - i;
  return s[i].map((c, k) => Math.round(c + (s[i + 1][k] - c) * t));
}

// 在 ctx 上把 h×w 的数值网格画成热力图（0~1），alpha 为透明度；smooth=true 时双线性平滑
// vmax 不传时按本组数据的最大值归一化；传入固定值可以让多张图共用同一个色标
function drawHeat(ctx, vals, h, w, X, Y, W, H, alpha = 0.6, name = "jet", smooth = true, vmax = null) {
  const off = document.createElement("canvas"); off.width = w; off.height = h;
  const oc = off.getContext("2d"), img = oc.createImageData(w, h);
  let mx = 1e-12; for (const v of vals) mx = Math.max(mx, v);
  if (vmax) mx = vmax;
  for (let i = 0; i < h * w; i++) {
    const c = cmap(vals[i] / mx, name);
    img.data.set([c[0], c[1], c[2], 255], i * 4);
  }
  oc.putImageData(img, 0, 0);
  ctx.save(); ctx.globalAlpha = alpha; ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(off, X, Y, W, H); ctx.restore();
}

// 画 h×w 的网格线
function drawGrid(ctx, n, X, Y, W, H, color = "rgba(255,255,255,.5)", lw = 0.5) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    ctx.moveTo(X + i * W / n, Y); ctx.lineTo(X + i * W / n, Y + H);
    ctx.moveTo(X, Y + i * H / n); ctx.lineTo(X + W, Y + i * H / n);
  }
  ctx.stroke(); ctx.restore();
}

// 简单的动画播放器：在 [0, n-1] 之间逐帧前进，绑定滑块和播放按钮
class Player {
  constructor({ n, onFrame, slider, playBtn, label, interval = 600, loop = true }) {
    Object.assign(this, { n, onFrame, slider, playBtn, label, interval, loop });
    this.i = 0; this.timer = null;
    if (slider) { slider.min = 0; slider.max = n - 1; slider.step = 1; slider.value = 0;
      slider.addEventListener("input", () => { this.stop(); this.go(+slider.value); }); }
    if (playBtn) playBtn.addEventListener("click", () => this.timer ? this.stop() : this.play());
  }
  go(i) {
    this.i = Math.max(0, Math.min(this.n - 1, i));
    if (this.slider) this.slider.value = this.i;
    this.onFrame(this.i);
  }
  play() {
    if (this.i >= this.n - 1) this.go(0);
    if (this.playBtn) { this.playBtn.textContent = "⏸ 暂停"; this.playBtn.classList.add("on"); }
    this.timer = setInterval(() => {
      if (this.i >= this.n - 1) { if (this.loop) this.go(0); else return this.stop(); }
      else this.go(this.i + 1);
    }, this.interval);
  }
  stop() {
    clearInterval(this.timer); this.timer = null;
    if (this.playBtn) { this.playBtn.textContent = "▶ 播放"; this.playBtn.classList.remove("on"); }
  }
}

// 平滑过渡动画：在 duration 毫秒内把 t 从 0 推到 1，每帧调用 draw(t)
function tween(duration, draw, done) {
  const t0 = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - t0) / duration);
    draw(t); if (t < 1) requestAnimationFrame(frame); else if (done) done();
  }
  requestAnimationFrame(frame);
}
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

// 横向柱状图（候选词概率等）
function renderBars(el, items, color = "var(--accent2)", fmt = v => (v * 100).toFixed(1) + "%") {
  const mx = Math.max(...items.map(d => d[1]), 1e-9);
  el.innerHTML = items.map(([w, v]) =>
    `<div class="bar"><span class="w" title="${esc(w)}">${esc(w)}</span><i style="width:${(v / mx * 220).toFixed(0)}px;background:${color}"></i><span>${fmt(v)}</span></div>`
  ).join("");
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])).replace(/\n/g, "↵"); }

// 简单折线图：series = [{name, values, color}]，highlight 为当前横坐标
function lineChart(canvas, series, { highlight = null, logY = false, xlabel = "", ylabel = "" } = {}) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, P = { l: 46, r: 10, t: 10, b: 30 };
  ctx.clearRect(0, 0, W, H);
  const all = series.flatMap(s => s.values).filter(v => !logY || v > 0);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (logY) { lo = Math.log10(lo); hi = Math.log10(hi); }
  if (hi === lo) hi = lo + 1;
  const n = series[0].values.length;
  const x = i => P.l + i * (W - P.l - P.r) / Math.max(1, n - 1);
  const y = v => { const u = logY ? Math.log10(v) : v; return H - P.b - (u - lo) / (hi - lo) * (H - P.t - P.b); };
  ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(P.l, P.t); ctx.lineTo(P.l, H - P.b); ctx.lineTo(W - P.r, H - P.b); ctx.stroke();
  ctx.fillStyle = "#888"; ctx.font = "11px sans-serif";
  ctx.fillText(logY ? (10 ** hi).toPrecision(2) : hi.toPrecision(3), 2, P.t + 8);
  ctx.fillText(logY ? (10 ** lo).toPrecision(2) : lo.toPrecision(3), 2, H - P.b);
  ctx.fillText(xlabel, W / 2 - 20, H - 6);
  if (highlight !== null) { ctx.fillStyle = "rgba(208,83,43,.12)"; ctx.fillRect(x(highlight) - 6, P.t, 12, H - P.t - P.b); }
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    s.values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.stroke();
    s.values.forEach((v, i) => { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(x(i), y(v), i === highlight ? 4.5 : 2.5, 0, 7); ctx.fill(); });
  });
  ctx.font = "12px sans-serif";
  series.forEach((s, k) => { ctx.fillStyle = s.color; ctx.fillText("● " + s.name, P.l + 8 + k * 110, P.t + 12); });
}
