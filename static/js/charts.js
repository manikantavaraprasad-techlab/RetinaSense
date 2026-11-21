/* charts.js — dependency-free animated SVG/HTML charts used by RetinaSense (works offline). */
const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id), sleep = ms => new Promise(r => setTimeout(r, ms));
const CC = { Normal: '#34d399', Glaucoma: '#fbbf24', Diabetes: '#c084fc', 'High blood pressure': '#fb7185' };
const BMC = ['#22d3ee', '#7c5cff', '#fbbf24', '#fb7185', '#34d399', '#60a5fa'];
const ART = '#fb7185', VEIN = '#60a5fa';
const fx = (n, d = 2) => (+n).toFixed(d);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const svg = (w, h, inner, cls = '') => `<svg viewBox="0 0 ${w} ${h}" class="${cls}" role="img">${inner}</svg>`;
const tip = t => `data-tip="${String(t).replace(/"/g, '&quot;')}"`;

/* ---------- reveal / animation trigger ---------- */
const IO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('go'); e.target.querySelectorAll('[data-n]').forEach(count); IO.unobserve(e.target); } }), { threshold: .15 });
function watchCharts(root = document) { root.querySelectorAll('.ch').forEach(c => IO.observe(c)); }
function replay(root) { root.querySelectorAll('.ch').forEach(c => { c.classList.remove('go'); IO.unobserve(c); void c.offsetWidth; IO.observe(c); }); }
function count(e) {
  const n = +e.dataset.n, d = +(e.dataset.d || 0), s = e.dataset.s || '', t0 = performance.now();
  (function f(t) { const k = Math.min(1, (t - t0) / 1500), v = n * (1 - Math.pow(1 - k, 3)); e.textContent = v.toFixed(d) + s; if (k < 1) requestAnimationFrame(f); })(t0);
}
/* tooltip */
(() => {
  const t = document.createElement('div'); t.id = 'tt'; document.addEventListener('DOMContentLoaded', () => document.body.append(t));
  document.addEventListener('mousemove', e => { const el = e.target.closest && e.target.closest('[data-tip]'); if (!el) { t.style.opacity = 0; return; }
    t.innerHTML = el.dataset.tip; t.style.opacity = 1; t.style.left = Math.min(e.clientX + 14, innerWidth - 240) + 'px'; t.style.top = (e.clientY + 16) + 'px'; });
})();

/* ---------- helpers ---------- */
function smooth(pts) {                       // Catmull-Rom -> cubic bezier path
  if (pts.length < 2) return '';
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    d += `C${p1[0] + (p2[0] - p0[0]) / 6},${p1[1] + (p2[1] - p0[1]) / 6} ${p2[0] - (p3[0] - p1[0]) / 6},${p2[1] - (p3[1] - p1[1]) / 6} ${p2[0]},${p2[1]}`;
  }
  return d;
}
const legend = items => `<div class="leg">${items.map(s => `<span><i style="background:${s.c}"></i>${s.n}</span>`).join('')}</div>`;

/* ---------- 1. donut ---------- */
function donut(items, centerTop, centerSub) {         // items [{n,v,c}] v in %
  const C = 2 * Math.PI * 70; let off = 0, segs = '';
  items.forEach((it, i) => { const len = it.v / 100 * C;
    segs += `<circle class="seg" cx="100" cy="100" r="70" stroke="${it.c}" style="--d:${Math.max(len - 2, 0)} ${C};--o:${-off};transition-delay:${i * .15}s" ${tip(`${it.n}: ${it.v}%`)}/>`; off += len; });
  return `<div class="ch donut">${svg(200, 200, `<circle cx="100" cy="100" r="70" class="trk-c"/><g transform="rotate(-90 100 100)">${segs}</g>
   <text x="100" y="98" text-anchor="middle" class="dn-t">${centerTop}</text><text x="100" y="118" text-anchor="middle" class="dn-s">${centerSub}</text>`)}</div>`;
}
/* ---------- 2. semicircle gauge ---------- */
function gauge(v, label, color, sub = '') {
  const L = Math.PI * 62;
  return `<div class="ch gauge2">${svg(160, 100, `<path d="M18,90 A62,62 0 0 1 142,90" class="gtrk"/><path class="gfill" d="M18,90 A62,62 0 0 1 142,90" stroke="${color}" style="--L:${L};--v:${L * (1 - v / 100)}"/>
   <text x="80" y="82" text-anchor="middle" class="g-n" data-n="${v}" data-d="0" data-s="%">0%</text>`)}<b>${label}</b><small>${sub}</small></div>`;
}
/* ---------- 3. radar ---------- */
function radar(axes, series, band, size = 380) {          // vals are radii 0..1 ; band {lo:[],hi:[]} radii
  const PADX = 90, VW = size + 2 * PADX, cx = VW / 2, cy = size / 2, R = size / 2 - 62, n = axes.length, ang = i => -Math.PI / 2 + i * 2 * Math.PI / n;
  const P = (i, r) => [cx + Math.cos(ang(i)) * R * r, cy + Math.sin(ang(i)) * R * r], poly = a => a.map((r, i) => P(i, r).join(',')).join(' ');
  let g = [.25, .5, .75, 1].map(r => `<polygon points="${poly(axes.map(() => r))}" class="rgrid"/>`).join('');
  g += axes.map((a, i) => { const [x, y] = P(i, 1.0), [lx, ly] = P(i, 1.2); return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" class="rgrid"/><text x="${lx}" y="${ly}" class="rlab" text-anchor="${lx < cx - 6 ? 'end' : lx > cx + 6 ? 'start' : 'middle'}">${a}</text>`; }).join('');
  if (band) g += `<polygon points="${poly(band.hi)}" class="rband"/><polygon points="${poly(band.lo)}" class="rband in"/>`;
  g += series.map((s, k) => `<polygon class="rpoly" points="${poly(s.vals)}" fill="${s.c}2e" stroke="${s.c}" style="transition-delay:${k * .25}s"/>` +
    s.vals.map((r, i) => { const [x, y] = P(i, r); return `<circle class="rdot" cx="${x}" cy="${y}" r="4.5" fill="${s.c}" ${tip(`${s.n}: ${axes[i]} → ${s.raw ? s.raw[i] : fx(r)}`)} style="transition-delay:${.5 + i * .06}s"/>`; }).join('')).join('');
  return `<div class="ch radar">${svg(VW, size, g)}</div>`;
}
/* ---------- 4. bullet bars (value vs reference band) ---------- */
function bullets(findings) {
  return `<div class="ch bul">${findings.map((f, i) => { const mx = Math.max(f.hi * 1.5, f.value * 1.15), ok = f.status === 'normal';
    return `<div class="bm"><div>${f.label}<small>reference ${f.lo}–${f.hi} ${f.unit}</small></div>
     <div class="trk"><div class="band" style="left:${f.lo / mx * 100}%;width:${(f.hi - f.lo) / mx * 100}%"></div><div class="fill ${ok ? 'within' : 'OUT'}" style="--w:${f.value / mx * 100}%;transition-delay:${i * .12}s"></div></div>
     <div class="v ${ok ? 'ok' : 'bad'}">${fx(f.value, 3)} ${ok ? '✓' : f.status === 'low' ? '▼ low' : '▲ high'}</div></div>`; }).join('')}</div>`;
}
/* ---------- 5. stacked evidence ---------- */
function evidenceBars(ev, keys, labels) {
  const rows = Object.keys(ev), tot = rows.map(c => ev[c].reduce((a, e) => a + e.contribution, 0)), mx = Math.max(...tot, .5);
  return `<div class="ch evid">${rows.map((c, r) => `<div class="er"><span style="color:${CC[c]}">${c}</span><div class="etr">${ev[c].length ? ev[c].map(e => `<i style="--w:${e.contribution / mx * 100}%;background:${BMC[keys.indexOf(e.key)]};transition-delay:${r * .15}s" ${tip(`${e.label} (${e.direction}) +${fx(e.contribution)}`)}></i>`).join('') : '<em>no supporting evidence</em>'}</div><b>${fx(tot[r])}</b></div>`).join('')}
   ${legend(keys.map((k, i) => ({ n: labels[k], c: BMC[i] })))}</div>`;
}
/* ---------- 6. area / line chart ---------- */
function areaChart(cats, series, o = {}) {
  const W = 560, H = 270, L = 44, Rr = 14, T = 16, B = 38, iw = W - L - Rr, ih = H - T - B, mx = o.max || Math.max(...series.flatMap(s => s.vals)) * 1.15 || 1;
  const X = i => L + i / (cats.length - 1) * iw, Y = v => T + ih - v / mx * ih;
  let g = [0, .25, .5, .75, 1].map(t => `<line x1="${L}" x2="${W - Rr}" y1="${T + ih * (1 - t)}" y2="${T + ih * (1 - t)}" class="gl"/><text x="${L - 6}" y="${T + ih * (1 - t) + 4}" class="ax" text-anchor="end">${fx(mx * t, mx < 5 ? 1 : 0)}</text>`).join('');
  g += cats.map((c, i) => i % (o.every || 1) === 0 ? `<text x="${X(i)}" y="${H - 20}" class="ax" text-anchor="middle">${c}</text>` : '').join('');
  g += `<text x="${W / 2}" y="${H - 4}" class="axt" text-anchor="middle">${o.xl || ''}</text><text transform="rotate(-90)" x="${-H / 2}" y="11" class="axt" text-anchor="middle">${o.yl || ''}</text>`;
  g += series.map((s, k) => { const pts = s.vals.map((v, i) => [X(i), Y(v)]), d = smooth(pts);
    return `<defs><linearGradient id="ag${o.id || 'x'}${k}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${s.c}" stop-opacity=".45"/><stop offset="1" stop-color="${s.c}" stop-opacity="0"/></linearGradient></defs>
     ${o.area === false ? '' : `<path class="aarea" d="${d}L${X(cats.length - 1)},${T + ih}L${X(0)},${T + ih}Z" fill="url(#ag${o.id || 'x'}${k})" style="transition-delay:${k * .3}s"/>`}
     <path class="aline" pathLength="1" d="${d}" stroke="${s.c}" style="transition-delay:${k * .3}s"/>` +
     pts.map(([x, y], i) => `<circle class="adot" cx="${x}" cy="${y}" r="4" fill="${s.c}" ${tip(`${s.n} · ${cats[i]}: ${s.vals[i]}`)} style="transition-delay:${1 + i * .04}s"/>`).join(''); }).join('');
  return `<div class="ch">${series.length > 1 ? legend(series.map(s => ({ n: s.n, c: s.c }))) : ''}${svg(W, H, g)}</div>`;
}
/* ---------- 7. polar rose ---------- */
function rose(labels, vals, color = '#22d3ee') {
  const n = vals.length * 2, S = 360, c = S / 2, R = 128, mx = Math.max(...vals, 1), wedge = 2 * Math.PI / n; let g = '';
  [.33, .66, 1].forEach(r => g += `<circle cx="${c}" cy="${c}" r="${R * r}" class="rgrid"/>`);
  for (let i = 0; i < n; i++) { const k = i % vals.length, v = vals[k], r = 18 + (R - 18) * v / mx, a0 = -Math.PI / 2 + i * wedge + .02, a1 = a0 + wedge - .04;
    const p = (a, rr) => `${c + Math.cos(a) * rr},${c + Math.sin(a) * rr}`;
    g += `<path class="rw" d="M${p(a0, 18)}L${p(a0, r)}A${r},${r} 0 0 1 ${p(a1, r)}L${p(a1, 18)}A18,18 0 0 0 ${p(a0, 18)}Z" fill="${color}" fill-opacity="${.35 + .6 * v / mx}" stroke="${color}" style="transition-delay:${i * .04}s;transform-origin:${c}px ${c}px" ${tip(`${labels[k]} orientation: ${v}% of vessel length`)}/>`; }
  g += `<text x="${c}" y="${c - R - 8}" class="ax" text-anchor="middle">90° (vertical)</text><text x="${c + R + 6}" y="${c + 4}" class="ax">0°</text>`;
  return `<div class="ch">${svg(S, S, g)}</div>`;
}
/* ---------- 8. heat-map grid ---------- */
function heatmap(grid) {
  const flat = grid.flat().filter(v => v != null), mx = Math.max(...flat, 1), n = grid.length, cs = 62, S = n * cs + 6;
  const col = t => { const a = [[16, 24, 58], [34, 211, 238], [251, 191, 36]], k = t * 2, i = Math.min(1, Math.floor(k)), f = k - i; return `rgb(${a[i].map((x, j) => Math.round(x + (a[i + 1][j] - x) * f)).join(',')})`; };
  let g = '';
  grid.forEach((row, i) => row.forEach((v, j) => { g += v == null ? `<rect x="${3 + j * cs}" y="${3 + i * cs}" width="${cs - 4}" height="${cs - 4}" rx="9" class="hmn"/>` :
    `<g class="hc" style="transition-delay:${(i + j) * .06}s;transform-origin:${3 + j * cs + cs / 2}px ${3 + i * cs + cs / 2}px" ${tip(`Density ${v}% of retina area`)}><rect x="${3 + j * cs}" y="${3 + i * cs}" width="${cs - 4}" height="${cs - 4}" rx="9" fill="${col(v / mx)}"/><text x="${3 + j * cs + cs / 2 - 2}" y="${3 + i * cs + cs / 2 + 4}" text-anchor="middle" class="hv" fill="${v / mx > .55 ? '#101428' : '#fff'}">${v}</text></g>`; }));
  return `<div class="ch hm">${svg(S, S, g)}<div class="scale"><span>sparse</span><i></i><span>dense</span></div></div>`;
}
/* ---------- 9. histogram ---------- */
function hist(labels, vals, colorFn, o = {}) {
  const W = 560, H = 250, L = 40, B = 36, T = 16, iw = W - L - 10, ih = H - T - B, mx = Math.max(...vals, 1) * 1.1, bw = iw / vals.length;
  let g = [0, .5, 1].map(t => `<line x1="${L}" x2="${W - 10}" y1="${T + ih * (1 - t)}" y2="${T + ih * (1 - t)}" class="gl"/><text x="${L - 6}" y="${T + ih * (1 - t) + 4}" class="ax" text-anchor="end">${fx(mx * t, 0)}</text>`).join('');
  vals.forEach((v, i) => g += `<rect class="hb" x="${L + i * bw + 4}" y="${T + ih - v / mx * ih}" width="${bw - 8}" height="${v / mx * ih}" rx="5" fill="${colorFn(i)}" style="transition-delay:${i * .06}s" ${tip(`${labels[i]}: ${v}%`)}/><text x="${L + i * bw + bw / 2}" y="${H - 20}" class="ax" text-anchor="middle">${labels[i]}</text>`);
  g += `<text x="${W / 2}" y="${H - 4}" class="axt" text-anchor="middle">${o.xl || ''}</text>`;
  return `<div class="ch">${svg(W, H, g)}</div>`;
}
/* ---------- 10. funnel ---------- */
function funnel(items) {
  const mx = Math.max(...items.map(i => i.value));
  return `<div class="ch fun">${items.map((it, i) => `<div class="fr"><span>${it.label}</span><div class="ftr"><i style="--w:${Math.max(it.value / mx * 100, 6)}%;transition-delay:${i * .2}s;background:linear-gradient(90deg,#7c5cff,#22d3ee)"></i></div><b>${it.value.toLocaleString()} px</b></div>`).join('')}</div>`;
}
/* ---------- 11. scatter ---------- */
function scatter(pts, o) {                                   // pts [{x,y,c,tip,id}]
  const W = o.w || 520, H = o.h || 440, L = 52, B = 46, T = 14, R = 14, iw = W - L - R, ih = H - T - B, [x0, x1] = o.xr, [y0, y1] = o.yr;
  const X = v => L + (v - x0) / (x1 - x0) * iw, Y = v => T + ih - (v - y0) / (y1 - y0) * ih; let g = '';
  const tk = (a, b, n) => Array.from({ length: n + 1 }, (_, i) => a + (b - a) * i / n);
  tk(x0, x1, o.n || 5).forEach(v => g += `<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${T + ih}" class="gl"/><text x="${X(v)}" y="${H - 28}" class="ax" text-anchor="middle">${fx(v, o.d ?? 2)}</text>`);
  tk(y0, y1, o.n || 5).forEach(v => g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="gl"/><text x="${L - 6}" y="${Y(v) + 4}" class="ax" text-anchor="end">${fx(v, o.d ?? 2)}</text>`);
  if (o.diag) { const a = Math.max(x0, y0), b = Math.min(x1, y1); g += `<line x1="${X(a)}" y1="${Y(a)}" x2="${X(b)}" y2="${Y(b)}" class="diag"/><text x="${X(b) - 6}" y="${Y(b) + 14}" class="ax" text-anchor="end">artery = vein</text>`; }
  g += `<text x="${L + iw / 2}" y="${H - 6}" class="axt" text-anchor="middle">${o.xl}</text><text transform="rotate(-90)" x="${-(T + ih / 2)}" y="13" class="axt" text-anchor="middle">${o.yl}</text>`;
  pts.forEach((p, i) => g += `<circle class="sp" cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.r || 6}" fill="${p.c}" fill-opacity=".85" ${p.id != null ? `data-id="${p.id}"` : ''} ${tip(p.tip)} style="transition-delay:${Math.min(i * .035, 1.2)}s;transform-origin:${X(p.x)}px ${Y(p.y)}px"/>`);
  return `<div class="ch sc ${o.cls || ''}" ${o.attr || ''}>${svg(W, H, g)}</div>`;
}
/* ---------- 12. dumbbell ---------- */
function dumbbell(rows, o) {                                 // rows [{label,a,b,id}]
  const W = 560, rh = 23, L = 108, R = 24, T = 26, H = T + rows.length * rh + 34, iw = W - L - R, [x0, x1] = o.xr, X = v => L + (v - x0) / (x1 - x0) * iw; let g = '';
  for (let i = 0; i <= 5; i++) { const v = x0 + (x1 - x0) * i / 5; g += `<line x1="${X(v)}" x2="${X(v)}" y1="${T - 6}" y2="${H - 30}" class="gl"/><text x="${X(v)}" y="${H - 14}" class="ax" text-anchor="middle">${fx(v, 2)}</text>`; }
  rows.forEach((r, i) => { const y = T + i * rh + rh / 2, xa = X(r.a), xb = X(r.b), lo = Math.min(xa, xb);
    g += `<text x="${L - 8}" y="${y + 4}" class="ax lbl" text-anchor="end">${r.label}</text><rect class="dbl" x="${lo}" y="${y - 2}" width="${Math.abs(xb - xa)}" height="4" rx="2" fill="${o.lc || '#ffffff40'}" style="transition-delay:${i * .05}s"/>
     <circle class="sp" cx="${xa}" cy="${y}" r="6" fill="${o.ca}" ${tip(`${r.label}<br>${o.na}: ${fx(r.a, 3)}`)} style="transition-delay:${i * .05}s;transform-origin:${xa}px ${y}px"/><circle class="sp" cx="${xb}" cy="${y}" r="6" fill="${o.cb}" ${tip(`${r.label}<br>${o.nb}: ${fx(r.b, 3)}<br>gap ${fx(r.a - r.b, 3)}`)} style="transition-delay:${i * .05 + .1}s;transform-origin:${xb}px ${y}px"/>`; });
  return `<div class="ch">${legend([{ n: o.na, c: o.ca }, { n: o.nb, c: o.cb }])}${svg(W, H, g)}</div>`;
}
/* ---------- 13. diverging bars ---------- */
function divBars(rows, o = {}) {                             // rows [{label,v,tip}]
  const W = 560, rh = o.rh || 24, L = 112, T = 10, H = T + rows.length * rh + 30, mx = Math.max(...rows.map(r => Math.abs(r.v)), .01), c = L + (W - L - 20) / 2, half = (W - L - 20) / 2 - 8; let g = `<line x1="${c}" x2="${c}" y1="${T}" y2="${H - 28}" class="diag" style="stroke-dasharray:none;stroke:#ffffff55"/>`;
  rows.forEach((r, i) => { const w = Math.abs(r.v) / mx * half, y = T + i * rh + 3, pos = r.v >= 0;
    g += `<text x="${L - 8}" y="${y + 12}" class="ax lbl" text-anchor="end">${r.label}</text><rect class="dvb ${pos ? '' : 'neg'}" x="${pos ? c : c - w}" y="${y}" width="${w}" height="${rh - 8}" rx="4" fill="${pos ? (o.cp || '#34d399') : (o.cn || '#fb7185')}" style="transition-delay:${i * .04}s" ${tip(r.tip || `${r.label}: ${fx(r.v, 3)}`)}/>`; });
  g += `<text x="${c + half}" y="${H - 10}" class="ax" text-anchor="end">${o.pl || ''} →</text><text x="${c - half}" y="${H - 10}" class="ax">← ${o.nl || ''}</text>`;
  return `<div class="ch">${svg(W, H, g)}</div>`;
}
/* ---------- 14. butterfly ---------- */
function butterfly(rows, an = 'Artery', vn = 'Vein') {
  const mx = Math.max(...rows.flatMap(r => [r.artery, r.vein]), 1);
  return `<div class="ch bf"><div class="bfh"><span style="color:${ART}">◀ ${an} pixels</span><span style="color:${VEIN}">${vn} pixels ▶</span></div>${rows.map((r, i) => `<div class="bfr"><div class="bl"><i style="--w:${r.artery / mx * 100}%;background:${ART};transition-delay:${i * .12}s" ${tip(`${r.label} artery: ${r.artery.toLocaleString()} px`)}></i></div><span>${r.label}</span><div class="br"><i style="--w:${r.vein / mx * 100}%;background:${VEIN};transition-delay:${i * .12}s" ${tip(`${r.label} vein: ${r.vein.toLocaleString()} px`)}></i></div></div>`).join('')}</div>`;
}
/* ---------- 15. lollipop ---------- */
function lollipop(cats, series, mx = 1) {
  return `<div class="ch lol">${legend(series)}${cats.map((c, i) => `<div class="lr"><span>${c}</span><div class="ltr">${series.map((s, k) => `<div class="ls" style="--w:${s.v[i] / mx * 100}%;transition-delay:${(i * 3 + k) * .08}s"><u style="background:${s.c}"></u><b style="background:${s.c}" ${tip(`${s.n} · ${c}: ${s.v[i]}`)}></b><em>${s.v[i]}</em></div>`).join('')}</div></div>`).join('')}</div>`;
}
/* ---------- 16. ring progress ---------- */
function ring(v, label, color, big) {
  const C = 2 * Math.PI * 50;
  return `<div class="ch ringc">${svg(120, 120, `<circle cx="60" cy="60" r="50" class="trk-c"/><circle class="rp" cx="60" cy="60" r="50" stroke="${color}" transform="rotate(-90 60 60)" style="--C:${C};--v:${C * (1 - v / 100)}"/>`)}<b data-n="${v}" data-d="${big ? 0 : 1}" data-s="%">0</b><span>${label}</span></div>`;
}
