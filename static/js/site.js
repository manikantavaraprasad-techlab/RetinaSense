/* site.js — page logic: upload queue, one separate report per image, reference sections. */
const BMK = ['vessel_density', 'mean_vessel_width', 'vessel_tortuosity', 'avr', 'mean_artery_width', 'mean_vein_width'];
const LAB = { vessel_density: 'Vessel density', mean_vessel_width: 'Vessel width', vessel_tortuosity: 'Tortuosity', avr: 'Artery/vein ratio', mean_artery_width: 'Artery width', mean_vein_width: 'Vein width' };
const SHORT = { vessel_density: 'Density', mean_vessel_width: 'Width', vessel_tortuosity: 'Tortuosity', avr: 'AVR', mean_artery_width: 'Artery w.', mean_vein_width: 'Vein w.' };
const RNG = { vessel_density: [7, 13, '%'], mean_vessel_width: [2.5, 7, 'px'], vessel_tortuosity: [1, 1.15, ''], avr: [.67, .75, ''], mean_artery_width: [2, 6, 'px'], mean_vein_width: [3, 9, 'px'] };
const CK = { 'vessel_density_%': 'vessel_density', mean_vessel_width_px: 'mean_vessel_width', tortuosity_index: 'vessel_tortuosity', artery_vein_ratio: 'avr', mean_artery_width_px: 'mean_artery_width', mean_vein_width_px: 'mean_vein_width' };
const DISPLAY = { Normal: 'Normal', Glaucoma: 'Glaucoma pattern', Diabetes: 'Diabetes pattern', 'High blood pressure': 'High blood pressure pattern' };
const NOTE = { p0: 'The photo exactly as uploaded.', p1: 'Vessels are darkest in green light.', p2: 'CLAHE boosts local contrast.', p3: 'Retinex removes uneven lighting.', p4: 'Green + CLAHE + Retinex = network input.',
  g1: 'Brighter = more likely vessel.', g2: 'Probability above 0.5, cleaned.', g3: 'Detected vessels painted red.', c1: 'Raw threshold result.', c2: 'Red specks were noise and removed.', c3: 'Green pixels bridge broken vessels.', c4: 'One-pixel centre-line of every vessel.', c5: 'Red = junctions, cyan = end-points.', a1: 'Red = artery · blue = vein.' };
const GROUPS = [['pre', '1 · Clean the image'], ['seg', '2 · Vessel segmentation'], ['cc', '3 · Clean and connect'], ['av', '4 · Artery or vein?']];
let chosenFiles = [], chosenSamples = new Set(), reports = {}, SCR = {}, PARTS = {};
const zoom = s => { const z = $('zoom'); z.querySelector('img').src = s; z.classList.add('on'); };
const ins = t => `<div class="insight">💡 ${t}</div>`;
const col = c => CC[c] || '#fff';

/* ---------- scroll, reveal ---------- */
addEventListener('scroll', () => $('bar').style.width = 100 * scrollY / Math.max(1, document.body.scrollHeight - innerHeight) + '%');
const RIO = new IntersectionObserver(es => es.forEach(x => { if (x.isIntersecting) { x.target.classList.add('in'); RIO.unobserve(x.target); } }), { threshold: .12 });
document.querySelectorAll('.rv').forEach(e => RIO.observe(e));
document.querySelectorAll('.cnt').forEach(e => { const o = new IntersectionObserver(([x]) => { if (x.isIntersecting) { count(e); o.disconnect(); } }); o.observe(e); });

/* ---------- upload queue ---------- */
const refreshGo = () => { const n = chosenFiles.length + chosenSamples.size; $('go').disabled = !n; $('go').textContent = n ? `Analyse ${n} image${n > 1 ? 's' : ''}` : 'Analyse selected images'; };
function addFiles(fl) { [...fl].forEach(f => { chosenFiles.push(f); const im = new Image(); im.src = URL.createObjectURL(f); $('thumbs').append(im); }); refreshGo(); }
$('file').onchange = e => addFiles(e.target.files);
const dr = $('drop'); ['dragover', 'dragenter'].forEach(e => dr.addEventListener(e, v => { v.preventDefault(); dr.classList.add('over'); })); ['dragleave', 'drop'].forEach(e => dr.addEventListener(e, v => { v.preventDefault(); dr.classList.remove('over'); }));
dr.addEventListener('drop', v => addFiles(v.dataTransfer.files));
fetch('/api/status').then(r => r.json()).then(s => { const b = $('badge'); b.textContent = s.message.split(' — ')[0]; b.className = 'badge ' + (s.mode == 'trained' ? 'trained' : '');
  $('samples').innerHTML = s.samples.map(x => `<div class="chip" data-s="${x}"><img src="/samples/${x}" alt="">${x.split('/').pop().replace(/\.\w+$/, '')}</div>`).join('');
  document.querySelectorAll('.chip').forEach(c => c.onclick = () => { const k = c.dataset.s; chosenSamples.has(k) ? chosenSamples.delete(k) : chosenSamples.add(k); c.classList.toggle('on'); refreshGo(); }); });
$('go').onclick = async () => {
  const jobs = [...chosenFiles.map(f => ({ f })), ...[...chosenSamples].map(s => ({ s }))]; if (!jobs.length) return; $('go').disabled = true; $('msg').innerHTML = ''; let last = null;
  for (let i = 0; i < jobs.length; i++) {
    $('msg').innerHTML = `<div class="busy"><div class="spin"></div>Analysing image ${i + 1} of ${jobs.length}… (segmenting vessels, classifying artery/vein, measuring biomarkers)</div>`;
    const fd = new FormData(); jobs[i].f ? fd.append('image', jobs[i].f) : fd.append('sample', jobs[i].s); if ($('fc').checked) fd.append('force_classical', '1');
    try { const r = await fetch('/api/analyze', { method: 'POST', body: fd }), j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); last = addReport(j); }
    catch (e) { $('msg').insertAdjacentHTML('beforeend', `<div class="err"><b>${jobs[i].f ? jobs[i].f.name : jobs[i].s}</b>: ${e.message}</div>`); }
  }
  if (!$('msg').querySelector('.err')) $('msg').innerHTML = ''; else $('msg').querySelector('.busy')?.remove();
  if (last) { activate(last); $('tray').scrollIntoView({ behavior: 'smooth', block: 'start' }); } refreshGo();
};

/* ---------- reports (one separate DOM block per image) ---------- */
function addReport(j) {
  const id = j.id, s = j.screening; reports[id] = j;
  $('tray').insertAdjacentHTML('beforeend', `<div class="tcase" data-id="${id}"><img src="${j.input}" alt=""><div><b>${j.image}</b><small>${DISPLAY[s.verdict]} · ${s.confidence}%</small></div><i style="background:${col(s.verdict)}"></i></div>`);
  $('tray').lastElementChild.onclick = () => activate(id);
  $('reports').insertAdjacentHTML('beforeend', `<div class="rep" id="rep-${id}">${reportHTML(j)}</div>`);
  const el = $('rep-' + id); el.querySelectorAll('.rtabs button').forEach(b => b.onclick = () => pane(id, b.dataset.p));
  el.querySelectorAll('[data-zoom]').forEach(i => i.onclick = () => zoom(i.src));
  bindCmp(el); el.querySelector('[data-replay]').onclick = () => playWalk(id); renderCompare();
  return id;
}
function activate(id) {
  document.querySelectorAll('.tcase').forEach(t => t.classList.toggle('on', t.dataset.id == id)); document.querySelectorAll('.rep').forEach(r => r.classList.toggle('on', r.id == 'rep-' + id)); pane(id, 'walk'); if (!PARTS[id].played) { PARTS[id].played = 1; playWalk(id); }
}
function bindCmp(root) { root.querySelectorAll('.cmp').forEach(c => { const r = c.querySelector('input'); r.oninput = () => { c.querySelector('.top').style.clipPath = `inset(0 ${100 - r.value}% 0 0)`; c.querySelector('.hd').style.left = r.value + '%'; }; }); }
function pane(id, p) {
  const el = $('rep-' + id); el.querySelectorAll('.rtabs button').forEach(b => b.classList.toggle('on', b.dataset.p == p)); el.querySelectorAll('.pane').forEach(x => x.classList.toggle('on', x.dataset.p == p));
  const pn = el.querySelector(`.pane[data-p="${p}"]`); if (p != 'walk') replay(pn); pn.querySelectorAll('.imgs figure').forEach((f, i) => { f.style.transitionDelay = i * .06 + 's'; });
}

function conclusionText(j) {
  const s = j.screening, q = j.analytics.quality, out = s.findings.filter(f => f.status != 'normal'), P = [];
  P.push(out.length ? `<b>Observation.</b> ${out.length} of 6 biomarkers lie outside their reference range: ${out.map(f => `${f.label.toLowerCase()} ${f.status} (${f.value}${f.unit == '%' ? ' %' : ' ' + f.unit})`).join(', ')}.` : `<b>Observation.</b> All six biomarkers lie inside their reference ranges.`);
  if (s.verdict == 'Normal') P.push(`<b>Prediction.</b> The vessel pattern matches <b style="color:${col('Normal')}">Normal</b> (${s.confidence}% match). No condition collected meaningful evidence.`);
  else { const top = [...s.evidence[s.verdict]].sort((a, b) => b.contribution - a.contribution)[0]; P.push(`<b>Prediction.</b> The pattern best matches <b style="color:${col(s.verdict)}">${DISPLAY[s.verdict]}</b> (${s.confidence}% match, ${s.severity.toLowerCase()} strength). The strongest driver is ${top.label.toLowerCase()} being ${top.direction}.${s.secondary.length ? ' Also partly matching: ' + s.secondary.join(', ') + '.' : ''}`); }
  P.push(`<b>Reliability.</b> Image-quality grade ${q.grade} (${q.score}/100); artery/vein reliability ${s.reliability.toLowerCase()}.${s.reliability_notes.length ? ' ' + s.reliability_notes.join(' ') : ''} ${q.grade == 'Poor' ? 'Because quality is poor, treat this result as provisional and re-capture the image.' : ''}`);
  P.push(`<b>Reminder.</b> ${s.note}`);
  return P.map(x => `<p>${x}</p>`).join('');
}

function reportHTML(j) {
  const s = j.screening, A = j.analytics, F = s.findings, c = col(s.verdict), P = s.probabilities, q = A.quality, T = id => `<button data-p="${id[0]}">${id[1]}</button>`;
  const tabs = `<div class="rtabs">${[['walk', '▶ Live walkthrough'], ['verdict', '① Prediction'], ['images', '② All output images'], ['bio', '③ Biomarkers'], ['graphs', '④ All graphs'], ['trust', '⑤ Trust & quality']].map(T).join('')}</div>`;
  /* --- verdict --- */
  const don = donut(Object.keys(P).map(k => ({ n: k, v: P[k], c: col(k) })), s.confidence + '%', 'match');
  const verdict = `<div class="card"><div class="vhero"><img data-zoom src="${j.input}" alt="input"><div><div class="vlab">Screening prediction · ${j.image}</div><div class="vbig" style="color:${c}">${DISPLAY[s.verdict]}</div><div style="color:var(--mut)">${s.verdict_long}</div>
   <div class="vmeta"><span class="tag ${s.verdict == 'Normal' ? 'ok' : 'bad'}">Strength: ${s.severity}</span><span class="tag ${s.reliability == 'High' ? 'ok' : 'warn'}">A/V reliability: ${s.reliability}</span><span class="tag ${q.grade == 'Good' ? 'ok' : q.grade == 'Fair' ? 'warn' : 'bad'}">Image quality: ${q.grade} ${q.score}/100</span>${s.secondary.map(x => `<span class="tag warn">also: ${x}</span>`).join('')}</div></div>
   <div>${don}<div class="dleg">${Object.keys(P).map(k => `<div><i style="background:${col(k)}"></i>${k}<b>${P[k]}%</b></div>`).join('')}</div></div></div></div>`;
  const dom = Object.entries(s.risk).sort((a, b) => b[1] - a[1])[0];
  const gauges = `<div class="card"><h3>Pattern strength per condition</h3><p class="why">How strongly the biomarkers point to each condition (0 = no evidence).</p><div class="gauges3">${Object.keys(s.risk).map(k => gauge(Math.round(s.risk[k]), k == 'High blood pressure' ? 'High BP' : k, col(k), 'pattern')).join('')}</div>${ins(dom[1] < 10 ? 'No condition shows a meaningful pattern — consistent with a normal vessel profile.' : `Strongest pattern: <b>${dom[0]}</b> at ${Math.round(dom[1])}/100.`)}</div>`;
  const evid = `<div class="card"><h3>Why? Evidence behind each condition</h3><p class="why">Each coloured block is one biomarker's contribution. Hover for the numbers.</p>${evidenceBars(s.evidence, BMK, LAB)}${ins(s.reasons.slice(0, 2).join(' '))}</div>`;
  const ftab = `<div class="card" style="margin-top:18px"><h3>Biomarker findings</h3><p class="why">Measured value, reference range and what a deviation means.</p><div class="ch"><table class="ftab"><tr><th>Biomarker</th><th>Value</th><th>Reference</th><th>Status</th><th>Meaning</th></tr>${F.map((f, i) => `<tr style="transition-delay:${i * .08}s"><td>${f.label}</td><td><b>${f.value} ${f.unit}</b></td><td>${f.lo}–${f.hi}</td><td><span class="st ${f.status}">${f.status == 'normal' ? 'within' : f.status}</span></td><td style="color:var(--mut)">${f.meaning}</td></tr>`).join('')}</table></div></div>`;
  const concl = `<div class="card" style="margin-top:18px"><h3>Conclusion</h3><div class="concl">${conclusionText(j)}</div></div>`;
  const pVerdict = `<div class="pane" data-p="verdict">${verdict}<div class="grid2" style="margin-top:18px">${gauges}${evid}</div>${ftab}${concl}</div>`;
  /* --- images --- */
  const st = j.steps, sd = k => st.find(x => x.key == k);
  const cmpHTML = `<div class="card" style="margin-bottom:6px"><h3>Before / after slider</h3><p class="why">Drag across the image: detected vessels ⇆ original fundus.</p><div class="cmp"><img src="${sd('p0').url}" alt="original"><div class="top"><img src="${sd('g3').url}" alt="overlay"></div><div class="hd"></div><input type="range" min="0" max="100" value="50"><span style="left:10px">Detected vessels</span><span style="right:10px">Original</span></div></div>`;
  const grp = g => `<div class="imgs">${st.filter(x => x.group == g).map(x => `<figure><img data-zoom src="${x.url}" alt=""><figcaption>${x.title}<small>${NOTE[x.key] || ''}</small></figcaption></figure>`).join('')}</div>`;
  const groups = GROUPS.map(([g, t]) => `<div class="sg">${t}</div><div class="imgs">${st.filter(x => x.group == g).map(x => `<figure><img data-zoom src="${x.url}" alt=""><figcaption>${x.title}<small>${NOTE[x.key] || ''}</small></figcaption></figure>`).join('')}</div>`).join('');
  const masks = `<div class="sg">5 · Individual masks (downloadable)</div><div class="imgs">${j.masks.map(m => `<figure><img data-zoom src="${m.url}" alt=""><figcaption>${m.title}<small><a style="color:var(--c2)" href="${m.url}" download>download png</a></small></figcaption></figure>`).join('')}</div>`;
  const figs = `<div class="sg">6 · Full-pipeline figures</div><div class="imgs">${j.figures.map(f => `<figure><img data-zoom src="${f.url}" alt=""><figcaption>${f.title}</figcaption></figure>`).join('')}</div>`;
  const pImages = `<div class="pane" data-p="images">${cmpHTML}${groups}${masks}${figs}</div>`;
  /* --- biomarkers --- */
  const rv = f => clamp(.4 + .3 * (f.value - f.lo) / (f.hi - f.lo), .06, 1), nOut = F.filter(f => f.status != 'normal').length;
  const rad = radar(F.map(f => SHORT[f.key]), [{ n: 'This image', c: c, vals: F.map(rv), raw: F.map(f => f.value + ' ' + f.unit) }], { lo: F.map(() => .4), hi: F.map(() => .7) });
  const bioInner = `<div class="card"><h3>Six biomarkers vs reference</h3><p class="why">Green band = reference range. Bars turn red outside it.</p>${bullets(F)}</div>
   <div class="card"><h3>Radar · shape of this retina</h3><p class="why">The green ring is the reference range; a healthy retina stays inside it.</p>${rad}${ins(nOut ? `<b>${nOut}</b> of 6 axes leave the green band: ${F.filter(f => f.status != 'normal').map(f => SHORT[f.key]).join(', ')}.` : 'All six axes stay inside the reference ring.')}</div>`;
  const pBio = `<div class="pane" data-p="bio"><div class="grid2">${bioInner}</div></div>`;
  /* --- graphs --- */
  const W = A.width_hist, mid = Math.floor, peak = A.rings.density.indexOf(Math.max(...A.rings.density)), seg = A.segments, hiT = seg.filter(p => p.tort > 1.15).length;
  const roseMax = A.rose.values.indexOf(Math.max(...A.rose.values)), gflat = [];
  A.grid.forEach((r, i) => r.forEach((v, k) => v != null && gflat.push([v, i, k]))); gflat.sort((a, b) => b[0] - a[0]);
  const nm = (i, k, n) => `${i < n / 3 ? 'upper' : i < 2 * n / 3 ? 'central' : 'lower'}-${k < n / 3 ? 'left' : k < 2 * n / 3 ? 'centre' : 'right'}`;
  const qt = A.quadrants, aT = qt.reduce((a, x) => a + x.artery, 0), vT = qt.reduce((a, x) => a + x.vein, 0), fn = A.funnel, rem = fn[0].value - fn[1].value, add = fn[2].value - fn[1].value;
  const dev = F.map(f => ({ label: LAB[f.key], v: f.deviation, tip: `${LAB[f.key]}: ${f.value} (${f.status}) · ${f.deviation > 0 ? '+' : ''}${f.deviation} range-widths` }));
  const card = (t, w, body, insight) => `<div class="card"><h3>${t}</h3><p class="why">${w}</p>${body}${insight ? ins(insight) : ''}</div>`;
  const sPts = seg.map(p => ({ x: p.len, y: p.tort, c: p.t == 1 ? ART : p.t == 0 ? VEIN : '#aab3d0', r: 5, tip: `${p.t == 1 ? 'Artery' : p.t == 0 ? 'Vein' : 'Vessel'} segment<br>length ${p.len}px · tortuosity ${p.tort}` }));
  const cDev = card('Deviation from reference (diverging bars)', 'Signed distance of each biomarker from its range, in range-widths. 0 = inside.', divBars(dev, { pl: 'above range', nl: 'below range', cp: '#fbbf24', cn: '#fb7185' }), F.some(f => f.status != 'normal') ? `Largest deviation: <b>${[...dev].sort((a, b) => Math.abs(b.v) - Math.abs(a.v))[0].label}</b>.` : 'Nothing deviates — all biomarkers are inside range.');
  const cWid = card('Vessel-width distribution (area chart)', 'Share of centre-line pixels at each vessel diameter, artery vs vein.', areaChart(W.bins, [{ n: 'Artery', c: ART, vals: W.artery }, { n: 'Vein', c: VEIN, vals: W.vein }, { n: 'All', c: '#22d3ee', vals: W.all }], { xl: 'vessel diameter (px)', yl: '% of vessel length', id: 'w' + j.id }), `Median width: artery <b>${W.artery_median}px</b>, vein <b>${W.vein_median}px</b> → ${Math.abs(W.artery_median - W.vein_median) < .5 ? 'almost identical, so the artery/vein split is weak and AVR should be read with caution' : W.artery_median < W.vein_median ? 'arteries are thinner than veins, as expected' : 'arteries are wider than veins — unusual, check the A/V labelling'}.`);
  const cBut = card('Artery vs vein by quadrant (butterfly)', 'Classified artery and vein pixels in each quarter of the image.', butterfly(qt), `Overall <b>${aT.toLocaleString()}</b> artery px vs <b>${vT.toLocaleString()}</b> vein px (ratio ${(aT / Math.max(vT, 1)).toFixed(2)}).`);
  const graphsInner = `
   ${cDev}
   ${cWid}
   ${card('Radial density profile (line chart)', 'Vessel density in 8 concentric rings from the retina centre outwards.', areaChart(A.rings.rings, [{ n: 'Density %', c: '#fbbf24', vals: A.rings.density }], { xl: 'distance from centre (% of radius)', yl: 'vessel density %', id: 'r' + j.id }), `Density peaks at <b>${A.rings.rings[peak]}</b> radius (${A.rings.density[peak]}%) and falls to ${A.rings.density.at(-1)}% at the rim. A flat or very low profile suggests rarefaction.`)}
   ${card('Vessel direction (polar rose)', 'Total vessel length by orientation (0°–180°, mirrored).', rose(A.rose.labels, A.rose.values), `Most vessel length runs near <b>${A.rose.labels[roseMax]}</b> (${A.rose.values[roseMax]}%) — the dominant vessel direction in this image.`)}
   ${card('Density heat-map (6 × 6 grid)', 'Vessel density in each region of the retina.', heatmap(A.grid), gflat.length ? `Densest region: <b>${nm(gflat[0][1], gflat[0][2], 6)}</b> (${gflat[0][0]}%); sparsest: <b>${nm(gflat.at(-1)[1], gflat.at(-1)[2], 6)}</b> (${gflat.at(-1)[0]}%).` : '')}
   ${card('Segment tortuosity (scatter)', 'Every vessel segment: length vs tortuosity (path ÷ straight line). Red = artery, blue = vein.', scatter(sPts, { xr: [0, Math.ceil(Math.max(60, ...seg.map(p => p.len)) / 20) * 20], yr: [1, Math.max(1.5, Math.ceil(Math.max(...seg.map(p => p.tort), 1.2) * 10) / 10)], xl: 'segment length (px)', yl: 'tortuosity', w: 520, h: 380, d: 1 }), seg.length ? `<b>${hiT}</b> of ${seg.length} segments (${(100 * hiT / seg.length).toFixed(0)}%) exceed 1.15, the normal upper limit. ${hiT / seg.length >= .3 ? 'A large share — consistent with a diabetic-type (twisted-vessel) pattern.' : 'Only a minority are twisted, which is within normal variation.'}` : 'Too few segments to plot.')}
   ${cBut}
   ${card('Segmentation confidence (histogram)', 'Distribution of the network\'s vessel probability. Green = confident, amber = uncertain (0.3–0.7).', hist(A.prob_hist.labels, A.prob_hist.values, i => (i < 3 || i > 6) ? '#34d399' : '#fbbf24', { xl: 'vessel probability' }), `<b>${A.prob_hist.confident_pct}%</b> of candidate vessel pixels are confidently classified.`)}
   ${card('Clean-and-connect funnel', 'How many pixels survive each post-processing stage.', funnel(fn), `Noise removal dropped <b>${rem.toLocaleString()}</b> px; gap-bridging added <b>${add.toLocaleString()}</b> px; the centre-line has <b>${fn[3].value.toLocaleString()}</b> px.`)}
  `;
  const pGraphs = `<div class="pane" data-p="graphs"><div class="grid2">${graphsInner}</div></div>`;
  /* --- trust --- */
  const pv = j.provenance;
  const pTrust = `<div class="pane" data-p="trust"><div class="grid2"><div class="card"><h3>Image & measurement quality</h3><p class="why">Nine automatic checks run on this image.</p><div class="ch" style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">${ring(q.score, q.grade + ' quality', q.grade == 'Good' ? '#34d399' : q.grade == 'Fair' ? '#fbbf24' : '#fb7185', true)}<div style="flex:1;min-width:220px;font-size:13px;color:var(--mut)">${q.items.filter(i => i.status == 'warn').length ? 'Warnings below explain what limits confidence.' : 'All checks passed.'}</div></div>
    ${q.items.map(i => `<div class="qrow"><i class="${i.status}">${i.status == 'pass' ? '✓' : '!'}</i><span>${i.name}</span><b>${i.value}</b><small>${i.text}</small></div>`).join('')}</div>
   <div><div class="card"><h3>Provenance (reproducibility)</h3><p class="why">Same image + same engine = same numbers.</p><div class="kv"><div>Image: <b>${j.image}</b></div><div>SHA-256: <b>${pv.sha256}…</b></div><div>Analysed: <b>${pv.analysed_at}</b></div><div>Engine: <b>${pv.engine}</b></div><div>Segmentation model: <b>${pv.model}</b></div><div>Threshold: <b>${pv.threshold}</b></div><div>Graph: <b>${j.graph.nodes} nodes · ${j.graph.segments} segments</b></div><div>Run time: <b>${pv.seconds}s</b></div></div>
   <div class="row noprint"><a class="cta" style="margin:0" href="${j.results_json}" target="_blank">Download results.json</a><button class="cta ghost" style="margin:0" onclick="window.print()">Print / save PDF report</button></div></div>
   <div class="card" style="margin-top:18px"><h3>How this prediction was made</h3><p class="note" style="margin:0">${s.method}</p><ul class="note"><li>Measured: six vessel biomarkers from this image only.</li><li>Predicted: ${s.verdict} (${s.confidence}% pattern match).</li>${s.reliability_notes.map(n => `<li style="color:#fde68a">${n}</li>`).join('')}</ul><div class="warnbox">${s.note}</div></div></div></div></div>`;
  PARTS[j.id] = { j, st, graphsInner, grp, cmpHTML, cWid, cBut, cDev, bioInner, verdict, gauges, evid, ftab, concl: conclusionText(j), masks: j.masks, figs: j.figures };
  const pWalk = `<div class="pane" data-p="walk"><div class="row noprint" style="margin:0 0 6px"><div class="rail"></div><button class="cta ghost" style="margin:0;padding:8px 16px" data-replay>↻ Replay animation</button></div><div class="walkbox"></div></div>`;
  return `<div class="card" style="margin-bottom:0;padding:14px 22px"><b>${j.image}</b> <span style="color:var(--mut)">· report ${j.id} · ${j.engine_label}</span></div>${tabs}${pWalk}${pVerdict}${pImages}${pBio}${pGraphs}${pTrust}`;
}

/* ---------- reference sections ---------- */
fetch('/api/reference').then(r => r.json()).then(R => {
  const m = R.models, ms = m.metrics, C = ['#7c5cff', '#22d3ee', '#fbbf24'], N = ['UNet++', 'Attention U-Net', 'UNet++ + SCSE'];
  $('mrad').innerHTML = radar(ms, N.map((n, i) => ({ n, c: C[i], vals: m[n].map(v => clamp((v - .6) / .4, .05, 1)), raw: m[n] })), null, 360) + legend(N.map((n, i) => ({ n, c: C[i] })));
  $('mlol').innerHTML = lollipop(ms, N.map((n, i) => ({ n, c: C[i], v: m[n] })));
  const bd = N.reduce((a, n) => m[n][0] > m[a][0] ? n : a), bs = N.reduce((a, n) => m[n][1] > m[a][1] ? n : a), bp = N.reduce((a, n) => m[n][2] > m[a][2] ? n : a);
  $('minsight').innerHTML = `💡 Best Dice: <b>${bd}</b> (${m[bd][0]}); best sensitivity: <b>${bs}</b> (${m[bs][1]}); best specificity: <b>${bp}</b> (${m[bp][2]}). The three networks are within ${(Math.max(...N.map(n => m[n][0])) - Math.min(...N.map(n => m[n][0]))).toFixed(3)} Dice of each other.`;
  const g = R.group, B = g.biomarkers, mx = B.map((_, i) => Math.max(g.Normal[i], g.Disease[i]));
  $('crad').innerHTML = radar(B, [{ n: 'Normal', c: '#34d399', vals: g.Normal.map((v, i) => v / mx[i] * .9), raw: g.Normal }, { n: 'Disease', c: '#fb7185', vals: g.Disease.map((v, i) => v / mx[i] * .9), raw: g.Disease }], null, 380) + legend([{ n: 'Normal', c: '#34d399' }, { n: 'Disease', c: '#fb7185' }]);
  const ch = B.map((b, i) => ({ label: b, v: +((g.Disease[i] - g.Normal[i]) / g.Normal[i] * 100).toFixed(1), tip: `${b}: ${g.Normal[i]} → ${g.Disease[i]}` }));
  $('cdiv').innerHTML = divBars(ch, { cp: '#34d399', cn: '#fb7185', pl: '% higher in disease', nl: '% lower in disease' });
  const big = [...ch].sort((a, b) => Math.abs(b.v) - Math.abs(a.v))[0]; $('cins').innerHTML = `💡 Biggest shift: <b>${big.label}</b> (${big.v > 0 ? '+' : ''}${big.v}%). ` + ch.filter(c => c.v < 0).length + ' of 6 biomarkers are lower in disease.';
  const showCase = k => { document.querySelectorAll('#cases .tabs button').forEach(x => x.classList.toggle('on', x.dataset.k == k)); const im = $('caseimg'); im.classList.remove('in'); im.src = k == 'normal' ? '/static/reference/normal_unetpp_scse.png' : '/static/reference/disease_unetpp_scse.jpg'; setTimeout(() => im.classList.add('in'), 80);
    const F = Object.entries(R.cases[k]).filter(([a]) => CK[a]).map(([a, v]) => { const key = CK[a], [lo, hi, u] = RNG[key]; return { key, label: LAB[key], unit: u, value: v, lo, hi, status: v < lo ? 'low' : v > hi ? 'high' : 'normal' }; });
    $('casebm').innerHTML = bullets(F) + `<div class="warnbox">Saved values from the trained network on this reference image, shown against the same ranges. Predictions are made on live uploads above, where image quality and A/V reliability are checked too.</div>`; replay($('casebm')); };
  document.querySelectorAll('#cases .tabs button').forEach(b => b.onclick = () => showCase(b.dataset.k)); showCase('normal');
  const sig = { Glaucoma: 'Low vessel density · narrow vessels · narrow arterioles', Diabetes: 'High tortuosity · dilated veins · high density', 'High blood pressure': 'Low AVR · narrow arterioles · slightly higher tortuosity', Normal: 'All six inside range' };
  $('sigtab').innerHTML = '<tr><th>Outcome</th><th>Biomarker signature</th></tr>' + Object.keys(sig).map(k => `<tr class="go"><td style="color:${col(k)}"><b>${k}</b></td><td>${sig[k]}</td></tr>`).join('');
  $('rngtab').innerHTML = '<tr><th>Biomarker</th><th>Normal range</th></tr>' + BMK.map(k => `<tr class="go"><td>${LAB[k]}</td><td>${RNG[k][0]}–${RNG[k][1]} ${RNG[k][2]}</td></tr>`).join('');
  watchCharts(document);
});

/* ---------- artery vs vein Dice per image ---------- */
fetch('/api/av_dice').then(r => r.json()).then(D => {
  const R = D.rows, nm = r => r.image.replace('DRIVE_test_', 'test '), mean = a => a.reduce((x, y) => x + y, 0) / a.length, med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const M = { att: { n: 'Attention U-Net', c: '#60a5fa', a: R.map(r => r.attention_artery), v: R.map(r => r.attention_vein) }, sc: { n: 'UNet++ + SCSE', c: '#f472b6', a: R.map(r => r.scse_artery), v: R.map(r => r.scse_vein) } };
  if (D.source == 'digitised') $('avwarn').innerHTML = `<div class="warnbox">⚠ ${D.note}</div>`;
  const st = (t, v) => `<div>${t}<b>${v}</b></div>`;
  $('avstats').innerHTML = [st('Attention · mean artery Dice', mean(M.att.a).toFixed(3)), st('Attention · mean vein Dice', mean(M.att.v).toFixed(3)), st('SCSE · mean artery Dice', mean(M.sc.a).toFixed(3)), st('SCSE · mean vein Dice', mean(M.sc.v).toFixed(3)), st('Images where artery > vein (SCSE)', M.sc.a.filter((x, i) => x > M.sc.v[i]).length + ' / ' + R.length)].join('');
  const rng = [.24, .56], pts = (m, id) => R.map((r, i) => ({ x: m.a[i], y: m.v[i], c: m.c, id: i, r: 7, tip: `<b>${r.image}</b><br>${m.n}<br>artery ${m.a[i]} · vein ${m.v[i]}` }));
  const scOpt = { xr: rng, yr: rng, xl: 'Artery Dice', yl: 'Vein Dice', diag: true, w: 500, h: 440, cls: 'avs', n: 4 };
  $('sc1').innerHTML = scatter(pts(M.att), scOpt); $('sc2').innerHTML = scatter(pts(M.sc), scOpt);
  $('avsel').innerHTML = R.map((r, i) => `<option value="${i}">${r.image}</option>`).join('');
  const detail = i => { const r = R[i], bar = (l, v, c) => `<div class="mrow"><span>${l}</span><div class="trk"><div class="fill" style="width:${v * 100 / .6}%;background:${c};transition:none"></div></div><b>${v.toFixed(3)}</b></div>`;
    const box = (m, a, v) => `<div class="mini"><b style="color:${m.c}">${m.n}</b>${bar('Artery', a, ART)}${bar('Vein', v, VEIN)}<div style="font-size:12px;color:var(--mut);margin-top:6px">gap (artery − vein): <b style="color:var(--ink)">${(a - v).toFixed(3)}</b> → ${a > v ? 'arteries segmented better' : 'veins segmented better'}</div></div>`;
    $('avdet').innerHTML = `<div class="dsel">${box(M.att, r.attention_artery, r.attention_vein)}${box(M.sc, r.scse_artery, r.scse_vein)}</div>` + ins(`<b>${r.image}</b>: SCSE ${r.scse_artery >= r.attention_artery ? 'improves' : 'lowers'} artery Dice by ${Math.abs(r.scse_artery - r.attention_artery).toFixed(3)} and ${r.scse_vein >= r.attention_vein ? 'improves' : 'lowers'} vein Dice by ${Math.abs(r.scse_vein - r.attention_vein).toFixed(3)} versus Attention U-Net.`);
    document.querySelectorAll('.avs .sp').forEach(p => p.classList.toggle('sel', +p.dataset.id === i)); $('avsel').value = i; };
  $('avsel').onchange = e => detail(+e.target.value); document.querySelectorAll('.avs').forEach(s => s.addEventListener('click', e => { const p = e.target.closest('.sp'); if (p && p.dataset.id != null) detail(+p.dataset.id); })); detail(R.findIndex(r => r.image == 'DRIVE_test_19') >= 0 ? 19 : 0);
  const labs = R.map(r => r.image.replace('DRIVE_test_', ''));
  $('avline').innerHTML = areaChart(labs, [{ n: 'Attention · artery', c: '#60a5fa', vals: M.att.a }, { n: 'Attention · vein', c: '#93c5fd', vals: M.att.v }, { n: 'SCSE · artery', c: '#f472b6', vals: M.sc.a }, { n: 'SCSE · vein', c: '#fbcfe8', vals: M.sc.v }], { xl: 'test image number', yl: 'Dice', area: false, max: .6, id: 'av' });
  const worst = M.sc.v.indexOf(Math.min(...M.sc.v)), best = M.sc.v.indexOf(Math.max(...M.sc.v));
  $('avinsight1').innerHTML = `💡 Artery Dice is higher than vein Dice for ${M.att.a.filter((x, i) => x > M.att.v[i]).length}/${R.length} images with Attention U-Net and ${M.sc.a.filter((x, i) => x > M.sc.v[i]).length}/${R.length} with SCSE — veins are the harder class. SCSE vein Dice ranges from <b>${M.sc.v[worst].toFixed(2)}</b> (${R[worst].image}) to <b>${M.sc.v[best].toFixed(2)}</b> (${R[best].image}).`;
  const dbs = m => R.map((r, i) => ({ label: nm(r), a: m.a[i], b: m.v[i] })).sort((x, y) => (y.a - y.b) - (x.a - x.b));
  const dbo = { xr: [.2, .6], ca: ART, cb: VEIN, na: 'Artery Dice', nb: 'Vein Dice' };
  $('db1').innerHTML = dumbbell(dbs(M.att), dbo); $('db2').innerHTML = dumbbell(dbs(M.sc), dbo);
  const dd = k => R.map((r, i) => ({ label: nm(r), v: +(M.sc[k][i] - M.att[k][i]).toFixed(3) })).sort((x, y) => y.v - x.v);
  $('dv1').innerHTML = divBars(dd('a'), { cp: '#34d399', cn: '#fb7185', pl: 'SCSE better', nl: 'Attention better', rh: 22 }); $('dv2').innerHTML = divBars(dd('v'), { cp: '#34d399', cn: '#fb7185', pl: 'SCSE better', nl: 'Attention better', rh: 22 });
  const wa = dd('a').filter(x => x.v > 0).length, wv = dd('v').filter(x => x.v > 0).length;
  $('avinsight2').innerHTML = `💡 SCSE beats Attention U-Net on <b>${wa}/${R.length}</b> images for arteries and <b>${wv}/${R.length}</b> for veins; mean change ${(mean(M.sc.a) - mean(M.att.a)).toFixed(3)} (artery) and ${(mean(M.sc.v) - mean(M.att.v)).toFixed(3)} (vein).`;
  watchCharts(document);
});


/* ---------- animated walkthrough (every output image + every graph, step by step) ---------- */
const STEPDEF = [['pre', 'Clean the image', 'Original fundus → green channel → CLAHE → Retinex → the 3-channel network input.'],
  ['seg', 'Vessel segmentation', 'Probability map → binary mask → overlay painted on your fundus photo.'],
  ['cc', 'Clean & connect', 'Threshold → remove noise → bridge gaps → skeleton → vessel graph.'],
  ['av', 'Artery or vein?', 'Artery / vein classification of every vessel segment, with the two masks and where each sits in the retina.'],
  ['bm', 'Six biomarkers vs reference', 'Green band = reference range; bars turn red outside it.'],
  ['dx', 'Disease screening', 'Prediction, pattern strength per condition and the evidence behind it.'],
  ['gr', 'Graphs computed from this image', 'Different image → different graphs. Each chart has a one-line reading under it.'],
  ['end', 'Conclusion & complete outputs', 'What the results mean, plus the full-pipeline figures and downloads.']];
async function playWalk(id) {
  const el = $('rep-' + id), box = el.querySelector('.walkbox'), rail = el.querySelector('.rail'), P = PARTS[id], j = P.j, tok = P.tok = (P.tok || 0) + 1, live = () => P.tok === tok;
  const active = () => el.querySelector('.pane[data-p="walk"]').classList.contains('on') && el.classList.contains('on');
  box.innerHTML = ''; rail.innerHTML = STEPDEF.map(s => `<i>${s[1]}</i>`).join('');
  const figs = async (host, list, delay = 420) => { const g = document.createElement('div'); g.className = 'imgs'; host.append(g);
    for (const x of list) { if (!live()) return; const f = document.createElement('figure'); f.innerHTML = `<img src="${x.url}" onclick="zoom(this.src)" alt=""><figcaption>${x.title}<small>${NOTE[x.key] || ''}</small></figcaption>`; g.append(f); await sleep(40); f.classList.add('in'); await sleep(delay); } };
  for (let i = 0; i < STEPDEF.length; i++) {
    if (!live()) return; const [k, t, d] = STEPDEF[i]; rail.children[i].classList.add('on');
    const s = document.createElement('div'); s.className = 'card step'; s.innerHTML = `<h3><span class="no">${i + 1}</span>${t}</h3><p class="why">${d}</p><div class="sb"></div>`; box.append(s); const B = s.querySelector('.sb');
    if (active()) s.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (k == 'pre' || k == 'cc') await figs(B, P.st.filter(x => x.group == k));
    if (k == 'seg') { await figs(B, P.st.filter(x => x.group == 'seg')); B.insertAdjacentHTML('beforeend', P.cmpHTML); bindCmp(B); }
    if (k == 'av') { await figs(B, [...P.st.filter(x => x.group == 'av'), ...P.masks.filter(m => /Artery|Vein/.test(m.title))]);
      B.insertAdjacentHTML('beforeend', `<div class="grid2" style="margin-top:16px">${P.cBut}${P.cWid}</div>`); watchCharts(B); await sleep(1400); }
    if (k == 'bm') { B.innerHTML = `<div class="grid2">${P.bioInner}</div>`; watchCharts(B); await sleep(1800); }
    if (k == 'dx') { B.innerHTML = `${P.verdict}<div class="grid2" style="margin-top:18px">${P.gauges}${P.evid}</div>${P.ftab}`; B.querySelectorAll('[data-zoom]').forEach(i => i.onclick = () => zoom(i.src)); watchCharts(B); await sleep(1800); }
    if (k == 'gr') { B.innerHTML = `<div class="grid2">${P.cDev}${graphsFor(P)}</div>`; watchCharts(B); await sleep(900); }
    if (k == 'end') { B.innerHTML = `<div class="concl">${P.concl}</div><div class="sg">Complete-output figures for this image</div>`; await figs(B, P.figs, 300);
      B.insertAdjacentHTML('beforeend', `<div class="row noprint"><a class="cta" style="margin:0" href="${j.results_json}" target="_blank">Download results.json</a><button class="cta ghost" style="margin:0" onclick="window.print()">Print / save PDF report</button></div>`); }
    await sleep(350);
  }
}
/* the remaining graph cards (everything in the Graphs tab except the ones already shown in earlier steps) */
function graphsFor(P) { const tmp = document.createElement('div'); tmp.innerHTML = P.graphsInner; return [...tmp.children].filter(c => !/Vessel-width|Artery vs vein by quadrant|Deviation from reference/.test(c.querySelector('h3')?.textContent || '')).map(c => c.outerHTML).join(''); }

/* ---------- compare all analysed images (graphs differ per image) ---------- */
function renderCompare() {
  const ids = Object.keys(reports); const host = $('cmpsec'); if (!host) return; host.hidden = ids.length < 2; if (ids.length < 2) return;
  const R = ids.map(i => reports[i]), pal = ['#22d3ee', '#fbbf24', '#c084fc', '#fb7185', '#34d399', '#60a5fa', '#f472b6', '#a3e635'];
  const rv = f => clamp(.4 + .3 * (f.value - f.lo) / (f.hi - f.lo), .06, 1);
  const ser = R.map((j, i) => ({ n: j.image, c: pal[i % pal.length], vals: j.screening.findings.map(rv), raw: j.screening.findings.map(f => f.value + ' ' + f.unit) }));
  $('cmprad').innerHTML = radar(BMK.map(k => SHORT[k]), ser, { lo: BMK.map(() => .4), hi: BMK.map(() => .7) }, 360) + legend(ser);
  const cl = Object.keys(CC); $('cmplol').innerHTML = lollipop(R.map(j => j.image), cl.map(c => ({ n: c, c: CC[c], v: R.map(j => j.screening.probabilities[c] / 100) })));
  $('cmptab').innerHTML = '<tr><th>Image</th><th>Prediction</th><th>Match</th><th>Quality</th>' + BMK.map(k => `<th>${SHORT[k]}</th>`).join('') + '</tr>' + R.map(j => `<tr class="go"><td><b>${j.image}</b></td><td style="color:${col(j.screening.verdict)}">${DISPLAY[j.screening.verdict]}</td><td>${j.screening.confidence}%</td><td>${j.analytics.quality.grade}</td>` + j.screening.findings.map(f => `<td style="color:${f.status == 'normal' ? 'var(--ok)' : 'var(--bad)'}">${f.value}</td>`).join('') + '</tr>').join('');
  const dif = BMK.map((k, x) => { const v = R.map(j => j.screening.findings[x].value); return { k, spread: (Math.max(...v) - Math.min(...v)) / (RNG[k][1] - RNG[k][0]) }; }).sort((a, b) => b.spread - a.spread)[0];
  $('cmpins').innerHTML = `💡 Across these ${R.length} images the biomarker that differs most is <b>${LAB[dif.k]}</b>. Each image keeps its own prediction, quality grade and graphs in its own report above.`;
  watchCharts(host);
}
