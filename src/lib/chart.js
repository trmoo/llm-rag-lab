/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * chart.js — 그래프를 직접 그린다 (외부 그래프 라이브러리 0개)
 *
 * 교실 인터넷이 끊겨도 열려야 하므로 CDN 을 쓰지 않는다.
 * 캔버스 하나를 만들어 돌려주고, .redraw() 를 부르면 다시 그린다.
 * 화면 크기가 바뀔 때는 화면 모듈이 ui.onResize(chart.redraw) 로 걸어 준다.
 * ========================================================================== */

export const PALETTE = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb923c', '#22d3ee', '#f87171'];

/** 캔버스를 만들고 고해상도 화면(레티나)에 맞춰 크기를 맞춘다. */
function makeCanvas(height) {
  const cv = document.createElement('canvas');
  cv.className = 'chart';
  cv.style.width = '100%';
  cv.style.height = height + 'px';
  return cv;
}

function prepare(cv, height) {
  const dpr = Math.min(3, window.devicePixelRatio || 1);
  const w = Math.max(60, cv.clientWidth || cv.parentElement?.clientWidth || 320);
  const hh = height;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(hh * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, hh);
  return { g, w, h: hh };
}

function css(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const theme = () => ({
  fg: css('--fg', '#e6edf7'),
  dim: css('--dim', '#8b9bb4'),
  grid: css('--grid', 'rgba(255,255,255,.10)'),
  axis: css('--axis', 'rgba(255,255,255,.35)'),
});

/* ───────────────────────────── 꺾은선 그래프 ────────────────────────────── */
/**
 * series: [{ name, points: [[x,y],…], color, dash }]
 */
export function lineChart({ height = 220, series, xLabel, yLabel, yMin, yMax, xMin, xMax, legend = true, hLine }) {
  const cv = makeCanvas(height);
  const draw = () => {
    const { g, w, h } = prepare(cv, height);
    const t = theme();
    const padL = 46;
    const padR = 12;
    const padT = 12;
    const padB = xLabel ? 34 : 24;
    const all = series.flatMap((s) => s.points);
    if (!all.length) return;
    const x0 = xMin !== undefined ? xMin : Math.min(...all.map((p) => p[0]));
    const x1 = xMax !== undefined ? xMax : Math.max(...all.map((p) => p[0]));
    let y0 = yMin !== undefined ? yMin : Math.min(...all.map((p) => p[1]));
    let y1 = yMax !== undefined ? yMax : Math.max(...all.map((p) => p[1]));
    if (y1 - y0 < 1e-9) { y1 = y0 + 1; }
    const sx = (v) => padL + ((v - x0) / (x1 - x0 || 1)) * (w - padL - padR);
    const sy = (v) => h - padB - ((v - y0) / (y1 - y0 || 1)) * (h - padT - padB);

    // 눈금
    g.strokeStyle = t.grid;
    g.lineWidth = 1;
    g.fillStyle = t.dim;
    g.font = '11px system-ui, sans-serif';
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4;
      const y = sy(v);
      g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
      g.fillText(fmtNum(v), padL - 6, y);
    }
    g.textAlign = 'center';
    g.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) {
      const v = x0 + ((x1 - x0) * i) / 4;
      g.fillText(fmtNum(v), sx(v), h - padB + 6);
    }
    if (xLabel) { g.fillText(xLabel, (padL + w - padR) / 2, h - 14); }
    if (yLabel) {
      g.save(); g.translate(12, (padT + h - padB) / 2); g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(yLabel, 0, 0); g.restore();
    }
    if (hLine !== undefined) {
      g.strokeStyle = t.axis;
      g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(padL, sy(hLine)); g.lineTo(w - padR, sy(hLine)); g.stroke();
      g.setLineDash([]);
    }

    series.forEach((s, i) => {
      g.strokeStyle = s.color || PALETTE[i % PALETTE.length];
      g.lineWidth = s.width || 2.2;
      g.setLineDash(s.dash || []);
      g.beginPath();
      s.points.forEach((p, j) => (j ? g.lineTo(sx(p[0]), sy(p[1])) : g.moveTo(sx(p[0]), sy(p[1]))));
      g.stroke();
      g.setLineDash([]);
      if (s.dots) {
        g.fillStyle = s.color || PALETTE[i % PALETTE.length];
        s.points.forEach((p) => { g.beginPath(); g.arc(sx(p[0]), sy(p[1]), 3, 0, 7); g.fill(); });
      }
    });

    if (legend && series.length > 1) {
      let lx = padL + 6;
      g.textAlign = 'left'; g.textBaseline = 'middle';
      series.forEach((s, i) => {
        const c = s.color || PALETTE[i % PALETTE.length];
        g.fillStyle = c;
        g.fillRect(lx, padT + 2, 14, 3);
        g.fillStyle = t.fg;
        g.fillText(s.name, lx + 19, padT + 4);
        lx += 26 + g.measureText(s.name).width;
      });
    }
  };
  cv.redraw = draw;
  requestAnimationFrame(draw);
  return cv;
}

/* ───────────────────────────── 막대 그래프 ─────────────────────────────── */
/** items: [{ label, value, color }] */
export function barChart({ height = 200, items, max, horizontal = false, valueFmt }) {
  const cv = makeCanvas(height);
  const draw = () => {
    const { g, w, h } = prepare(cv, height);
    const t = theme();
    const top = max !== undefined ? max : Math.max(1e-9, ...items.map((d) => d.value));
    g.font = '12px system-ui, sans-serif';
    if (horizontal) {
      const padL = Math.min(140, Math.max(...items.map((d) => g.measureText(d.label).width)) + 12);
      const rowH = (h - 8) / Math.max(1, items.length);
      items.forEach((d, i) => {
        const y = 4 + i * rowH;
        const bh = Math.max(8, rowH * 0.62);
        g.fillStyle = t.dim; g.textAlign = 'right'; g.textBaseline = 'middle';
        g.fillText(d.label, padL - 8, y + rowH / 2);
        const bw = ((d.value / top) * (w - padL - 56));
        g.fillStyle = d.color || PALETTE[i % PALETTE.length];
        roundRect(g, padL, y + (rowH - bh) / 2, Math.max(1, bw), bh, 3);
        g.fill();
        g.fillStyle = t.fg; g.textAlign = 'left';
        g.fillText(valueFmt ? valueFmt(d.value) : fmtNum(d.value), padL + bw + 6, y + rowH / 2);
      });
    } else {
      const padB = 26;
      const colW = (w - 8) / Math.max(1, items.length);
      items.forEach((d, i) => {
        const bw = Math.max(6, colW * 0.6);
        const x = 4 + i * colW + (colW - bw) / 2;
        const bh = ((d.value / top) * (h - padB - 18));
        g.fillStyle = d.color || PALETTE[i % PALETTE.length];
        roundRect(g, x, h - padB - bh, bw, Math.max(1, bh), 3);
        g.fill();
        g.fillStyle = t.dim; g.textAlign = 'center'; g.textBaseline = 'top';
        g.fillText(d.label, x + bw / 2, h - padB + 5);
        g.fillStyle = t.fg; g.textBaseline = 'bottom';
        g.fillText(valueFmt ? valueFmt(d.value) : fmtNum(d.value), x + bw / 2, h - padB - bh - 3);
      });
    }
  };
  cv.redraw = draw;
  requestAnimationFrame(draw);
  return cv;
}

/* ────────────────────────────── 흩뿌린 점 ─────────────────────────────── */
/**
 * points: [{ x, y, label, color, r }]
 * onDrag(index, x, y) 를 주면 점을 끌 수 있다 (좌표계 값으로 돌려준다).
 */
export function scatter({ height = 300, points, xMin = -1, xMax = 1, yMin = -1, yMax = 1, onDrag, axes = true, links }) {
  const cv = makeCanvas(height);
  let hot = -1;
  const toPx = (w, h) => ({
    sx: (v) => 30 + ((v - xMin) / (xMax - xMin)) * (w - 42),
    sy: (v) => h - 26 - ((v - yMin) / (yMax - yMin)) * (h - 40),
  });
  const draw = () => {
    const { g, w, h } = prepare(cv, height);
    const t = theme();
    const { sx, sy } = toPx(w, h);
    if (axes) {
      g.strokeStyle = t.grid; g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const x = 30 + (i / 4) * (w - 42);
        const y = h - 26 - (i / 4) * (h - 40);
        g.beginPath(); g.moveTo(x, 14); g.lineTo(x, h - 26); g.stroke();
        g.beginPath(); g.moveTo(30, y); g.lineTo(w - 12, y); g.stroke();
      }
      if (xMin < 0 && xMax > 0) {
        g.strokeStyle = t.axis;
        g.beginPath(); g.moveTo(sx(0), 14); g.lineTo(sx(0), h - 26); g.stroke();
      }
      if (yMin < 0 && yMax > 0) {
        g.strokeStyle = t.axis;
        g.beginPath(); g.moveTo(30, sy(0)); g.lineTo(w - 12, sy(0)); g.stroke();
      }
    }
    if (links) {
      g.strokeStyle = 'rgba(148,163,184,.35)';
      g.lineWidth = 1;
      links.forEach(([a, bIdx]) => {
        const p = points[a]; const q = points[bIdx];
        if (!p || !q) return;
        g.beginPath(); g.moveTo(sx(p.x), sy(p.y)); g.lineTo(sx(q.x), sy(q.y)); g.stroke();
      });
    }
    g.font = '12px system-ui, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'bottom';
    points.forEach((p, i) => {
      const c = p.color || PALETTE[i % PALETTE.length];
      g.fillStyle = c;
      g.beginPath(); g.arc(sx(p.x), sy(p.y), p.r || (i === hot ? 7 : 5), 0, 7); g.fill();
      if (p.label) {
        g.fillStyle = p.labelColor || theme().fg;
        g.fillText(p.label, sx(p.x), sy(p.y) - 8);
      }
      if (p.arrow) {
        g.strokeStyle = c; g.lineWidth = 2;
        g.beginPath(); g.moveTo(sx(0), sy(0)); g.lineTo(sx(p.x), sy(p.y)); g.stroke();
      }
    });
  };
  cv.redraw = draw;

  if (onDrag) {
    cv.style.cursor = 'grab';
    const pick = (ev) => {
      const r = cv.getBoundingClientRect();
      const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
      const { sx, sy } = toPx(r.width, r.height);
      let best = -1; let bd = 1e9;
      points.forEach((p, i) => {
        const d = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
        if (d < bd) { bd = d; best = i; }
      });
      return { i: bd < 900 ? best : -1, px, py, r };
    };
    const move = (ev) => {
      if (hot < 0) return;
      ev.preventDefault();
      const r = cv.getBoundingClientRect();
      const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
      const x = xMin + ((px - 30) / (r.width - 42)) * (xMax - xMin);
      const y = yMin + ((r.height - 26 - py) / (r.height - 40)) * (yMax - yMin);
      onDrag(hot, clamp(x, xMin, xMax), clamp(y, yMin, yMax));
      draw();
    };
    const up = () => { hot = -1; cv.style.cursor = 'grab'; };
    cv.addEventListener('pointerdown', (ev) => {
      const { i } = pick(ev);
      hot = i;
      if (i >= 0) { cv.setPointerCapture(ev.pointerId); cv.style.cursor = 'grabbing'; }
    });
    cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
  }
  requestAnimationFrame(draw);
  return cv;
}

/* ─────────────────────────────── 히트맵 ────────────────────────────────── */
/** matrix: number[][], rowLabels / colLabels */
export function heatmap({ matrix, rowLabels = [], colLabels = [], cellMax, height, fmt, colorOf }) {
  const rows = matrix.length;
  const colsN = matrix[0]?.length || 0;
  const H = height || Math.min(420, 40 + rows * 34);
  const cv = makeCanvas(H);
  const draw = () => {
    const { g, w, h } = prepare(cv, H);
    const t = theme();
    g.font = '11px system-ui, sans-serif';
    const padL = Math.min(110, Math.max(30, ...rowLabels.map((s) => g.measureText(s).width + 10)));
    const padT = colLabels.length ? 24 : 6;
    const cw = (w - padL - 6) / Math.max(1, colsN);
    const ch = (h - padT - 6) / Math.max(1, rows);
    const top = cellMax !== undefined ? cellMax : Math.max(1e-9, ...matrix.flat());
    g.textBaseline = 'middle';
    colLabels.forEach((s, j) => {
      g.fillStyle = t.dim; g.textAlign = 'center';
      g.fillText(clip(g, s, cw - 2), padL + cw * (j + 0.5), padT / 2);
    });
    matrix.forEach((row, i) => {
      g.fillStyle = t.dim; g.textAlign = 'right';
      if (rowLabels[i]) g.fillText(rowLabels[i], padL - 6, padT + ch * (i + 0.5));
      row.forEach((v, j) => {
        const a = top > 0 ? Math.max(0, Math.min(1, v / top)) : 0;
        g.fillStyle = colorOf ? colorOf(v, a) : `rgba(96,165,250,${(a * 0.9).toFixed(3)})`;
        g.fillRect(padL + cw * j + 1, padT + ch * i + 1, cw - 2, ch - 2);
        g.fillStyle = a > 0.55 ? '#0b1020' : t.fg;
        g.textAlign = 'center';
        g.fillText(fmt ? fmt(v) : v.toFixed(2), padL + cw * (j + 0.5), padT + ch * (i + 0.5));
      });
    });
  };
  cv.redraw = draw;
  requestAnimationFrame(draw);
  return cv;
}

/* ─────────────────────────── 함수 곡선 그리기 ──────────────────────────── */
export function curve({ height = 180, fns, xMin = -6, xMax = 6, yMin, yMax, marker }) {
  const series = fns.map((f) => {
    const pts = [];
    for (let i = 0; i <= 200; i++) {
      const x = xMin + ((xMax - xMin) * i) / 200;
      pts.push([x, f.fn(x)]);
    }
    return { name: f.name, points: pts, color: f.color };
  });
  if (marker) series.push({ name: '', points: [[marker.x, marker.y]], color: marker.color || '#fbbf24', dots: true, width: 0 });
  return lineChart({ height, series, yMin, yMax, xMin, xMax, legend: fns.length > 1 });
}

/* ────────────────────────────── 잔손질 ────────────────────────────────── */
function roundRect(g, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + rr, y);
  g.arcTo(x + w, y, x + w, y + h, rr);
  g.arcTo(x + w, y + h, x, y + h, rr);
  g.arcTo(x, y + h, x, y, rr);
  g.arcTo(x, y, x + w, y, rr);
  g.closePath();
}

function clip(g, s, maxW) {
  if (g.measureText(s).width <= maxW) return s;
  let out = s;
  while (out.length > 1 && g.measureText(out + '…').width > maxW) out = out.slice(0, -1);
  return out + '…';
}

function fmtNum(v) {
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
  return v.toFixed(2);
}

const clamp = (v, a, z) => Math.max(a, Math.min(z, v));
