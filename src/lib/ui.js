/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * ui.js — 화면을 만들 때 공통으로 쓰는 부품
 *
 * 여기서 가장 중요한 것은 「화면 수명 관리」다.
 * 탭을 옮기면 화면 요소는 지워지지만, 그 화면이 window 에 걸어 둔
 * resize 리스너와 setInterval 타이머는 살아남아 오갈 때마다 쌓인다.
 * 그래서 새 화면을 그리기 직전에 beginScreen() 으로 전부 걷어 낸다.
 *
 * ⚠️ 화면 모듈에서 window.addEventListener('resize', …) 나 setInterval 을
 *    직접 쓰지 말고 아래 onResize()·screenInterval() 을 쓸 것.
 * ========================================================================== */

/* ────────────────────────────── DOM 만들기 ──────────────────────────────── */

/**
 * h('div.card', { onclick }, '글자', 자식요소…)
 * 태그 이름 뒤에 .클래스 를 붙여 쓴다. 첫 인자가 순수 객체면 속성으로 본다.
 */
export function h(spec, ...rest) {
  const [tagRaw, ...classes] = String(spec).split('.');
  const tag = tagRaw || 'div';
  const el = document.createElement(tag);
  if (classes.length) el.className = classes.join(' ');

  let children = rest;
  if (rest.length && isPlainObject(rest[0])) {
    const attrs = rest[0];
    children = rest.slice(1);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = (el.className ? el.className + ' ' : '') + v;
      else if (k === 'style' && isPlainObject(v)) Object.assign(el.style, v);
      else if (k === 'dataset' && isPlainObject(v)) Object.assign(el.dataset, v);
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k in el && k !== 'list' && typeof v !== 'object') el[k] = v;
      else el.setAttribute(k, v);
    }
  }
  append(el, children);
  return el;
}

function isPlainObject(x) {
  return x !== null && typeof x === 'object' && !Array.isArray(x) && !(x instanceof Node);
}

function append(el, children) {
  for (const c of children.flat(4)) {
    if (c === null || c === undefined || c === false || c === true) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** 자식을 모두 지우고 새로 채운다. */
export function fill(el, ...children) {
  el.textContent = '';
  append(el, children);
  return el;
}

/** 고정폭 글자 (코드·숫자용) */
export const mono = (t) => h('code.mono', t);
/** 강조 */
export const b = (t) => h('b', t);

/* ─────────────────────────── 화면 수명 관리 ─────────────────────────────── */

let resizeHandlers = [];
let intervals = [];
let timeouts = [];
let rafs = [];
let keyHandlers = [];

/** 새 화면을 그리기 직전에 부른다. 이전 화면이 남긴 리스너·타이머를 모두 끊는다. */
export function beginScreen() {
  resizeHandlers.forEach((fn) => window.removeEventListener('resize', fn));
  keyHandlers.forEach((fn) => window.removeEventListener('keydown', fn));
  intervals.forEach(clearInterval);
  timeouts.forEach(clearTimeout);
  rafs.forEach(cancelAnimationFrame);
  resizeHandlers = [];
  keyHandlers = [];
  intervals = [];
  timeouts = [];
  rafs = [];
}

export function onResize(fn) {
  window.addEventListener('resize', fn);
  resizeHandlers.push(fn);
  return fn;
}

export function onKey(fn) {
  window.addEventListener('keydown', fn);
  keyHandlers.push(fn);
  return fn;
}

export function screenInterval(fn, ms) {
  const id = setInterval(fn, ms);
  intervals.push(id);
  return id;
}

export function screenTimeout(fn, ms) {
  const id = setTimeout(fn, ms);
  timeouts.push(id);
  return id;
}

export function screenRaf(fn) {
  const id = requestAnimationFrame(fn);
  rafs.push(id);
  return id;
}

/** 화면에서 쓰는 애니메이션 루프. stop() 을 돌려준다. */
export function loop(step) {
  let alive = true;
  let last = performance.now();
  const tick = (now) => {
    if (!alive) return;
    const dt = Math.min(100, now - last);
    last = now;
    step(dt, now);
    screenRaf(tick);
  };
  screenRaf(tick);
  return () => { alive = false; };
}

/* ───────────────────────────── 기본 조각들 ─────────────────────────────── */

/** 화면 맨 위 제목 + 한 줄 설명 */
export function screenHead(title, lead, badge) {
  return h('header.screen-head',
    h('div.screen-head-row',
      h('h2.screen-title', title),
      badge ? h('span.badge', badge) : null,
    ),
    lead ? h('p.screen-lead', lead) : null,
  );
}

/** 네모 상자 하나 */
export function card(title, ...children) {
  return h('section.card',
    title ? h('h3.card-title', title) : null,
    h('div.card-body', ...children),
  );
}

/** 눈에 띄는 안내 상자. kind: info | warn | good | bad */
export function note(kind, ...children) {
  const icon = { info: '💡', warn: '⚠️', good: '✅', bad: '❌', ask: '❓' }[kind] || '💡';
  return h(`div.note.note-${kind}`, h('span.note-icon', icon), h('div.note-body', ...children));
}

/** 두 칸 나란히 (좁은 화면에서는 자동으로 위아래) */
export function cols(...children) {
  return h('div.cols', ...children);
}

/** 접었다 펴는 심화 서술. 기본은 접힘. */
export function deepDive(title, ...children) {
  const body = h('div.deep-body', ...children);
  body.hidden = true;
  const arrow = h('span.deep-arrow', '▶');
  const btn = h('button.deep-head', {
    type: 'button',
    onclick: () => {
      body.hidden = !body.hidden;
      arrow.textContent = body.hidden ? '▶' : '▼';
      btn.classList.toggle('open', !body.hidden);
    },
  }, arrow, h('span.deep-tag', '더 깊이'), h('span.deep-title', title));
  return h('section.deep', btn, body);
}

/** 용어 한 줄 정의 목록 */
export function terms(pairs) {
  return h('dl.terms', ...pairs.flatMap(([t, d]) => [h('dt', t), h('dd', d)]));
}

/** 표. head 는 문자열 배열, rows 는 (문자열|Node)[][] */
export function table(head, rows, opts = {}) {
  const t = h('table.tbl' + (opts.compact ? '.compact' : ''),
    head ? h('thead', h('tr', ...head.map((x) => h('th', x)))) : null,
    h('tbody', ...rows.map((r, ri) => h('tr' + (opts.rowClass ? '.' + opts.rowClass(r, ri) : ''),
      ...r.map((c) => h('td', c))))),
  );
  return h('div.tbl-wrap', t);
}

/** 값이 바뀌면 알려 주는 슬라이더 */
export function slider({ label, min, max, step = 1, value, format, onInput }) {
  const out = h('output.slider-val', format ? format(value) : String(value));
  const input = h('input.slider', {
    type: 'range', min, max, step, value,
    oninput: (e) => {
      const v = Number(e.target.value);
      out.textContent = format ? format(v) : String(v);
      onInput(v);
    },
  });
  const row = h('label.slider-row',
    h('span.slider-label', label),
    input,
    out,
  );
  row.setValue = (v) => {
    input.value = String(v);
    out.textContent = format ? format(v) : String(v);
  };
  return row;
}

/** 여러 개 중 하나 고르는 알약 단추 묶음 (첫 항목이 기본 선택) */
export function pillGroup(items, onPick, initial) {
  const wrap = h('div.pills');
  let cur = initial !== undefined ? initial : items[0]?.value;
  const btns = items.map((it) => {
    const btn = h('button.pill', {
      type: 'button',
      onclick: () => {
        cur = it.value;
        btns.forEach((x) => x.classList.remove('on'));
        btn.classList.add('on');
        onPick(it.value);
      },
    }, it.label);
    if (it.value === cur) btn.classList.add('on');
    wrap.appendChild(btn);
    return btn;
  });
  return wrap;
}

/**
 * 채점이 붙는 「고르기」. pillGroup 과 달리 처음에는 아무것도 안 골라져 있다.
 * (안 고르고 [확인] 을 눌렀는데 골라진 것으로 처리되는 사고를 막는다)
 */
export function choicePicker(items, onPick) {
  const wrap = h('div.pills.choice');
  let picked = null;
  const btns = items.map((it) => {
    const btn = h('button.pill', {
      type: 'button',
      onclick: () => {
        picked = it.value;
        btns.forEach((x) => x.classList.remove('on'));
        btn.classList.add('on');
        if (onPick) onPick(it.value);
      },
    }, it.label);
    wrap.appendChild(btn);
    return btn;
  });
  wrap.getValue = () => picked;
  wrap.reset = () => { picked = null; btns.forEach((x) => x.classList.remove('on')); };
  return wrap;
}

/** 단추 */
export function button(label, onclick, kind = '') {
  return h('button.btn' + (kind ? '.btn-' + kind : ''), { type: 'button', onclick }, label);
}

/** 여러 줄 입력 상자 */
export function textarea({ value = '', rows = 4, onInput, placeholder = '' }) {
  const ta = h('textarea.ta', { rows, placeholder });
  ta.value = value;
  if (onInput) ta.addEventListener('input', () => onInput(ta.value));
  return ta;
}

/** 한 줄 입력 상자 */
export function input({ value = '', placeholder = '', onInput, onEnter, size }) {
  const el = h('input.inp', { type: 'text', placeholder });
  if (size) el.size = size;
  el.value = value;
  if (onInput) el.addEventListener('input', () => onInput(el.value));
  if (onEnter) el.addEventListener('keydown', (e) => { if (e.key === 'Enter') onEnter(el.value); });
  return el;
}

/** 껐다 켜는 스위치 */
export function toggle(label, value, onChange) {
  const box = h('input', { type: 'checkbox' });
  box.checked = !!value;
  box.addEventListener('change', () => onChange(box.checked));
  const row = h('label.toggle', box, h('span.toggle-track', h('span.toggle-knob')), h('span.toggle-label', label));
  row.setValue = (v) => { box.checked = !!v; };
  row.getValue = () => box.checked;
  return row;
}

/** 0~1 값을 막대로 */
export function bar(value, opts = {}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return h('div.bar' + (opts.small ? '.bar-sm' : ''),
    h('div.bar-fill', { style: { width: pct.toFixed(1) + '%', background: opts.color || '' } }),
    opts.label !== undefined ? h('span.bar-label', opts.label) : null,
  );
}

/** 코드 블록 (파이썬) */
export function code(text, opts = {}) {
  const pre = h('pre.code' + (opts.small ? '.code-sm' : ''), h('code', text));
  if (opts.copy !== false) {
    const btn = h('button.code-copy', {
      type: 'button',
      onclick: async () => {
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = '복사됨';
          setTimeout(() => { btn.textContent = '복사'; }, 1200);
        } catch { btn.textContent = '복사 실패'; }
      },
    }, '복사');
    pre.appendChild(btn);
  }
  return pre;
}

/** 「파이썬으로는 이렇게 씁니다」 상자 */
export function pyBox(text, caption) {
  return h('section.pybox',
    h('div.pybox-head', h('span.pybox-tag', '🐍 파이썬으로는'), caption ? h('span.pybox-cap', caption) : null),
    code(text),
  );
}

/** 화면 하단의 「다음 화면」 안내 */
export function nextHint(text, go) {
  return h('div.next-hint',
    h('span', '다음 → '),
    h('button.link', { type: 'button', onclick: go }, text),
  );
}

/* ────────────────────────────── 숫자 다루기 ─────────────────────────────── */

export const fx = (v, n = 3) => (Number.isFinite(v) ? v.toFixed(n) : '—');
export const pct = (v, n = 1) => (Number.isFinite(v) ? (v * 100).toFixed(n) + '%' : '—');

/** 값 크기에 따라 배경색을 입힌 셀 (히트맵용) */
export function heatCell(v, max, text) {
  const t = max > 0 ? Math.max(0, Math.min(1, v / max)) : 0;
  const el = h('span.heat', text !== undefined ? text : fx(v));
  el.style.background = `rgba(96,165,250,${(t * 0.85).toFixed(3)})`;
  if (t > 0.55) el.style.color = '#0b1020';
  return el;
}
