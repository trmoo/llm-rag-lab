/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * nlp.js — 텍스트를 숫자로 바꾸는 계산기들
 *
 * 여기 있는 것은 전부 자바스크립트로 직접 구현했다. 외부 라이브러리가 없다.
 * 그래야 교실 인터넷이 끊겨도, 학생이 값을 고쳐도 그 자리에서 다시 계산된다.
 *
 *   정제·정규화 → 토큰화 → 불용어      (전처리)
 *   BoW → TF → IDF → TF-IDF            (수치화)
 *   코사인 유사도 · BM25 · MMR          (검색)
 *   나이브 베이즈 · 로지스틱 회귀        (분류)
 * ========================================================================== */

import { nouns, morphs } from './korean.js';

/* ══════════════════════════ 1. 전처리 ═══════════════════════════════════ */

/** 자주 쓰는 정제 규칙 — 화면에서 하나씩 껐다 켤 수 있게 조각으로 둔다. */
export const CLEAN_RULES = [
  { key: 'url', label: '주소(URL) 지우기', re: /https?:\S+|www\.\S+/g, to: ' ',
    why: 'http 로 시작해 공백이 아닌 글자가 이어지는 덩어리를 지운다.' },
  { key: 'html', label: 'HTML 표시 지우기', re: /<[^>]+>/g, to: ' ',
    why: '웹에서 긁어 온 글에는 <b> 같은 표시가 섞여 있다.' },
  { key: 'emoji', label: '이모지 지우기', re: /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}]/gu, to: ' ',
    why: '그림 문자는 낱말 하나로 잘못 세어질 수 있다.' },
  { key: 'special', label: '특수문자 지우기', re: /[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]/g, to: ' ',
    why: '한글·영문·숫자·공백이 아닌 것을 모두 공백으로 바꾼다.' },
  { key: 'jamo', label: '자음·모음만 남은 것 지우기', re: /[ㄱ-ㅎㅏ-ㅣ]+/g, to: ' ',
    why: 'ㅋㅋ, ㅠㅠ 처럼 글자를 이루지 못한 조각을 지운다.' },
  { key: 'space', label: '연속 공백 정리', re: /\s+/g, to: ' ',
    why: '공백이 여러 개면 하나로 줄이고 앞뒤를 다듬는다.' },
];

/** 규칙 몇 가지를 골라 적용한다. steps 에 단계별 결과가 쌓인다. */
export function clean(text, on = {}) {
  const steps = [];
  let s = text;
  for (const r of CLEAN_RULES) {
    if (on[r.key] === false) continue;
    const before = s;
    s = s.replace(r.re, r.to);
    if (r.key === 'space') s = s.trim();
    steps.push({ key: r.key, label: r.label, before, after: s, changed: before !== s });
  }
  return { text: s, steps };
}

/** 같은 글자가 3번 넘게 이어지면 2번으로 줄인다 (ㅋㅋㅋㅋ → ㅋㅋ) */
export function squeezeRepeat(text) {
  return text.replace(/(.)\1{2,}/gu, '$1$1');
}

/** 공백으로만 쪼개기 — 가장 단순한 토큰화 */
export const splitTokens = (text) => text.split(/\s+/).filter(Boolean);

/** 수업에서 쓰는 한국어 불용어 (필요하면 화면에서 더하고 뺄 수 있다) */
export const STOPWORDS = [
  '것', '수', '나', '저', '제', '그', '이', '때', '등', '좀', '잘', '더', '한', '안', '못', '또',
  '거', '점', '중', '내', '더', '위', '전', '후', '및', '지', '데', '뭐', '왜',
];

/**
 * 이 앱의 기본 전처리 통로.
 * 원문 → 정제 → 반복 문자 줄이기 → 명사 뽑기 → 불용어·한 글자 버리기
 */
export function preprocess(text, opt = {}) {
  const stop = new Set(opt.stopwords || STOPWORDS);
  const minLen = opt.minLen ?? 2;
  const c = clean(text, opt.rules);
  const squeezed = squeezeRepeat(c.text);
  const words = opt.mode === 'morphs' ? morphs(squeezed, { stem: true }) : nouns(squeezed);
  return words.filter((w) => !stop.has(w) && w.length >= minLen);
}

/* ══════════════════════════ 2. 세기와 무게 ══════════════════════════════ */

/** 단어 → 횟수 */
export function counter(tokens) {
  const m = new Map();
  for (const t of tokens) m.set(t, (m.get(t) || 0) + 1);
  return m;
}

/** 여러 문서(단어 리스트들)에서 어휘 목록을 만든다 (가나다순) */
export function vocabulary(docs) {
  const s = new Set();
  docs.forEach((d) => d.forEach((w) => s.add(w)));
  return [...s].sort((a, b) => a.localeCompare(b, 'ko'));
}

/** 문서-단어 행렬 (BoW) — 행이 문서, 열이 단어, 값은 횟수 */
export function bowMatrix(docs, vocab) {
  const V = vocab || vocabulary(docs);
  const idx = new Map(V.map((w, i) => [w, i]));
  const M = docs.map((d) => {
    const row = new Array(V.length).fill(0);
    d.forEach((w) => { const i = idx.get(w); if (i !== undefined) row[i] += 1; });
    return row;
  });
  return { vocab: V, matrix: M };
}

/** TF — 문서 안에서의 비율 */
export function tf(term, doc) {
  if (!doc.length) return 0;
  let n = 0;
  for (const w of doc) if (w === term) n += 1;
  return n / doc.length;
}

/** 단어가 등장한 문서 수 */
export function docFreq(term, docs) {
  let n = 0;
  for (const d of docs) if (d.includes(term)) n += 1;
  return n;
}

/**
 * IDF — 흔한 낱말의 값을 깎는다.
 *  mode 'plain'   : ln(N / (df+1))          ← 손으로 계산할 때 쓰는 단순한 꼴
 *  mode 'sklearn' : ln((N+1)/(df+1)) + 1    ← 실제 도구가 쓰는 꼴 (0 이 되지 않게 +1)
 */
export function idf(term, docs, mode = 'plain') {
  const N = docs.length;
  const df = docFreq(term, docs);
  if (mode === 'sklearn') return Math.log((N + 1) / (df + 1)) + 1;
  return Math.log(N / (df + 1));
}

/**
 * TF-IDF 행렬.
 * normalize 를 켜면 문서마다 벡터 길이를 1로 맞춘다 (L2 정규화).
 * 실제 도구의 기본 설정은 mode:'sklearn' + normalize:true 이다.
 */
export function tfidfMatrix(docs, opt = {}) {
  const mode = opt.mode || 'plain';
  const normalize = opt.normalize ?? (mode === 'sklearn');
  const vocab = opt.vocab || vocabulary(docs);
  const idfs = vocab.map((w) => idf(w, docs, mode));
  const matrix = docs.map((d) => {
    const row = vocab.map((w, j) => tf(w, d) * idfs[j]);
    if (normalize) {
      const n = Math.hypot(...row);
      if (n > 0) for (let j = 0; j < row.length; j++) row[j] /= n;
    }
    return row;
  });
  return { vocab, idfs, matrix, mode, normalize };
}

/** 한 문서의 상위 낱말 [{word, score}] */
export function topTerms(row, vocab, k = 5) {
  return vocab
    .map((w, j) => ({ word: w, score: row[j] }))
    .filter((x) => x.score > 1e-12)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/* ══════════════════════════ 3. 닮은 정도 ═══════════════════════════════ */

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const norm = (a) => Math.sqrt(dot(a, a));

/** 코사인 유사도 — 벡터의 길이는 무시하고 방향만 본다 */
export function cosine(a, b) {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

/** 유클리드 거리 — 두 점 사이의 직선 거리 */
export function euclid(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

/** 모든 문서 쌍의 코사인 유사도 (대칭 행렬) */
export function similarityMatrix(rows) {
  return rows.map((a) => rows.map((b) => cosine(a, b)));
}

/* ══════════════════════════ 4. 낱말로 찾기 (BM25) ══════════════════════ */
/**
 * 키워드 검색의 표준 점수. 정확한 낱말·고유명사에 강하다.
 *   score = Σ IDF(q) · f·(k1+1) / (f + k1·(1 − b + b·|D|/avgdl))
 */
export function bm25Search(query, docs, { k1 = 1.5, b = 0.75, topK = 3 } = {}) {
  const N = docs.length;
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / Math.max(1, N);
  const scores = docs.map((d, i) => {
    const c = counter(d);
    let s = 0;
    for (const q of new Set(query)) {
      const f = c.get(q) || 0;
      if (!f) continue;
      const df = docFreq(q, docs);
      const w = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      s += w * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.length) / avgdl)));
    }
    return { i, score: s };
  });
  return scores.sort((x, y) => y.score - x.score).slice(0, topK);
}

/* ══════════════════════════ 5. 겹치지 않게 고르기 (MMR) ═══════════════ */
/**
 * 관련은 있으면서 서로 다른 것을 고른다.
 *   MMR = λ·(질문과의 유사도) − (1−λ)·(이미 고른 것과의 최대 유사도)
 * @param {number[]} sim   후보마다 질문과의 유사도
 * @param {number[][]} pair 후보끼리의 유사도 행렬
 */
export function mmrSelect(sim, pair, { k = 3, lambda = 0.5, fetchK } = {}) {
  const pool = sim
    .map((s, i) => ({ i, s }))
    .sort((a, b) => b.s - a.s)
    .slice(0, fetchK || sim.length)
    .map((x) => x.i);
  const picked = [];
  const trace = [];
  while (picked.length < k && pool.length) {
    let best = null;
    for (const c of pool) {
      const redundancy = picked.length ? Math.max(...picked.map((p) => pair[c][p])) : 0;
      const score = lambda * sim[c] - (1 - lambda) * redundancy;
      if (!best || score > best.score) best = { i: c, score, redundancy };
    }
    picked.push(best.i);
    trace.push(best);
    pool.splice(pool.indexOf(best.i), 1);
  }
  return { picked, trace };
}

/* ══════════════════════════ 6. 글자 n-gram ════════════════════════════ */
/**
 * 한국어는 「맛있다 / 맛있어요 / 맛있었어요」 처럼 꼴이 자주 바뀐다.
 * 글자 2~4개 묶음으로 쪼개면 공통 조각 「맛있」이 늘 남아, 형태소 분석기 없이도 통한다.
 */
export function charNgrams(text, min = 2, max = 4) {
  const words = text.split(/\s+/).filter(Boolean);
  const out = [];
  for (const w of words) {
    for (let n = min; n <= max; n++) {
      for (let i = 0; i + n <= w.length; i++) out.push(w.slice(i, i + n));
    }
  }
  return out;
}

/* ══════════════════════════ 7. 분류기 ═════════════════════════════════ */

/** 재현 가능한 난수 (같은 씨앗이면 늘 같은 결과) */
export function rng(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/** 훈련·시험 나누기. 섞은 뒤 뒤쪽 testSize 비율을 시험으로 뺀다. */
export function trainTestSplit(X, y, { testSize = 0.25, seed = 42 } = {}) {
  const idx = X.map((_, i) => i);
  const rand = rng(seed);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.max(1, Math.round(X.length * testSize));
  const test = idx.slice(0, nTest);
  const train = idx.slice(nTest);
  return {
    trainIdx: train, testIdx: test,
    Xtrain: train.map((i) => X[i]), ytrain: train.map((i) => y[i]),
    Xtest: test.map((i) => X[i]), ytest: test.map((i) => y[i]),
  };
}

/** 글자 n-gram 을 세어 특성으로 만드는 변환기 (fit / transform 을 나눠 쓴다) */
export function makeVectorizer({ min = 2, max = 4, minDf = 1 } = {}) {
  let vocab = [];
  let index = new Map();
  let idfs = [];
  return {
    /** 훈련 자료로만 fit 한다 — 시험 자료로 fit 하면 데이터 누수다. */
    fit(texts) {
      const dfs = new Map();
      const grams = texts.map((t) => new Set(charNgrams(t, min, max)));
      grams.forEach((g) => g.forEach((x) => dfs.set(x, (dfs.get(x) || 0) + 1)));
      vocab = [...dfs.entries()].filter(([, c]) => c >= minDf).map(([w]) => w).sort();
      index = new Map(vocab.map((w, i) => [w, i]));
      const N = texts.length;
      idfs = vocab.map((w) => Math.log((N + 1) / (dfs.get(w) + 1)) + 1);
      return this;
    },
    transform(texts) {
      return texts.map((t) => {
        const row = new Array(vocab.length).fill(0);
        const g = charNgrams(t, min, max);
        g.forEach((x) => { const i = index.get(x); if (i !== undefined) row[i] += 1; });
        for (let i = 0; i < row.length; i++) row[i] = (row[i] / Math.max(1, g.length)) * idfs[i];
        const n = Math.hypot(...row);
        if (n > 0) for (let i = 0; i < row.length; i++) row[i] /= n;
        return row;
      });
    },
    /** 횟수만 (나이브 베이즈용) */
    counts(texts) {
      return texts.map((t) => {
        const row = new Array(vocab.length).fill(0);
        charNgrams(t, min, max).forEach((x) => { const i = index.get(x); if (i !== undefined) row[i] += 1; });
        return row;
      });
    },
    get vocab() { return vocab; },
  };
}

/** 나이브 베이즈 — 낱말이 각 부류에서 나올 확률을 세어 곱한다 */
export function naiveBayes(X, y, { alpha = 1 } = {}) {
  const classes = [...new Set(y)].sort();
  const V = X[0]?.length || 0;
  const logPrior = {};
  const logProb = {};
  for (const c of classes) {
    const rows = X.filter((_, i) => y[i] === c);
    logPrior[c] = Math.log(rows.length / X.length);
    const sums = new Array(V).fill(0);
    rows.forEach((r) => { for (let j = 0; j < V; j++) sums[j] += r[j]; });
    const total = sums.reduce((a, b) => a + b, 0) + alpha * V;
    logProb[c] = sums.map((s) => Math.log((s + alpha) / total));
  }
  const scoreOne = (row) => {
    const out = {};
    for (const c of classes) {
      let s = logPrior[c];
      for (let j = 0; j < row.length; j++) if (row[j]) s += row[j] * logProb[c][j];
      out[c] = s;
    }
    return out;
  };
  return {
    classes,
    predict: (rows) => rows.map((r) => {
      const s = scoreOne(r);
      return classes.reduce((a, c) => (s[c] > s[a] ? c : a), classes[0]);
    }),
    predictProba: (rows) => rows.map((r) => {
      const s = scoreOne(r);
      const m = Math.max(...classes.map((c) => s[c]));
      const ex = classes.map((c) => Math.exp(s[c] - m));
      const sum = ex.reduce((a, x) => a + x, 0);
      return Object.fromEntries(classes.map((c, i) => [c, ex[i] / sum]));
    }),
    logProb,
  };
}

/** 로지스틱 회귀 — 낱말마다 가중치를 두고 점수를 더한다 (경사 하강법) */
export function logisticRegression(X, y, { lr = 0.6, epochs = 400, l2 = 0.002 } = {}) {
  const V = X[0]?.length || 0;
  const w = new Array(V).fill(0);
  let bias = 0;
  const n = X.length;
  const history = [];
  const sig = (z) => 1 / (1 + Math.exp(-z));
  for (let e = 0; e < epochs; e++) {
    const gw = new Array(V).fill(0);
    let gb = 0;
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let z = bias;
      const row = X[i];
      for (let j = 0; j < V; j++) if (row[j]) z += w[j] * row[j];
      const p = sig(z);
      const err = p - y[i];
      loss += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12));
      for (let j = 0; j < V; j++) if (row[j]) gw[j] += err * row[j];
      gb += err;
    }
    for (let j = 0; j < V; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    bias -= lr * (gb / n);
    if (e % 10 === 0 || e === epochs - 1) history.push({ epoch: e, loss: loss / n });
  }
  const proba = (rows) => rows.map((row) => {
    let z = bias;
    for (let j = 0; j < V; j++) if (row[j]) z += w[j] * row[j];
    return sig(z);
  });
  return {
    weights: w, bias, history,
    predictProba: proba,
    predict: (rows, th = 0.5) => proba(rows).map((p) => (p >= th ? 1 : 0)),
  };
}

/* ══════════════════════════ 8. 성능 재기 ══════════════════════════════ */

export function confusion(yTrue, yPred) {
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp += 1;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn += 1;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp += 1;
    else fn += 1;
  }
  return { tp, tn, fp, fn };
}

export function metrics(yTrue, yPred) {
  const { tp, tn, fp, fn } = confusion(yTrue, yPred);
  const acc = (tp + tn) / Math.max(1, yTrue.length);
  const prec = tp + fp ? tp / (tp + fp) : 0;
  const rec = tp + fn ? tp / (tp + fn) : 0;
  const f1 = prec + rec ? (2 * prec * rec) / (prec + rec) : 0;
  return { tp, tn, fp, fn, accuracy: acc, precision: prec, recall: rec, f1 };
}

/** 임계값을 바꿔 가며 정밀도·재현율이 어떻게 움직이는지 */
export function thresholdTable(yTrue, probs, steps = 9) {
  const out = [];
  for (let i = 1; i <= steps; i++) {
    const th = i / (steps + 1);
    const pred = probs.map((p) => (p >= th ? 1 : 0));
    out.push({ th, ...metrics(yTrue, pred) });
  }
  return out;
}
