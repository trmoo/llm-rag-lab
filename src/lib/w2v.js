/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * w2v.js — 단어 임베딩을 브라우저에서 실제로 학습시킨다
 *
 * 여기 있는 것은 흉내가 아니라 진짜 학습이다.
 * 「비슷한 자리에 함께 나오는 낱말은 비슷한 뜻이다」(분포 가설)를 그대로 옮겨,
 * 중심 낱말로 주변 낱말을 맞히는 문제를 풀며 벡터를 조금씩 고쳐 나간다.
 *   (Skip-gram + 네거티브 샘플링)
 *
 * 문장 서른 개쯤이면 브라우저에서 1초 안에 끝난다. 그래서 학생이
 * 「문장을 더 넣으면 유사도가 어떻게 달라지나」를 직접 실험할 수 있다.
 * ========================================================================== */

import { rng } from './nlp.js';

/**
 * @param {string[][]} sentences 낱말 리스트들의 배열
 * @param {object} opt vectorSize·window·minCount·epochs·negative·lr·seed
 */
export function trainWord2Vec(sentences, opt = {}) {
  const dim = opt.vectorSize ?? 24;
  const win = opt.window ?? 3;
  const minCount = opt.minCount ?? 1;
  const epochs = opt.epochs ?? 120;
  const neg = opt.negative ?? 5;
  const lr0 = opt.lr ?? 0.05;
  const rand = rng(opt.seed ?? 42);

  /* ① 어휘 만들기 */
  const freq = new Map();
  sentences.forEach((s) => s.forEach((w) => freq.set(w, (freq.get(w) || 0) + 1)));
  const vocab = [...freq.entries()].filter(([, c]) => c >= minCount).map(([w]) => w);
  vocab.sort((a, b) => freq.get(b) - freq.get(a) || a.localeCompare(b, 'ko'));
  const idx = new Map(vocab.map((w, i) => [w, i]));
  const V = vocab.length;
  if (!V) return emptyModel();

  /* ② 네거티브 샘플링 표 — 자주 나오는 낱말이 조금 더 자주 뽑히게 (빈도^0.75) */
  const weights = vocab.map((w) => Math.pow(freq.get(w), 0.75));
  const total = weights.reduce((a, b) => a + b, 0);
  const cum = [];
  let acc = 0;
  for (const w of weights) { acc += w / total; cum.push(acc); }
  const sampleNeg = () => {
    const r = rand();
    let lo = 0; let hi = cum.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; }
    return lo;
  };

  /* ③ 벡터 두 벌 — 중심 낱말용(입력)과 주변 낱말용(출력) */
  const W = new Float64Array(V * dim);
  const C = new Float64Array(V * dim);
  for (let i = 0; i < V * dim; i++) W[i] = (rand() - 0.5) / dim;

  /* ④ 학습할 (중심, 주변) 짝을 미리 뽑아 둔다 */
  const pairs = [];
  for (const s of sentences) {
    const ids = s.map((w) => idx.get(w)).filter((i) => i !== undefined);
    for (let i = 0; i < ids.length; i++) {
      const lo = Math.max(0, i - win);
      const hi = Math.min(ids.length - 1, i + win);
      for (let j = lo; j <= hi; j++) if (j !== i) pairs.push([ids[i], ids[j]]);
    }
  }
  if (!pairs.length) return emptyModel();

  const sig = (z) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  const history = [];

  /* ⑤ 학습 — 맞으면 가까이, 아니면 멀리 */
  for (let e = 0; e < epochs; e++) {
    const lr = lr0 * (1 - e / (epochs + 1)) + 0.0005;
    let loss = 0;
    // 짝의 순서를 매 회 섞는다
    for (let i = pairs.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
    }
    for (const [center, ctx] of pairs) {
      const co = center * dim;
      const grad = new Float64Array(dim);
      for (let k = 0; k <= neg; k++) {
        const target = k === 0 ? ctx : sampleNeg();
        if (k > 0 && target === ctx) continue;
        const label = k === 0 ? 1 : 0;
        const to = target * dim;
        let z = 0;
        for (let d = 0; d < dim; d++) z += W[co + d] * C[to + d];
        const p = sig(z);
        loss += -(label ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
        const g = (label - p) * lr;
        for (let d = 0; d < dim; d++) {
          grad[d] += g * C[to + d];
          C[to + d] += g * W[co + d];
        }
      }
      for (let d = 0; d < dim; d++) W[co + d] += grad[d];
    }
    if (e % Math.max(1, Math.floor(epochs / 20)) === 0 || e === epochs - 1) {
      history.push({ epoch: e, loss: loss / pairs.length });
    }
  }

  /* ⑥ 쓰기 좋은 모양으로 감싸기 */
  const vec = (w) => {
    const i = idx.get(w);
    if (i === undefined) return null;
    return Array.from(W.subarray(i * dim, i * dim + dim));
  };
  const unit = new Map();
  for (const w of vocab) {
    const v = vec(w);
    const n = Math.hypot(...v) || 1;
    unit.set(w, v.map((x) => x / n));
  }
  const sim = (a, b) => {
    const va = unit.get(a);
    const vb = unit.get(b);
    if (!va || !vb) return null;
    let s = 0;
    for (let d = 0; d < dim; d++) s += va[d] * vb[d];
    return s;
  };

  return {
    dim, vocab, history, freq,
    has: (w) => idx.has(w),
    get: vec,
    getUnit: (w) => unit.get(w) || null,
    similarity: sim,
    /** 가장 비슷한 낱말 위에서부터 */
    mostSimilar(word, topn = 5) {
      if (!idx.has(word)) return [];
      return vocab
        .filter((w) => w !== word)
        .map((w) => ({ word: w, sim: sim(word, w) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, topn);
    },
    /** 「가 − 나 + 다 = ?」 낱말 셈 */
    analogy(plusA, minusB, plusC, topn = 3) {
      const a = unit.get(plusA); const b = unit.get(minusB); const c = unit.get(plusC);
      if (!a || !b || !c) return [];
      const t = a.map((x, i) => x - b[i] + c[i]);
      const n = Math.hypot(...t) || 1;
      const tn = t.map((x) => x / n);
      const skip = new Set([plusA, minusB, plusC]);
      return vocab
        .filter((w) => !skip.has(w))
        .map((w) => {
          const v = unit.get(w);
          let s = 0;
          for (let d = 0; d < dim; d++) s += tn[d] * v[d];
          return { word: w, sim: s };
        })
        .sort((x, y) => y.sim - x.sim)
        .slice(0, topn);
    },
    /** 문서 벡터 — 낱말 벡터의 평균 */
    docVector(words) {
      const acc = new Array(dim).fill(0);
      let n = 0;
      for (const w of words) {
        const v = unit.get(w);
        if (!v) continue;
        for (let d = 0; d < dim; d++) acc[d] += v[d];
        n += 1;
      }
      if (!n) return acc;
      return acc.map((x) => x / n);
    },
  };
}

function emptyModel() {
  return {
    dim: 0, vocab: [], history: [], freq: new Map(),
    has: () => false, get: () => null, getUnit: () => null, similarity: () => null,
    mostSimilar: () => [], analogy: () => [], docVector: () => [],
  };
}
