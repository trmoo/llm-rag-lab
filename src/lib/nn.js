/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * nn.js — 신경망 계산기
 *
 *   활성화 함수 · 손실 함수 · 경사 하강법
 *   RNN / LSTM 를 한 걸음씩 (게이트 값까지 꺼내 볼 수 있게)
 *   Self-Attention (Q·K·V → 점수 → softmax → 가중 합)
 *   위치 인코딩 · 인과 마스크
 *
 * 화면에서 "값이 어떻게 변하는지"를 보여 주는 것이 목적이라,
 * 중간 계산 결과를 전부 돌려준다.
 * ========================================================================== */

import { rng } from './nlp.js';

/* ════════════════════════════ 활성화·손실 ═════════════════════════════ */

export const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))));
export const relu = (z) => Math.max(0, z);
export const leakyRelu = (z) => (z >= 0 ? z : 0.01 * z);
export const tanh = (z) => Math.tanh(z);

export const ACTIVATIONS = {
  Sigmoid: { fn: sigmoid, range: '0 ~ 1', use: '출력층 — 확률로 읽는다', d: (z) => sigmoid(z) * (1 - sigmoid(z)) },
  ReLU: { fn: relu, range: '0 ~ ∞', use: '은닉층 — 가장 많이 쓴다', d: (z) => (z > 0 ? 1 : 0) },
  Tanh: { fn: tanh, range: '−1 ~ 1', use: 'RNN·LSTM 안쪽', d: (z) => 1 - Math.tanh(z) ** 2 },
  LeakyReLU: { fn: leakyRelu, range: '−∞ ~ ∞', use: 'ReLU 가 죽는 것을 막을 때', d: (z) => (z > 0 ? 1 : 0.01) },
};

/** 여러 점수를 합이 1인 확률로 */
export function softmax(arr, temperature = 1) {
  const t = Math.max(1e-6, temperature);
  const z = arr.map((x) => x / t);
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / s);
}

/** 이진 교차 엔트로피 — 정답에 가까울수록 작다 */
export const bce = (p, y) => -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12));

/* ════════════════════════════ 아주 작은 신경망 ════════════════════════ */
/**
 * 입력 → 은닉(ReLU) → 출력(Sigmoid) 2층 신경망을 진짜로 학습시킨다.
 * 순전파 → 손실 → 역전파 → 가중치 갱신 이 네 걸음이 그대로 들어 있다.
 */
export function trainMLP(X, y, { hidden = 8, epochs = 300, lr = 0.35, seed = 7 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const rand = rng(seed);
  const W1 = Array.from({ length: hidden }, () => Array.from({ length: d }, () => (rand() - 0.5) * 0.8));
  const b1 = new Array(hidden).fill(0);
  const W2 = Array.from({ length: hidden }, () => (rand() - 0.5) * 0.8);
  let b2 = 0;
  const history = [];

  const forward = (x) => {
    const z1 = W1.map((row, h) => row.reduce((s, w, i) => s + w * x[i], b1[h]));
    const a1 = z1.map(relu);
    const z2 = a1.reduce((s, a, h) => s + a * W2[h], b2);
    return { z1, a1, z2, p: sigmoid(z2) };
  };

  for (let e = 0; e < epochs; e++) {
    let loss = 0;
    const gW1 = W1.map((r) => r.map(() => 0));
    const gb1 = new Array(hidden).fill(0);
    const gW2 = new Array(hidden).fill(0);
    let gb2 = 0;
    for (let i = 0; i < n; i++) {
      const { a1, z1, p } = forward(X[i]);
      loss += bce(p, y[i]);
      const dz2 = p - y[i];                       // 출력층 오차
      for (let h = 0; h < hidden; h++) {
        gW2[h] += dz2 * a1[h];
        const da1 = dz2 * W2[h];
        const dz1 = da1 * (z1[h] > 0 ? 1 : 0);    // ReLU 를 거슬러 올라간다
        gb1[h] += dz1;
        for (let j = 0; j < d; j++) gW1[h][j] += dz1 * X[i][j];
      }
      gb2 += dz2;
    }
    for (let h = 0; h < hidden; h++) {
      W2[h] -= (lr * gW2[h]) / n;
      b1[h] -= (lr * gb1[h]) / n;
      for (let j = 0; j < d; j++) W1[h][j] -= (lr * gW1[h][j]) / n;
    }
    b2 -= (lr * gb2) / n;
    if (e % 5 === 0 || e === epochs - 1) {
      const acc = X.filter((x, i) => (forward(x).p >= 0.5 ? 1 : 0) === y[i]).length / n;
      history.push({ epoch: e, loss: loss / n, acc });
    }
  }
  return { forward, predict: (x) => forward(x).p, history, W1, W2, b1, b2 };
}

/** 경사 하강법 한 걸음 — 화면에서 「공이 굴러 내려가는」 그림에 쓴다 */
export function gradientSteps(fn, dfn, x0, lr, steps) {
  const out = [{ x: x0, y: fn(x0) }];
  let x = x0;
  for (let i = 0; i < steps; i++) {
    x -= lr * dfn(x);
    out.push({ x, y: fn(x) });
  }
  return out;
}

/* ════════════════════════════ 순환 신경망 ═════════════════════════════ */
/** 재현 가능한 작은 가중치 행렬 */
export function randMatrix(rows, cols, seed = 1, scale = 0.9) {
  const rand = rng(seed);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rand() - 0.5) * 2 * scale));
}

const matVec = (M, v) => M.map((row) => row.reduce((s, w, i) => s + w * v[i], 0));
const addVec = (a, b) => a.map((x, i) => x + b[i]);
const mulVec = (a, b) => a.map((x, i) => x * b[i]);

/**
 * RNN 순전파. h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)
 * 각 시점의 h 를 모두 돌려주어 「기억이 어떻게 바뀌는지」 보여 준다.
 */
export function rnnForward(inputs, { hiddenDim = 4, seed = 3 } = {}) {
  const inDim = inputs[0].length;
  const Wxh = randMatrix(hiddenDim, inDim, seed, 0.8);
  const Whh = randMatrix(hiddenDim, hiddenDim, seed + 1, 0.6);
  const bias = new Array(hiddenDim).fill(0);
  let h = new Array(hiddenDim).fill(0);
  const steps = [];
  inputs.forEach((x, t) => {
    const fromX = matVec(Wxh, x);
    const fromH = matVec(Whh, h);
    const pre = addVec(addVec(fromX, fromH), bias);
    const hNew = pre.map(tanh);
    steps.push({ t, x, fromX, fromH, pre, h: hNew, hPrev: h });
    h = hNew;
  });
  return { steps, hFinal: h, Wxh, Whh };
}

/**
 * LSTM 순전파. 게이트 세 개와 셀 상태를 그대로 꺼내 준다.
 *   f = σ(...)  잊을 비율 · i = σ(...)  받아들일 비율 · o = σ(...)  내보낼 비율
 *   C_t = f⊙C_{t−1} + i⊙C̃_t
 *   h_t = o⊙tanh(C_t)
 */
export function lstmForward(inputs, { hiddenDim = 4, seed = 11 } = {}) {
  const inDim = inputs[0].length;
  const mk = (s) => ({ Wx: randMatrix(hiddenDim, inDim, s, 0.9), Wh: randMatrix(hiddenDim, hiddenDim, s + 100, 0.5) });
  const F = mk(seed);
  const I = mk(seed + 1);
  const O = mk(seed + 2);
  const G = mk(seed + 3);
  let h = new Array(hiddenDim).fill(0);
  let c = new Array(hiddenDim).fill(0);
  const steps = [];
  const gate = (P, x, hp, act) => addVec(matVec(P.Wx, x), matVec(P.Wh, hp)).map(act);
  inputs.forEach((x, t) => {
    const f = gate(F, x, h, sigmoid);
    const i = gate(I, x, h, sigmoid);
    const o = gate(O, x, h, sigmoid);
    const g = gate(G, x, h, tanh);
    const cNew = addVec(mulVec(f, c), mulVec(i, g));
    const hNew = mulVec(o, cNew.map(tanh));
    steps.push({ t, x, f, i, o, g, c: cNew, h: hNew, cPrev: c, hPrev: h });
    c = cNew;
    h = hNew;
  });
  return { steps, hFinal: h, cFinal: c };
}

/** 기울기가 층을 지날수록 어떻게 되는지 (0.9 를 계속 곱하면?) */
export function gradientDecay(factor, steps) {
  const out = [];
  let g = 1;
  for (let i = 0; i <= steps; i++) { out.push({ step: i, g }); g *= factor; }
  return out;
}

/* ════════════════════════════ Self-Attention ══════════════════════════ */
/**
 * Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k)·V
 * 중간값(Q·Kᵀ, 나눈 값, softmax 결과)을 전부 돌려준다.
 * @param {number[][]} X 낱말마다 한 줄인 입력 행렬 (토큰 수 × 임베딩 차원)
 */
export function selfAttention(X, { dk = 4, seed = 5, mask = null, Wq, Wk, Wv } = {}) {
  const dModel = X[0].length;
  const WQ = Wq || randMatrix(dModel, dk, seed, 0.7);
  const WK = Wk || randMatrix(dModel, dk, seed + 1, 0.7);
  const WV = Wv || randMatrix(dModel, dk, seed + 2, 0.7);
  const proj = (M) => X.map((row) => M[0].map((_, j) => row.reduce((s, x, i) => s + x * M[i][j], 0)));
  const Q = proj(WQ);
  const K = proj(WK);
  const V = proj(WV);
  const raw = Q.map((q) => K.map((k) => q.reduce((s, x, i) => s + x * k[i], 0)));
  const scaled = raw.map((r) => r.map((v) => v / Math.sqrt(dk)));
  const masked = scaled.map((r, i) => r.map((v, j) => (mask && !mask[i][j] ? -Infinity : v)));
  const weights = masked.map((r) => softmax(r));
  const out = weights.map((w) => V[0].map((_, d) => w.reduce((s, wi, j) => s + wi * V[j][d], 0)));
  return { Q, K, V, raw, scaled, masked, weights, out, WQ, WK, WV, dk };
}

/** 머리 여러 개로 나눠 각각 attention 을 돌린 뒤 이어 붙인다 */
export function multiHeadAttention(X, { heads = 4, dk = 4, seed = 5, mask = null } = {}) {
  const per = [];
  for (let hI = 0; hI < heads; hI++) per.push(selfAttention(X, { dk, seed: seed + hI * 17, mask }));
  const concat = X.map((_, i) => per.flatMap((p) => p.out[i]));
  return { heads: per, concat };
}

/** 미래를 가리는 삼각 마스크 — GPT 가 왼쪽만 보게 만드는 장치 */
export function causalMask(n) {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j <= i));
}

/** 위치 인코딩 — 짝수 차원은 sin, 홀수 차원은 cos */
export function positionalEncoding(len, dim) {
  const out = [];
  for (let pos = 0; pos < len; pos++) {
    const row = [];
    for (let i = 0; i < dim; i++) {
      const p = Math.floor(i / 2);
      const angle = pos / Math.pow(10000, (2 * p) / dim);
      row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    out.push(row);
  }
  return out;
}

/** 층 정규화 — 한 줄의 평균 0, 표준편차 1 */
export function layerNorm(v) {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  const s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) || 1;
  return v.map((x) => (x - m) / s);
}

/** 자리마다 따로 도는 작은 2층 신경망 (Attention 이 본 것을 해석하는 부분) */
export function feedForward(v, { expand = 4, seed = 21 } = {}) {
  const d = v.length;
  const W1 = randMatrix(expand * d, d, seed, 0.5);
  const W2 = randMatrix(d, expand * d, seed + 1, 0.5);
  const hidden = matVec(W1, v).map(relu);
  return { hidden, out: matVec(W2, hidden) };
}
