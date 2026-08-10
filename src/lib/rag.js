/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * rag.js — 문서를 잘라 넣고, 찾아서, 근거로 답하기
 *
 *   ① 자르기(청킹)   글자 수 기준 · 제목(구조) 기준
 *   ② 찾기(검색)     의미 검색 · 낱말 검색(BM25) · 둘을 섞기 · MMR · 꼬리표 거르기
 *   ③ 넣기(증강)     프롬프트의 [컨텍스트] 자리에 끼워 넣기
 *   ④ 답하기(생성)   찾은 문장에서 근거를 뽑아 답을 만든다
 *
 * ⚠️ ④의 「답하기」는 진짜 LLM 이 아니다.
 *   교실에서 인터넷·API 키 없이도 「문서를 주면 답이 어떻게 달라지는가」를
 *   보여 주기 위해, 찾은 글에서 질문과 가장 맞는 문장을 뽑아 답을 조립한다.
 *   그래서 문서를 안 주면 못 답하고, 프롬프트 지시를 약하게 하면 지어낸다 —
 *   RAG 수업에서 봐야 할 성질은 그대로 재현된다. 화면에도 이렇게 적어 두었다.
 * ========================================================================== */

import { cosine, bm25Search, mmrSelect, preprocess, counter } from './nlp.js';
import { embed } from './embed.js';

/* ══════════════════════════ ① 자르기 ═══════════════════════════════════ */

/**
 * 글자 수로 자르되, 문단 → 문장 → 낱말 순으로 「자연스러운 경계」부터 찾는다.
 * 그래서 낱말 한가운데가 뚝 잘리는 일이 적다. (RecursiveCharacterTextSplitter 와 같은 생각)
 */
export function splitByChars(text, { chunkSize = 300, overlap = 40 } = {}) {
  const seps = ['\n\n', '\n', '. ', '다. ', ' ', ''];
  const chunks = [];

  const rec = (s, depth) => {
    if (s.length <= chunkSize) { if (s.trim()) chunks.push(s); return; }
    const sep = seps[Math.min(depth, seps.length - 1)];
    if (sep === '') {
      for (let i = 0; i < s.length; i += chunkSize) chunks.push(s.slice(i, i + chunkSize));
      return;
    }
    const parts = s.split(sep).filter((x) => x !== '');
    let buf = '';
    for (const p of parts) {
      const piece = buf ? buf + sep + p : p;
      if (piece.length <= chunkSize) { buf = piece; continue; }
      if (buf) chunks.push(buf);
      if (p.length > chunkSize) rec(p, depth + 1);
      else buf = p;
      if (p.length > chunkSize) buf = '';
    }
    if (buf.trim()) chunks.push(buf);
  };
  rec(text.trim(), 0);

  // 겹치기 — 앞 조각의 끝부분을 다음 조각 앞에 덧붙인다
  if (overlap > 0 && chunks.length > 1) {
    return chunks.map((c, i) => (i === 0 ? c : chunks[i - 1].slice(-overlap) + c));
  }
  return chunks;
}

/**
 * 제목 구조를 따라 자른다. 「## 반납」 같은 줄을 만나면 새 조각을 시작하고,
 * 제목을 꼬리표(메타데이터)로 함께 저장한다. 나중에 「반납 부분만 검색」이 가능해진다.
 */
export function splitByHeadings(text) {
  const lines = text.split('\n');
  const out = [];
  let cur = null;
  for (const line of lines) {
    const m = /^(#{1,4})\s+(.*)$/.exec(line.trim());
    if (m) {
      if (cur && cur.body.trim()) out.push(cur);
      cur = { title: m[2].trim(), level: m[1].length, body: '' };
    } else if (cur) {
      cur.body += line + '\n';
    } else if (line.trim()) {
      cur = { title: '머리말', level: 1, body: line + '\n' };
    }
  }
  if (cur && cur.body.trim()) out.push(cur);
  return out.map((s) => ({
    text: `[${s.title}] ${s.body.trim().replace(/\n+/g, ' ')}`,
    metadata: { 섹션: s.title },
  }));
}

/** 잘린 조각을 검색용 문서로 만든다 (벡터를 미리 계산해 둔다 = 인덱싱) */
export function buildIndex(chunks) {
  const docs = chunks.map((c, i) => {
    const text = typeof c === 'string' ? c : c.text;
    const metadata = typeof c === 'string' ? {} : (c.metadata || {});
    return { id: i, text, metadata, vector: embed(text), tokens: preprocess(text) };
  });
  return {
    docs,
    /** 조각끼리 얼마나 닮았는지 (MMR 에서 쓴다) */
    pairSim: docs.map((a) => docs.map((b) => cosine(a.vector, b.vector))),
  };
}

/* ══════════════════════════ ② 찾기 ═════════════════════════════════════ */

/**
 * @param {object} index buildIndex 결과
 * @param {string} question
 * @param {object} opt
 *   k            몇 개를 최종으로 쓸지
 *   mode         'vector' | 'keyword' | 'hybrid'
 *   weightVector 하이브리드에서 의미 검색의 비중 (0~1)
 *   mmr          true 면 겹치지 않게 고른다
 *   lambda       관련성 ↔ 다양성 균형
 *   fetchK       MMR 후보 수
 *   filter       { 섹션: '반납' } 처럼 꼬리표로 범위 좁히기
 *   threshold    이 점수보다 낮으면 아예 안 돌려준다
 */
export function retrieve(index, question, opt = {}) {
  const k = opt.k ?? 3;
  const mode = opt.mode || 'vector';
  const wv = opt.weightVector ?? 0.6;

  let pool = index.docs;
  if (opt.filter) {
    pool = pool.filter((d) => Object.entries(opt.filter).every(([key, val]) => !val || d.metadata[key] === val));
  }
  if (!pool.length) return { hits: [], scores: [], note: '꼬리표 조건에 맞는 조각이 없습니다.' };

  const qv = embed(question);
  const qTokens = preprocess(question);
  const vecScore = pool.map((d) => cosine(qv, d.vector));

  // 낱말 검색 점수를 0~1 로 맞춰 둔다 (두 점수를 섞으려면 자 눈금이 같아야 한다)
  const bm = bm25Search(qTokens, pool.map((d) => d.tokens), { topK: pool.length });
  const bmRaw = new Array(pool.length).fill(0);
  bm.forEach((r) => { bmRaw[r.i] = r.score; });
  const bmMax = Math.max(1e-9, ...bmRaw);
  const kwScore = bmRaw.map((s) => s / bmMax);

  let score;
  if (mode === 'keyword') score = kwScore;
  else if (mode === 'hybrid') score = vecScore.map((v, i) => wv * v + (1 - wv) * kwScore[i]);
  else score = vecScore;

  let pickedIdx;
  let mmrTrace = null;
  if (opt.mmr) {
    const pair = pool.map((a) => pool.map((b) => cosine(a.vector, b.vector)));
    const r = mmrSelect(score, pair, { k, lambda: opt.lambda ?? 0.5, fetchK: opt.fetchK ?? Math.min(pool.length, 8) });
    pickedIdx = r.picked;
    mmrTrace = r.trace;
  } else {
    pickedIdx = score.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);
  }

  let hits = pickedIdx.map((i) => ({ doc: pool[i], score: score[i], vec: vecScore[i], kw: kwScore[i] }));
  if (opt.threshold !== undefined) hits = hits.filter((hh) => hh.score >= opt.threshold);

  return {
    hits,
    scores: pool.map((d, i) => ({ id: d.id, score: score[i], vec: vecScore[i], kw: kwScore[i] })),
    mmrTrace,
    note: hits.length ? '' : '기준 점수를 넘는 조각이 없습니다.',
  };
}

/* ══════════════════════════ ③ 넣기 ═════════════════════════════════════ */

export const PROMPT_PARTS = {
  role: { label: '역할 정해 주기', text: '당신은 안내문을 읽고 답하는 도우미입니다.' },
  only: { label: '컨텍스트에만 근거하기', text: '반드시 아래 [컨텍스트]에 있는 내용만으로 답하세요.' },
  refuse: { label: '없으면 없다고 말하기', text: '컨텍스트에 없는 내용은 추측하지 말고 "안내되어 있지 않습니다"라고 답하세요.' },
  short: { label: '짧게 답하기', text: '답변은 세 문장 이내로 간결하게 하세요.' },
  cite: { label: '출처 함께 보여 주기', text: '답변 끝에 근거가 된 부분의 제목을 함께 적으세요.' },
  fewshot: {
    label: '모르는 질문 예시 보여 주기',
    text: '[예시]\n컨텍스트: (대출·반납 안내만 있음)\n질문: 주차는 무료인가요?\n답변: 제공된 안내문에는 주차에 관한 내용이 없습니다.',
  },
};

/** 프롬프트 문자열을 조립한다 — 화면에서 그대로 보여 준다 */
export function buildPrompt(question, hits, on = {}) {
  const rules = [];
  ['role', 'only', 'refuse', 'short', 'cite'].forEach((key) => {
    if (on[key]) rules.push(PROMPT_PARTS[key].text);
  });
  const ctx = hits.map((hh, i) => `(${i + 1}) ${hh.doc.text}`).join('\n');
  const parts = [];
  if (rules.length) parts.push('규칙:\n' + rules.map((r, i) => `${i + 1}) ${r}`).join('\n'));
  if (on.fewshot) parts.push(PROMPT_PARTS.fewshot.text);
  parts.push('[컨텍스트]\n' + (ctx || '(없음)'));
  parts.push('[질문]\n' + question);
  return parts.join('\n\n');
}

/* ══════════════════════════ ④ 답하기 ═══════════════════════════════════ */

/**
 * 글을 문장 단위로 나눈다.
 * ⚠️ 「…다 」로 끊으면 안 된다 — 「하루마다 이틀씩」의 '다' 에서 잘려 버린다.
 *    마침표·물음표·느낌표에서만 끊는다.
 */
export function sentences(text) {
  const body = text.replace(/^\[[^\]]*\]\s*/, '');
  const parts = body.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter((s) => s.length > 4);
  return parts.length ? parts : [body.trim()].filter((s) => s.length > 4);
}

/**
 * 찾은 조각에서 질문과 가장 맞는 문장을 뽑아 답을 만든다.
 * (진짜 LLM 이 아니라, 「근거를 보고 답한다」가 무슨 뜻인지 보여 주는 장치)
 */
/** 「며칠 · 몇 권 · 얼마」처럼 수를 묻는 질문인가 */
const asksNumber = (q) => /며칠|몇|얼마|언제|기간|시간|권|번|시\b/.test(q);

/**
 * 질문 넓히기(Query Expansion).
 * 사람은 「며칠」이라고 묻지만 안내문에는 「기간은 21일」이라고 적혀 있다.
 * 낱말이 하나도 겹치지 않으면 검색이 실패한다 — RAG 가 무너지는 흔한 자리다.
 * 그래서 물음말을 문서에서 쓸 법한 낱말로 넓혀 준다.
 * 화면에서는 무엇이 넓혀졌는지 그대로 보여 준다 (몰래 하지 않는다).
 */
export const SYNONYMS = {
  며칠: ['기간', '일'],
  얼마나: ['기간', '시간', '동안'],
  언제: ['시간', '요일', '시'],
  몇: ['최대', '권'],
  빌릴: ['대출'],
  빌리: ['대출'],
  빌려: ['대출'],
  돈: ['연체료', '수수료', '요금', '값'],
  연체료: ['연체', '수수료'],
  닫나요: ['시간', '시'],
  무료: ['요금', '비용'],
  주차: ['주차장'],
  자리: ['좌석', '예약'],
};

export function expandQuery(tokens) {
  const added = [];
  const out = [...tokens];
  for (const t of tokens) {
    const syn = SYNONYMS[t];
    if (!syn) continue;
    for (const s of syn) {
      if (!out.includes(s)) { out.push(s); added.push({ from: t, to: s }); }
    }
  }
  return { tokens: out, added };
}

export function answerFromContext(question, hits, opt = {}) {
  const qv = embed(question);
  // 질문 쪽은 한 글자 낱말(돈·값·시)도 살린다 — 짧은 물음말이 핵심일 때가 많다
  const base = preprocess(question, { minLen: 1 });
  const expanded = expandQuery(base);
  const qTokens = expanded.tokens;
  const qTok = new Set(qTokens);
  const wantsNumber = asksNumber(question);
  const maxHit = Math.max(1e-9, ...hits.map((hh) => hh.score));
  const cands = [];
  hits.forEach((hh, rank) => {
    sentences(hh.doc.text).forEach((s) => {
      const sTok = [...counter(preprocess(s)).keys()];
      // ① 낱말이 그대로 겹치는가
      const overlap = sTok.filter((w) => qTok.has(w)).length;
      // ② 글자로라도 겹치는가 (「연체」 ↔ 「연체료」처럼 토큰이 달라도 잡아낸다)
      const partial = qTokens.filter((w) => w.length >= 2 && s.includes(w)).length;
      // ③ 수를 묻는 질문에는 숫자가 든 문장을 앞세운다
      const numberBonus = wantsNumber && /\d/.test(s) ? 1 : 0;
      // ④ 검색이 잘된 조각에서 나온 문장을 앞세운다.
      //    특히 1등 조각을 크게 우대한다 — 검색이 답을 좌우한다는 것이 RAG 의 성질이다.
      const chunkBonus = rank === 0 ? 1 : 0.5 * (hh.score / maxHit);
      const sim = cosine(qv, embed(s));
      cands.push({
        s,
        score: sim + overlap * 0.22 + partial * 0.15 + numberBonus * 0.30 + chunkBonus * 0.40,
        sim, overlap, numberBonus, chunkBonus, source: hh.doc,
      });
    });
  });
  cands.sort((a, b) => b.score - a.score);

  const need = opt.minScore ?? 0.85;
  const best = cands[0];
  // 어느 조각도 질문과 가깝지 않으면 문장 하나가 우연히 닮았어도 답하지 않는다
  const retrievalOk = maxHit >= (opt.retrievalFloor ?? 0.33);
  const supported = best && best.score >= need && retrievalOk;

  if (!supported) {
    if (opt.strict) {
      return {
        text: '제공된 안내문에는 그 내용이 나와 있지 않습니다.',
        supported: false, used: [], candidates: cands.slice(0, 4),
      };
    }
    return {
      text: opt.guess || '아마 일반적으로는 그렇게 알고 있습니다. (근거 없이 지어낸 답)',
      supported: false, hallucinated: true, used: [], candidates: cands.slice(0, 4),
      expanded: expanded.added,
    };
  }

  const limit = opt.short ? 1 : 2;
  const used = [];
  const picked = [];
  for (const c of cands) {
    if (picked.length >= limit) break;
    if (c.score < best.score - 0.20) break;           // 가장 좋은 문장과 많이 떨어지면 그만
    if (picked.length && c.source !== best.source) continue; // 근거는 한 조각에서만 모은다
    if (picked.some((p) => p.s === c.s)) continue;
    picked.push(c);
    if (!used.includes(c.source)) used.push(c.source);
  }
  let text = picked.map((p) => p.s).join(' ');
  if (opt.cite && used.length) {
    const tags = used.map((d) => d.metadata?.섹션 || `조각 ${d.id + 1}`);
    text += `\n\n📎 근거: ${[...new Set(tags)].join(', ')}`;
  }
  return { text, supported: true, used, picked, candidates: cands.slice(0, 4), expanded: expanded.added };
}

/** 문서 없이 답할 때 — 그럴듯하지만 틀릴 수 있는 「기억에 의존한 답」 */
export function answerWithoutContext(question, guesses) {
  const qTok = preprocess(question);
  let best = null;
  for (const g of guesses) {
    const hit = g.keys.filter((k) => question.includes(k) || qTok.includes(k)).length;
    if (hit && (!best || hit > best.hit)) best = { ...g, hit };
  }
  return best
    ? { text: best.text, wrong: best.wrong }
    : { text: '일반적으로는 그렇게 하는 곳이 많습니다. (확실하지 않습니다)', wrong: true };
}
