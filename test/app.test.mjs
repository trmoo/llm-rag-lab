/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * app.test.mjs — 그래프 엔진·언어 모델·신경망 계산, 그리고 퀴즈 자료의 무결성.
 *
 * 「화면에 적어 둔 정답과 데이터가 어긋나지 않는가」를 지키는 것이 목적이다.
 * 보기 번호가 범위를 넘거나, 해설이 빠진 문항이 있으면 여기서 걸린다.
 * ========================================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { describe, test, eq, near, ok, includes } from './harness.mjs';
import { stateGraph, memorySaver, addReducer, addMessages, START, END, toMermaid } from '../src/lib/graph.js';
import { trainNgram, sample, generate, tokenize, countTokens, EOS } from '../src/lib/lm.js';
import {
  softmax, sigmoid, relu, tanh, bce, trainMLP, selfAttention, causalMask,
  positionalEncoding, layerNorm, lstmForward, rnnForward, gradientDecay, ACTIVATIONS,
} from '../src/lib/nn.js';
import { trainWord2Vec } from '../src/lib/w2v.js';
import { embed } from '../src/lib/embed.js';
import { LM_SENTENCES, W2V_SENTENCES } from '../src/data/corpus.js';

/* ══════════════════════════ 그래프 엔진 ═══════════════════════════════ */
describe('그래프', () => {
  const build = (reducer) => stateGraph({ log: reducer ? addReducer : undefined })
    .addNode('더하기', (s) => ({ number: s.number + 1, log: ['+1'] }))
    .addNode('두배', (s) => ({ number: s.number * 2, log: ['x2'] }))
    .addEdge(START, '더하기').addEdge('더하기', '두배').addEdge('두배', END)
    .compile();

  test('3 → +1 → ×2 = 8', () => {
    eq(build(true).invoke({ number: 3, log: [] }).state.number, 8);
  });
  test('이어붙이기 규칙이 있으면 기록이 쌓인다', () => {
    eq(build(true).invoke({ number: 3, log: [] }).state.log, ['+1', 'x2']);
  });
  test('규칙이 없으면 덮어써서 앞의 것이 사라진다', () => {
    eq(build(false).invoke({ number: 3, log: [] }).state.log, ['x2']);
  });
  test('노드가 돌려주지 않은 칸은 그대로 남는다', () => {
    const g = stateGraph({}).addNode('a', () => ({ x: 1 }))
      .addEdge(START, 'a').addEdge('a', END).compile();
    eq(g.invoke({ x: 0, y: '그대로' }).state.y, '그대로');
  });
  test('조건 분기 — 짝수와 홀수', () => {
    const g = stateGraph({})
      .addNode('판단', (s) => ({ n: s.number }))
      .addNode('짝', () => ({ result: '짝수' }))
      .addNode('홀', () => ({ result: '홀수' }))
      .addEdge(START, '판단')
      .addConditionalEdges('판단', (s) => (s.number % 2 === 0 ? 'e' : 'o'), { e: '짝', o: '홀' })
      .addEdge('짝', END).addEdge('홀', END)
      .compile();
    eq(g.invoke({ number: 4, result: '' }).state.result, '짝수');
    eq(g.invoke({ number: 7, result: '' }).state.result, '홀수');
  });
  test('루프는 조건을 만나면 멈춘다', () => {
    const g = stateGraph({})
      .addNode('세기', (s) => ({ count: s.count + 1 }))
      .addEdge(START, '세기')
      .addConditionalEdges('세기', (s) => (s.count < 5 ? 'c' : 'e'), { c: '세기', e: END })
      .compile();
    const r = g.invoke({ count: 0 });
    eq(r.state.count, 5);
    eq(r.hops, 5);
    eq(r.stopped, '');
  });
  test('멈추지 않으면 안전장치가 잡는다', () => {
    const g = stateGraph({})
      .addNode('세기', (s) => ({ count: s.count + 1 }))
      .addEdge(START, '세기')
      .addConditionalEdges('세기', () => 'c', { c: '세기', e: END })
      .compile({ recursionLimit: 12 });
    const r = g.invoke({ count: 0 });
    ok(r.stopped, '안전장치가 작동해야 한다');
    eq(r.hops, 12);
  });
  test('대화방 번호가 다르면 기억이 섞이지 않는다', () => {
    const saver = memorySaver();
    const g = stateGraph({ messages: addMessages })
      .addNode('말하기', (s) => ({ messages: [{ role: 'ai', text: '응답' }] }))
      .addEdge(START, '말하기').addEdge('말하기', END)
      .compile({ checkpointer: saver });
    g.invoke({ messages: [{ role: 'human', text: 'A 의 말' }] }, { threadId: 'A' });
    g.invoke({ messages: [{ role: 'human', text: 'B 의 말' }] }, { threadId: 'B' });
    const a = saver.get('A').messages.map((m) => m.text).join(' ');
    const b = saver.get('B').messages.map((m) => m.text).join(' ');
    ok(a.includes('A 의 말') && !a.includes('B 의 말'), 'A 방에 B 의 말이 있으면 안 된다');
    ok(b.includes('B 의 말') && !b.includes('A 의 말'), 'B 방에 A 의 말이 있으면 안 된다');
  });
  test('같은 대화방은 대화가 이어진다', () => {
    const saver = memorySaver();
    const g = stateGraph({ messages: addMessages })
      .addNode('말하기', () => ({ messages: [{ role: 'ai', text: '네' }] }))
      .addEdge(START, '말하기').addEdge('말하기', END)
      .compile({ checkpointer: saver });
    g.invoke({ messages: [{ role: 'human', text: '첫 말' }] }, { threadId: 'X' });
    const r = g.invoke({ messages: [{ role: 'human', text: '둘째 말' }] }, { threadId: 'X' });
    eq(r.state.messages.length, 4);
  });
  test('그래프 구조를 그림 글자로 뽑을 수 있다', () => {
    const g = build(true);
    const m = toMermaid(g.describe());
    includes(m, 'graph TD');
    includes(m, '더하기');
  });
});

/* ══════════════════════════ 언어 모델 ═════════════════════════════════ */
describe('다음 낱말 예측', () => {
  const M = trainNgram(LM_SENTENCES.map((s) => s.split(' ')));

  test('확률의 합은 1', () => {
    const { list } = M.distribution(['도서관']);
    near(list.reduce((a, x) => a + x.p, 0), 1, 1e-9);
  });
  test('같은 시작이 여러 갈래로 이어지도록 자료가 만들어져 있다', () => {
    const { list } = M.distribution(['도서관', '열람실은']);
    ok(list.length >= 3, '후보가 셋 이상이어야 온도 실험이 뜻을 갖는다');
  });
  test('greedy 는 늘 1등을 고른다', () => {
    const { list } = M.distribution(['도서관', '열람실은']);
    const r = sample(list, { mode: 'greedy' }, () => 0.99);
    eq(r.picked, list[0].word);
  });
  test('top-k 는 k 개만 남긴다', () => {
    const { list } = M.distribution(['도서관']);
    const r = sample(list, { mode: 'topk', k: 2, temperature: 1 }, () => 0.5);
    eq(r.pool.filter((x) => x.keep).length, Math.min(2, list.length));
  });
  test('top-p 는 누적 확률까지만 남긴다', () => {
    const list = [{ word: 'a', p: 0.6 }, { word: 'b', p: 0.3 }, { word: 'c', p: 0.1 }];
    const r = sample(list, { mode: 'topp', p: 0.85, temperature: 1 }, () => 0.1);
    ok(r.pool.filter((x) => x.keep).length <= 2, '0.6+0.3 이면 두 개까지');
  });
  test('온도가 낮으면 분포가 뾰족해진다', () => {
    const list = [{ word: 'a', p: 0.5 }, { word: 'b', p: 0.3 }, { word: 'c', p: 0.2 }];
    const cold = sample(list, { mode: 'temperature', temperature: 0.2 }, () => 0.99);
    const hot = sample(list, { mode: 'temperature', temperature: 2 }, () => 0.99);
    ok(cold.pool[0].q > hot.pool[0].q, '온도가 낮을수록 1등의 확률이 커야 한다');
  });
  test('같은 씨앗이면 같은 문장이 나온다', () => {
    const a = generate(M, ['오늘'], { mode: 'topk', k: 3, seed: 7 }).words.join(' ');
    const b = generate(M, ['오늘'], { mode: 'topk', k: 3, seed: 7 }).words.join(' ');
    eq(a, b);
  });
  test('생성이 끝없이 이어지지 않는다', () => {
    const r = generate(M, ['오늘'], { mode: 'greedy', maxLen: 15 });
    ok(r.words.length <= 16, '최대 길이를 지켜야 한다');
    ok(!r.words.includes(EOS), '끝 표시는 결과에 들어가지 않는다');
  });
});

describe('토큰 세기', () => {
  test('한글은 두 글자씩 묶인다', () => {
    eq(tokenize('안녕하세요').map((t) => t.text), ['안녕', '하세', '요']);
  });
  test('영어는 네 글자씩 묶인다', () => {
    eq(tokenize('library').map((t) => t.text), ['libr', 'ary']);
  });
  test('같은 뜻이면 한국어가 토큰을 더 먹는 편이다', () => {
    ok(countTokens('도서관 이용 안내입니다') >= countTokens('library guide'), '경험칙대로여야 한다');
  });
  test('빈 글은 토큰 0', () => eq(countTokens('   '), 0));
});

/* ══════════════════════════ 신경망 ════════════════════════════════════ */
describe('신경망', () => {
  test('활성화 함수의 값', () => {
    near(sigmoid(0), 0.5);
    near(relu(-3), 0);
    near(relu(5), 5);
    near(tanh(0), 0);
  });
  test('활성화 함수마다 설명이 있다', () => {
    for (const [k, v] of Object.entries(ACTIVATIONS)) ok(v.range && v.use, `${k} 설명 누락`);
  });
  test('softmax 의 합은 1', () => {
    near(softmax([3, 1, 0]).reduce((a, b) => a + b, 0), 1, 1e-9);
  });
  test('온도가 0에 가까우면 1등이 독차지', () => {
    const s = softmax([3, 1, 0], 0.05);
    ok(s[0] > 0.99, `1등이 독차지해야 한다: ${s[0]}`);
  });
  test('손실은 틀릴수록 커진다', () => {
    ok(bce(0.9, 1) < bce(0.5, 1), '정답에 가까울수록 작아야 한다');
    ok(bce(0.5, 1) < bce(0.1, 1), '멀수록 커야 한다');
  });
  test('XOR 은 은닉층 없이 못 풀고, 있으면 푼다', () => {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const y = [0, 1, 1, 0];
    const small = trainMLP(X, y, { hidden: 1, epochs: 800, lr: 1.2, seed: 5 });
    const big = trainMLP(X, y, { hidden: 8, epochs: 800, lr: 1.2, seed: 5 });
    ok(small.history.at(-1).acc < 1, '뉴런 하나로는 못 푼다');
    eq(big.history.at(-1).acc, 1, '은닉층이 넉넉하면 푼다');
  });
  test('학습하면 손실이 줄어든다', () => {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const m = trainMLP(X, [0, 1, 1, 0], { hidden: 8, epochs: 400, lr: 1.2, seed: 5 });
    ok(m.history.at(-1).loss < m.history[0].loss, '손실이 내려가야 한다');
  });
  test('기울기는 층을 지날수록 사그라든다', () => {
    const d = gradientDecay(0.9, 20);
    ok(d.at(-1).g < 0.15, `20층 뒤에는 거의 사라져야 한다: ${d.at(-1).g}`);
  });
});

describe('Attention', () => {
  const toks = ['카페', '커피', '향', '진하다'];
  const X = toks.map((t) => embed(t).slice(0, 8));

  test('가중치는 줄마다 합이 1', () => {
    const A = selfAttention(X, { dk: 4, seed: 5 });
    for (const row of A.weights) near(row.reduce((a, b) => a + b, 0), 1, 1e-9);
  });
  test('출력 모양이 입력 토큰 수와 같다', () => {
    const A = selfAttention(X, { dk: 4, seed: 5 });
    eq(A.out.length, toks.length);
    eq(A.out[0].length, 4);
  });
  test('마스크를 씌우면 오른쪽 위가 0 이 된다', () => {
    const A = selfAttention(X, { dk: 4, seed: 5, mask: causalMask(toks.length) });
    for (let i = 0; i < toks.length; i++) {
      for (let j = i + 1; j < toks.length; j++) near(A.weights[i][j], 0, 1e-12, `(${i},${j})`);
    }
    near(A.weights[0][0], 1, 1e-9, '첫 낱말은 자기 자신만 본다');
  });
  test('Attention 만으로는 순서를 구별하지 못한다', () => {
    const a = ['나는', '카페에', '갔다'].map((t) => embed(t).slice(0, 8));
    const b = ['카페에', '나는', '갔다'].map((t) => embed(t).slice(0, 8));
    const A = selfAttention(a, { dk: 4, seed: 9 });
    const B = selfAttention(b, { dk: 4, seed: 9 });
    ok(A.out[2].every((x, i) => Math.abs(x - B.out[2][i]) < 1e-9), '위치를 안 보므로 같아야 한다');
  });
  test('위치 인코딩을 더하면 순서를 구별한다', () => {
    const pe = positionalEncoding(3, 8);
    const add = (arr) => arr.map((row, i) => row.map((v, j) => v + pe[i][j] * 0.5));
    const A = selfAttention(add(['나는', '카페에', '갔다'].map((t) => embed(t).slice(0, 8))), { dk: 4, seed: 9 });
    const B = selfAttention(add(['카페에', '나는', '갔다'].map((t) => embed(t).slice(0, 8))), { dk: 4, seed: 9 });
    ok(A.out[2].some((x, i) => Math.abs(x - B.out[2][i]) > 1e-9), '이제는 달라야 한다');
  });
  test('위치 인코딩은 짝수 칸 sin, 홀수 칸 cos', () => {
    const pe = positionalEncoding(4, 6);
    near(pe[0][0], 0, 1e-12, 'sin(0) = 0');
    near(pe[0][1], 1, 1e-12, 'cos(0) = 1');
  });
  test('층 정규화는 평균 0, 표준편차 1', () => {
    const v = layerNorm([1, 2, 3, 10]);
    near(v.reduce((a, b) => a + b, 0) / v.length, 0, 1e-9);
  });
});

describe('RNN·LSTM', () => {
  const X = ['이', '카페', '는', '조용하다'].map((t) => embed(t).slice(0, 6));
  test('RNN 은 시점마다 기억을 갱신한다', () => {
    const { steps } = rnnForward(X, { hiddenDim: 4 });
    eq(steps.length, 4);
    ok(steps[0].hPrev.every((v) => v === 0), '처음 기억은 0');
    ok(steps[1].hPrev.some((v) => v !== 0), '두 번째부터는 앞 기억이 들어온다');
  });
  test('LSTM 게이트 값은 0~1 사이', () => {
    const { steps } = lstmForward(X, { hiddenDim: 4 });
    for (const s of steps) {
      for (const arr of [s.f, s.i, s.o]) {
        ok(arr.every((v) => v >= 0 && v <= 1), '게이트는 0~1 이어야 한다');
      }
    }
  });
});

describe('Word2Vec', () => {
  const m = trainWord2Vec(W2V_SENTENCES, { vectorSize: 24, window: 3, epochs: 150, seed: 42 });

  test('어휘가 만들어진다', () => ok(m.vocab.length > 50, `어휘가 너무 적습니다: ${m.vocab.length}`));
  test('학습으로 손실이 줄어든다', () => {
    ok(m.history.at(-1).loss < m.history[0].loss * 0.7, '손실이 눈에 띄게 줄어야 한다');
  });
  test('같은 갈래의 낱말이 서로 가깝다', () => {
    ok(m.similarity('커피', '라떼') > m.similarity('커피', '반납'), '커피는 라떼와 더 가까워야 한다');
    ok(m.similarity('도서관', '열람실') > m.similarity('도서관', '커피'), '도서관은 열람실과 더 가까워야 한다');
  });
  test('없는 낱말은 null 을 돌려준다 (OOV)', () => {
    eq(m.has('우주선'), false);
    eq(m.getUnit('우주선'), null);
    eq(m.similarity('우주선', '커피'), null);
  });
  test('벡터의 길이는 1로 맞춰져 있다', () => {
    near(Math.hypot(...m.getUnit('커피')), 1, 1e-9);
  });
  test('문서 벡터는 낱말 벡터의 평균', () => {
    const v = m.docVector(['커피', '향']);
    eq(v.length, m.dim);
  });
});

/* ══════════════════════════ 퀴즈 자료 무결성 ══════════════════════════ */
describe('퀴즈 자료', () => {
  const tabDir = new URL('../src/tabs/', import.meta.url);
  const files = fs.readdirSync(tabDir).filter((f) => f.endsWith('.js'));

  test('탭 파일이 일곱 개 있다', () => eq(files.length, 7));

  test('quizBlock 의 열쇠가 겹치지 않는다', () => {
    const keys = [];
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      for (const m of src.matchAll(/quizBlock\(\s*'([^']+)'/g)) keys.push(m[1]);
    }
    eq(keys.length, new Set(keys).size, `열쇠가 겹칩니다: ${keys.filter((k, i) => keys.indexOf(k) !== i)}`);
    ok(keys.length >= 20, `퀴즈 묶음이 너무 적습니다: ${keys.length}`);
  });

  test('모든 화면에 퀴즈나 마무리가 붙어 있다', () => {
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      const screens = [...src.matchAll(/^\s*id: '([a-z]+)',\s*$/gm)].length;
      ok(screens > 0, `${f} 에 화면이 없습니다`);
    }
  });

  test('보기 번호(answer)가 보기 개수를 넘지 않는다', () => {
    // 각 문항의 options 배열 길이와 answer 를 정적으로 훑어 확인한다
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      const re = /options:\s*\[([\s\S]*?)\],\s*\n\s*answer:\s*(\[[^\]]*\]|\d+)/g;
      for (const m of src.matchAll(re)) {
        const n = m[1].split(/',\s*'|",\s*"|\),\s*mono|\],\s*\[/).length;
        const ans = m[2].startsWith('[')
          ? JSON.parse(m[2].replace(/\s/g, ''))
          : [Number(m[2])];
        for (const a of ans) ok(a >= 0 && a < 8, `${f}: 보기 번호가 이상합니다 (${a})`);
      }
    }
  });

  test('모든 문항에 해설(why)이 있다', () => {
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      const answers = (src.match(/^\s*answer:/gm) || []).length;
      const accepts = (src.match(/^\s*accept:/gm) || []).length;
      const whys = (src.match(/^\s*why:/gm) || []).length;
      eq(whys, answers + accepts, `${f}: 해설 수가 문항 수와 다릅니다`);
    }
  });

  test('짧은 답 문항의 정답이 비어 있지 않다', () => {
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      for (const m of src.matchAll(/accept:\s*\[([^\]]*)\]/g)) {
        ok(m[1].trim().length > 2, `${f}: accept 가 비었습니다`);
      }
    }
  });
});

/* ══════════════════════════ 화면 수명 관리 ════════════════════════════ */
describe('화면 수명 관리', () => {
  const tabDir = new URL('../src/tabs/', import.meta.url);
  const files = fs.readdirSync(tabDir).filter((f) => f.endsWith('.js'));

  test('탭 파일에서 window 리스너·타이머를 직접 쓰지 않는다', () => {
    for (const f of files) {
      const src = fs.readFileSync(new URL(f, tabDir), 'utf8');
      ok(!/window\.addEventListener/.test(src), `${f}: onResize( ) 를 쓰세요`);
      ok(!/\bsetInterval\(/.test(src), `${f}: screenInterval( ) 를 쓰세요`);
    }
  });
});

/* ══════════════════════════ 저작권 표시 ═══════════════════════════════ */
describe('저작권 표시', () => {
  const root = new URL('../', import.meta.url);
  const srcFiles = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(new URL(dir, root), { withFileTypes: true })) {
      if (e.isDirectory()) walk(`${dir}${e.name}/`);
      else if (e.name.endsWith('.js') || e.name.endsWith('.css')) srcFiles.push(`${dir}${e.name}`);
    }
  };
  walk('src/');

  test('모든 소스 파일 머리에 저작권 주석이 있다', () => {
    const missing = srcFiles.filter((f) => !fs.readFileSync(new URL(f, root), 'utf8').startsWith('/*!'));
    eq(missing, [], '저작권 주석이 없는 파일');
  });
  test('LICENSE 와 README 에 표시가 있다', () => {
    includes(fs.readFileSync(new URL('LICENSE', root), 'utf8'), '티쳐무');
    includes(fs.readFileSync(new URL('index.html', root), 'utf8'), '티쳐무');
  });
  test('package.json 의 license 가 UNLICENSED', () => {
    const p = JSON.parse(fs.readFileSync(new URL('package.json', root), 'utf8'));
    eq(p.license, 'UNLICENSED');
  });
  test('vite.config.js 에 배너가 있다', () => {
    const v = fs.readFileSync(new URL('vite.config.js', root), 'utf8');
    includes(v, '/*!');
    includes(v, '티쳐무');
  });
});
