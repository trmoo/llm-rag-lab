/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ③ 맥락을 읽기 — 순서와 문맥을 다루는 신경망
 *   1. 신경망 한 조각    가중합 → 활성화 → 손실 → 경사 하강 (XOR 로 확인)
 *   2. 순서를 기억하기   RNN·LSTM 을 한 걸음씩, 그리고 기울기 소실
 *   3. Attention 계산기  Q·K·V → 점수 → softmax → 가중 합
 *   4. 인코더 블록       위치 인코딩·잔차·머리 여러 개, 그리고 BERT vs GPT
 * ========================================================================== */

import {
  h, card, note, deepDive, terms, table, screenHead, button, input, textarea,
  toggle, pillGroup, code, pyBox, mono, b, nextHint, fx, pct, heatCell, slider, bar, onResize,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import {
  ACTIVATIONS, softmax, bce, trainMLP, gradientSteps, rnnForward, lstmForward,
  gradientDecay, selfAttention, multiHeadAttention, causalMask, positionalEncoding, layerNorm,
} from '../lib/nn.js';
import { embed } from '../lib/embed.js';
import { lineChart, heatmap, curve, barChart, PALETTE } from '../lib/chart.js';

const statBox = (k, v, kind = '') => h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));

/* ══════════════════════════ 1. 신경망 한 조각 ═════════════════════════ */
const nnScreen = {
  id: 'nn',
  title: '신경망 한 조각',
  render(ctx) {
    const wrap = h('div');

    /* ① 뉴런 하나 */
    const nState = { x1: 1, x2: 1, w1: 0.6, w2: -0.4, bias: 0.1, act: 'Sigmoid' };
    const nOut = h('div');
    const drawNeuron = () => {
      const z = nState.x1 * nState.w1 + nState.x2 * nState.w2 + nState.bias;
      const a = ACTIVATIONS[nState.act].fn(z);
      nOut.textContent = '';
      const cv = curve({
        height: 170,
        fns: [{ name: nState.act, fn: ACTIVATIONS[nState.act].fn, color: PALETTE[0] }],
        xMin: -6, xMax: 6,
        marker: { x: Math.max(-6, Math.min(6, z)), y: a },
      });
      onResize(() => cv.redraw());
      nOut.append(
        h('p', mono(`z = (${fx(nState.x1, 1)} × ${fx(nState.w1, 2)}) + (${fx(nState.x2, 1)} × ${fx(nState.w2, 2)}) + ${fx(nState.bias, 2)} = ${fx(z, 3)}`)),
        h('p', mono(`출력 = ${nState.act}(${fx(z, 3)}) = ${fx(a, 4)}`)),
        bar(nState.act === 'Sigmoid' ? a : Math.max(0, Math.min(1, a / 5)), { label: fx(a, 3) }),
        cv,
      );
    };

    /* ② 손실 */
    const lState = { p: 0.9, y: 1 };
    const lOut = h('div');
    const drawLoss = () => {
      lOut.textContent = '';
      const cv = curve({
        height: 170,
        fns: [{ name: `정답이 ${lState.y} 일 때의 손실`, fn: (p) => bce(Math.min(0.999, Math.max(0.001, p)), lState.y), color: PALETTE[3] }],
        xMin: 0.01, xMax: 0.99, yMin: 0, yMax: 5,
        marker: { x: lState.p, y: bce(lState.p, lState.y) },
      });
      onResize(() => cv.redraw());
      lOut.append(
        table(['예측 확률', `정답 ${lState.y} 에 대한 손실`], [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01].map((p) => [
          fx(p, 2), h('span', { style: { color: bce(p, lState.y) > 1 ? 'var(--bad)' : 'var(--good)' } }, fx(bce(p, lState.y), 3)),
        ]), { compact: true, rowClass: (r, i) => (Math.abs([0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01][i] - lState.p) < 0.06 ? 'hi' : '') }),
        cv,
      );
    };

    /* ③ 경사 하강 */
    const gState = { lr: 0.3, start: -4.5 };
    const gOut = h('div');
    const f = (x) => 0.25 * x * x + 0.6;
    const df = (x) => 0.5 * x;
    const drawGrad = () => {
      const steps = gradientSteps(f, df, gState.start, gState.lr, 12);
      const cv = lineChart({
        height: 210,
        series: [
          { name: '손실 곡선', points: Array.from({ length: 100 }, (_, i) => { const x = -6 + (12 * i) / 99; return [x, f(x)]; }), color: PALETTE[0] },
          { name: '한 걸음씩', points: steps.map((s) => [s.x, s.y]), color: PALETTE[3], dots: true, width: 1.4 },
        ],
        xMin: -6, xMax: 6, yMin: 0,
      });
      onResize(() => cv.redraw());
      gOut.textContent = '';
      gOut.append(cv,
        h('p', { style: { color: gState.lr > 3.6 ? 'var(--bad)' : 'var(--dim)' } },
          gState.lr > 3.6
            ? '⚠️ 학습률이 너무 큽니다. 최저점을 지나쳐 튕겨 나가며 손실이 오히려 커집니다.'
            : gState.lr < 0.08
              ? '학습률이 너무 작습니다. 옳은 방향이지만 12걸음으로는 바닥까지 못 갑니다.'
              : '적당합니다. 기울기가 가리키는 반대 방향으로 한 걸음씩 내려갑니다.'),
        table(['걸음', '자리(x)', '손실'], steps.slice(0, 7).map((s, i) => [i, fx(s.x, 3), fx(s.y, 3)]), { compact: true }),
      );
    };

    /* ④ XOR — 층을 쌓아야 하는 이유 */
    const xorBox = h('div');
    const runXor = (hidden) => {
      const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
      const y = [0, 1, 1, 0];
      const m = trainMLP(X, y, { hidden, epochs: 800, lr: 1.2, seed: 5 });
      const grid = [];
      for (let r = 0; r < 11; r++) {
        const row = [];
        for (let c = 0; c < 11; c++) row.push(m.predict([c / 10, 1 - r / 10]));
        grid.push(row);
      }
      const hm = heatmap({
        matrix: grid, cellMax: 1, height: 240, fmt: () => '',
        colorOf: (v) => `rgba(${v > 0.5 ? '52,211,153' : '248,113,113'},${(Math.abs(v - 0.5) * 1.7 + 0.15).toFixed(2)})`,
      });
      const lc = lineChart({
        height: 160,
        series: [
          { name: '손실', points: m.history.map((x) => [x.epoch, x.loss]) },
          { name: '정확도', points: m.history.map((x) => [x.epoch, x.acc]), color: PALETTE[2] },
        ],
        yMin: 0,
      });
      onResize(() => { hm.redraw(); lc.redraw(); });
      const acc = m.history.at(-1).acc;
      xorBox.textContent = '';
      xorBox.append(
        h('div.stat-row',
          statBox('은닉층 뉴런', `${hidden}개`),
          statBox('마지막 정확도', pct(acc, 0), acc > 0.99 ? 'good' : 'bad'),
          statBox('손실', fx(m.history.at(-1).loss, 3)),
        ),
        table(['입력', '정답', '모형의 답'], X.map((x, i) => [
          mono(`(${x[0]}, ${x[1]})`), y[i],
          h('span', { style: { color: (m.predict(x) >= 0.5 ? 1 : 0) === y[i] ? 'var(--good)' : 'var(--bad)' } },
            fx(m.predict(x), 3)),
        ]), { compact: true }),
        h('div.cols', h('div', h('p', b('학습 곡선')), lc), h('div', h('p', b('모형이 가른 영역')), hm)),
        hidden === 0 || hidden === 1
          ? note('bad', h('b', '직선 하나로는 XOR 을 가를 수 없습니다. '),
            '(0,0)과 (1,1)이 한편, (0,1)과 (1,0)이 다른 편인데, 이 넷을 직선 하나로 나눌 방법이 없습니다. ',
            '은닉층을 늘려 보세요.')
          : note('good', h('b', '층을 쌓으면 갈립니다. '),
            '은닉층의 뉴런들이 각자 다른 직선을 긋고, 출력층이 그 결과를 다시 조합해 굽은 경계를 만들었습니다. ',
            '층을 쌓아 얻는 것이 바로 이 「굽힘」입니다.'),
      );
    };

    wrap.append(
      screenHead('신경망 한 조각', '가중합 → 활성화 → 손실 → 되돌아가며 고치기. 이 네 걸음이 전부입니다.', '③ 맥락을 읽기'),

      card('왜 신경망이 필요해졌나',
        h('p', '탭 ②의 방법은 낱말을 ', b('세기만'), ' 했습니다. 그래서 이런 문장에서 무너졌습니다.'),
        code([
          '"조용하지 않아서 좋았어요"     ← 「않다」가 어디에 걸리는지 모름',
          '"배가 고프다" vs "배를 타다"   ← 같은 「배」를 늘 같은 것으로 봄',
        ].join('\n'), { copy: false }),
        h('p', b('순서'), ' 와 ', b('문맥'), ' 을 다루려면 새로운 연장이 필요합니다. 그 출발점이 신경망입니다.'),
      ),

      card('① 뉴런 하나가 하는 일',
        h('p', mono('출력 = 활성화함수( 입력₁×가중치₁ + 입력₂×가중치₂ + 편향 )')),
        h('p', { style: { color: 'var(--dim)' } },
          b('가중치'), ' 는 「이 입력이 얼마나 중요한가」, ', b('편향'), ' 은 「기본적으로 얼마나 켜져 있는가」입니다. ',
          b('학습이란 이 숫자들을 고쳐 나가는 일'), ' 이 전부입니다.'),
        h('div.cols',
          h('div',
            slider({ label: '입력 x₁', min: -2, max: 2, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { nState.x1 = v; drawNeuron(); } }),
            slider({ label: '입력 x₂', min: -2, max: 2, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { nState.x2 = v; drawNeuron(); } }),
          ),
          h('div',
            slider({ label: '가중치 w₁', min: -2, max: 2, step: 0.05, value: 0.6, format: (v) => v.toFixed(2), onInput: (v) => { nState.w1 = v; drawNeuron(); } }),
            slider({ label: '가중치 w₂', min: -2, max: 2, step: 0.05, value: -0.4, format: (v) => v.toFixed(2), onInput: (v) => { nState.w2 = v; drawNeuron(); } }),
            slider({ label: '편향 b', min: -2, max: 2, step: 0.05, value: 0.1, format: (v) => v.toFixed(2), onInput: (v) => { nState.bias = v; drawNeuron(); } }),
          ),
        ),
        pillGroup(Object.keys(ACTIVATIONS).map((k) => ({ label: k, value: k })), (v) => { nState.act = v; drawNeuron(); }),
        nOut,
        table(['활성화 함수', '나오는 값의 범위', '주로 쓰는 곳'],
          Object.entries(ACTIVATIONS).map(([k, v]) => [mono(k), v.range, v.use]), { compact: true }),
        note('info',
          h('b', '활성화 함수가 없으면 어떻게 될까요? '),
          '아무리 층을 많이 쌓아도 결국 ', b('직선 하나'), ' 와 똑같아집니다. ',
          '(직선 변환을 여러 번 이어 붙여도 직선입니다.) 굽은 경계를 만들려면 반드시 비선형이 필요합니다. ',
          '아래 XOR 실험에서 직접 확인합니다.'),
      ),

      card('② 손실 — 얼마나 틀렸나를 숫자 하나로',
        h('p', '학습의 목표는 이 값을 ', b('줄이는'), ' 것입니다. 정답에 가까울수록 작고, 멀수록 급격히 커집니다.'),
        h('div.pills',
          pillGroup([{ label: '정답이 1(긍정)', value: 1 }, { label: '정답이 0(부정)', value: 0 }],
            (v) => { lState.y = v; drawLoss(); }),
        ),
        slider({ label: '모형의 예측 확률', min: 0.01, max: 0.99, step: 0.01, value: 0.9, format: (v) => v.toFixed(2), onInput: (v) => { lState.p = v; drawLoss(); } }),
        lOut,
        note('info', '틀렸을 때 손실이 ', b('아주 크게'), ' 커지는 것이 중요합니다. ',
          '「확신을 갖고 틀리는 것」을 세게 벌해야 모형이 겸손해집니다.'),
      ),

      card('③ 경사 하강 — 눈 감고 산을 내려오기',
        h('p', '발밑의 기울기를 느끼고, 가장 가파른 ', b('반대 방향'), ' 으로 한 걸음. 이것을 되풀이합니다. ',
          '한 걸음의 크기가 ', b('학습률'), ' 입니다.'),
        slider({ label: '학습률', min: 0.02, max: 4.2, step: 0.02, value: 0.3, format: (v) => v.toFixed(2), onInput: (v) => { gState.lr = v; drawGrad(); } }),
        gOut,
        code([
          '한 걸음 = 지금 자리 − 학습률 × 기울기',
          '',
          'for epoch in range(에포크):',
          '    예측 = 모델(X)              # 1. 순전파',
          '    손실 = 손실함수(예측, y)      # 2. 손실 계산',
          '    optimizer.zero_grad()       # 3. 이전 기울기 지우기',
          '    손실.backward()             # 4. 역전파 — 기울기 계산',
          '    optimizer.step()            # 5. 가중치 고치기',
        ].join('\n'), { copy: false }),
      ),

      card('④ 층을 쌓아야만 풀리는 문제 (XOR)',
        h('p', '입력 두 개가 ', b('서로 다를 때만'), ' 1이 되는 문제입니다. 아주 단순해 보이지만 ',
          '직선 하나로는 절대 가를 수 없습니다.'),
        h('div.pills', ...[0, 1, 2, 4, 8].map((n) => h('button.pill', {
          type: 'button',
          onclick: () => runXor(n),
        }, n === 0 ? '은닉층 없이 (직선 하나)' : `은닉 뉴런 ${n}개`))),
        xorBox,
      ),

      deepDive('역전파는 무엇을 되돌리는가',
        h('p', '손실이 3.2 나왔다고 합시다. 그런데 ', b('어느 가중치를 얼마나 고쳐야'), ' 이 값이 줄어들까요? ',
          '가중치가 수백만 개면 하나씩 시험해 볼 수 없습니다.'),
        h('p', '역전파는 미분의 연쇄 법칙을 써서, ', b('출력에서 입력 쪽으로 거슬러 올라가며'), ' ',
          '「이 가중치를 조금 키우면 손실이 얼마나 변하나」를 한 번에 계산합니다. ',
          '그래서 층이 아무리 많아도 전체를 한 번 훑는 비용으로 모든 기울기를 얻습니다.'),
        code([
          '순전파:  입력 → [층1] → [층2] → 출력 → 손실',
          '역전파:  입력 ← [층1] ← [층2] ← 출력 ← 손실',
          '                ↑ 각 층에서 「내 가중치를 얼마나 고칠지」를 받아 간다',
        ].join('\n'), { copy: false }),
        h('p', b('이것이 딥러닝을 가능하게 한 결정적 도구입니다. '),
          '위 XOR 실험에서 손실이 내려가는 것은, 실제로 이 계산이 800번 돌아간 결과입니다.'),
      ),

      quizBlock('t3/nn', [
        {
          type: 'choice',
          q: '활성화 함수를 빼고 층만 여러 개 쌓으면 어떻게 되나요?',
          options: [
            '더 복잡한 패턴을 배운다',
            '결국 직선 하나(선형 변환)와 같아진다',
            '학습이 더 빨라진다',
            '층 수만큼 정확도가 오른다',
          ],
          answer: 1,
          why: '선형 변환을 아무리 이어 붙여도 선형입니다. 비선형을 끼워 넣어야 굽은 경계를 만들 수 있습니다.',
        },
        {
          type: 'choice',
          q: '학습률을 너무 크게 잡으면 무슨 일이 생기나요?',
          options: [
            '학습이 아주 빨리 끝난다',
            '최저점을 지나쳐 튕겨 나가며 손실이 오히려 커질 수 있다',
            '아무 변화도 없다',
            '기울기가 0이 된다',
          ],
          answer: 1,
          why: '위 경사 하강 그림에서 학습률을 4 가까이 올려 보면 점들이 좌우로 튀는 것을 볼 수 있습니다.',
        },
        {
          type: 'choice',
          q: '출력층에서 Sigmoid 를 즐겨 쓰는 까닭은?',
          options: [
            '계산이 가장 빨라서',
            '결과가 0~1 사이라 확률로 읽을 수 있어서',
            '음수를 만들 수 있어서',
            '기울기가 항상 1이라서',
          ],
          answer: 1,
          why: '「이 리뷰가 긍정일 확률 0.87」처럼 바로 해석할 수 있습니다. 은닉층에는 계산이 빠른 ReLU 를 주로 씁니다.',
        },
      ]),
      nextHint('순서를 기억하기로 →', () => ctx.go('context', 'rnn')),
    );

    drawNeuron();
    drawLoss();
    drawGrad();
    runXor(4);
    return wrap;
  },
};

/* ══════════════════════════ 2. 순서를 기억하기 ════════════════════════ */
const rnnScreen = {
  id: 'rnn',
  title: '순서를 기억하기',
  render(ctx) {
    const wrap = h('div');
    const state = { text: '이 카페는 조용하지 않아서 별로였다', kind: 'rnn', decay: 0.85 };
    const out = h('div');

    const draw = () => {
      const words = state.text.split(/\s+/).filter(Boolean).slice(0, 10);
      const inputs = words.map((w) => embed(w).slice(0, 6));
      out.textContent = '';
      if (!inputs.length) return;

      if (state.kind === 'rnn') {
        const { steps } = rnnForward(inputs, { hiddenDim: 4, seed: 3 });
        out.append(
          h('p', mono('h_t = tanh( W_xh · x_t  +  W_hh · h_(t−1) )')),
          h('p', { style: { color: 'var(--dim)' } },
            'h 는 「지금까지 읽은 내용의 요약」입니다. 낱말을 하나 읽을 때마다 ',
            b('앞의 요약 + 지금 낱말'), ' 로 새 요약을 만듭니다.'),
          table(['시점', '읽은 낱말', '기억 h (4칸)'], steps.map((s) => [
            `t=${s.t + 1}`,
            h('b', words[s.t]),
            h('span', ...s.h.map((v) => heatCell(Math.abs(v), 1, fx(v, 2)))),
          ])),
        );
      } else {
        const { steps } = lstmForward(inputs, { hiddenDim: 4, seed: 11 });
        out.append(
          h('p', { style: { color: 'var(--dim)' } },
            'LSTM 은 문 세 개로 정보를 조절합니다. 값이 0에 가까우면 ', b('막고'), ', 1에 가까우면 ', b('통과'), ' 시킵니다.'),
          table(['시점', '낱말', '잊을까 (f)', '받아들일까 (i)', '내보낼까 (o)', '기억 c'], steps.map((s) => [
            `t=${s.t + 1}`,
            h('b', words[s.t]),
            h('span', ...s.f.map((v) => heatCell(v, 1, fx(v, 2)))),
            h('span', ...s.i.map((v) => heatCell(v, 1, fx(v, 2)))),
            h('span', ...s.o.map((v) => heatCell(v, 1, fx(v, 2)))),
            h('span', ...s.c.map((v) => heatCell(Math.abs(v), 2, fx(v, 2)))),
          ]), { compact: true }),
          note('info',
            mono('C_t = f ⊙ C_(t−1) + i ⊙ C̃_t'), ' — ',
            '「앞의 기억을 f 만큼 남기고, 새 정보를 i 만큼 더한다」는 뜻입니다. ',
            'RNN 과 달리 기억이 ', b('곱해지지 않고 더해져'), ' 흘러가는 것이 핵심입니다.'),
        );
      }
      out.appendChild(h('div.honest',
        h('b', '가중치는 학습한 것이 아니라 고정된 무작위 값입니다. '),
        '여기서 볼 것은 「학습 결과」가 아니라 ', b('계산이 시점마다 어떻게 흘러가는지'), ' 입니다.'));
    };

    const decayBox = h('div');
    const drawDecay = () => {
      const data = gradientDecay(state.decay, 25);
      const cv = lineChart({
        height: 190,
        series: [{ name: `한 층 지날 때마다 ×${state.decay.toFixed(2)}`, points: data.map((d) => [d.step, d.g]) }],
        xLabel: '거슬러 올라간 층 수', yMin: 0, yMax: 1,
      });
      onResize(() => cv.redraw());
      decayBox.textContent = '';
      decayBox.append(cv,
        h('p', mono(`20층 뒤: ${fx(Math.pow(state.decay, 20), 6)}`), ' — ',
          h('span', { style: { color: Math.pow(state.decay, 20) < 0.02 ? 'var(--bad)' : 'var(--fg)' } },
            Math.pow(state.decay, 20) < 0.02
              ? '거의 0 입니다. 앞쪽 낱말까지 고칠 신호가 사라졌습니다.'
              : '아직 신호가 남아 있습니다.')),
      );
    };

    wrap.append(
      screenHead('순서를 기억하기', '왼쪽에서 오른쪽으로 읽으며 「지금까지의 요약」을 이어 가는 구조.', '③ 맥락을 읽기'),

      card('순서가 뜻을 바꾼다',
        code([
          '"조용하지 않아서 별로였다"   ← 부정',
          '"조용하지 않아서 좋았어요"   ← 긍정',
          '',
          '낱말을 세기만 하면 두 문장은 거의 같아 보인다.',
          '「않아서」가 무엇에 걸리는지 알려면 순서를 봐야 한다.',
        ].join('\n'), { copy: false }),
      ),

      card('한 걸음씩 따라가기',
        input({ value: state.text, onInput: (v) => { state.text = v; draw(); } }),
        h('div.pills',
          pillGroup([{ label: 'RNN', value: 'rnn' }, { label: 'LSTM (문 세 개)', value: 'lstm' }],
            (v) => { state.kind = v; draw(); }),
        ),
        out,
      ),

      card('RNN 의 병 — 앞쪽이 잊힌다',
        h('p', '기억을 이어 갈 때 매번 어떤 값을 곱합니다. 그 값이 1보다 작으면 ',
          b('층을 지날수록 신호가 0으로 사그라듭니다'), '. 문장 앞쪽 낱말이 뒤쪽에 영향을 주지 못하게 되는 것입니다.'),
        slider({
          label: '한 층마다 곱해지는 값', min: 0.5, max: 1.1, step: 0.01, value: 0.85,
          format: (v) => v.toFixed(2), onInput: (v) => { state.decay = v; drawDecay(); },
        }),
        decayBox,
        note('warn', h('b', '기울기 소실(Vanishing Gradient) '), '이라고 합니다. ',
          '값을 1보다 크게 해 보면 반대로 ', b('폭발'), ' 합니다. 둘 다 학습을 망칩니다.'),
      ),

      card('LSTM 은 어떻게 고쳤나',
        h('p', '「노트에 필기하며 강의를 듣는 학생」으로 생각하면 쉽습니다.'),
        table(['문', '하는 일', '노트 비유'], [
          ['망각 게이트 f', '앞의 기억 중 무엇을 버릴지 정한다', '필요 없어진 부분에 줄을 긋는다'],
          ['입력 게이트 i', '새 정보 중 무엇을 적을지 정한다', '중요한 것만 골라 적는다'],
          ['출력 게이트 o', '기억 중 무엇을 내보낼지 정한다', '시험에 나올 것만 답안지에 옮긴다'],
          ['셀 상태 C', '길게 이어지는 기억 그 자체', '노트 전체'],
        ]),
        note('good', '핵심은 셀 상태가 ', b('곱셈이 아니라 덧셈으로 이어진다'), ' 는 점입니다. ',
          '그래서 앞쪽 정보가 사그라들지 않고 멀리까지 흘러갑니다.'),
      ),

      deepDive('LSTM 으로도 부족했던 두 가지',
        h('p', b('① 느립니다. '), 't=100 을 계산하려면 t=1부터 99까지 순서대로 끝나야 합니다. ',
          'GPU 에 코어가 수천 개 있어도 한 번에 하나씩만 처리하니 소용이 없습니다.'),
        h('p', b('② 멀리 있는 관계는 여전히 흐릿합니다. '),
          '「그 배우가 출연한 작년 개봉작이 정말 재미있었다」에서 「배우」와 「재미있었다」를 잇자면 ',
          '중간의 열 낱말을 모두 거쳐야 합니다. 지날 때마다 조금씩 변형되고 흐려집니다.'),
        h('p', b('그래서 나온 생각 — '), '「순서대로 기억하지 말고, ',
          b('모든 낱말이 서로를 한 번에 쳐다보게'), ' 하면 안 될까?」 ',
          '이것이 다음 화면의 Self-Attention 입니다. 「배우」에서 「재미있었다」까지가 ', b('한 걸음'), ' 이 됩니다.'),
        terms([
          ['GRU', 'LSTM 을 간소화해 문을 두 개로 줄인 것. 파라미터가 적고 빠르며 성능은 비슷할 때가 많다.'],
          ['양방향 LSTM', '왼→오, 오→왼 두 방향으로 읽어 양쪽 문맥을 모두 본다. 다만 글을 만들어 낼 때는 쓸 수 없다(미래를 미리 보게 되므로).'],
        ]),
      ),

      pyBox([
        'import torch.nn as nn',
        '',
        '# 임베딩 → LSTM → 분류',
        'embedding = nn.Embedding(어휘크기, 64, padding_idx=0)',
        'lstm      = nn.LSTM(64, 32, batch_first=True)',
        'classify  = nn.Linear(32, 1)',
        '',
        'x = embedding(정수시퀀스)          # (배치, 길이, 64)',
        'output, (h_n, c_n) = lstm(x)      # h_n = 마지막 시점의 요약',
        '확률 = torch.sigmoid(classify(h_n[-1]))',
      ].join('\n'), '마지막 은닉 상태 h_n 을 「문장 전체의 요약」으로 씁니다'),

      quizBlock('t3/rnn', [
        {
          type: 'choice',
          q: 'RNN 에서 h(hidden state) 는 무엇인가요?',
          options: [
            '다음에 올 낱말',
            '지금까지 읽은 내용의 요약',
            '학습률',
            '문장의 길이',
          ],
          answer: 1,
          why: '사람이 글을 왼쪽부터 읽으며 뜻을 쌓아 가는 것과 같습니다. 매 시점 h 가 갱신됩니다.',
        },
        {
          type: 'choice',
          q: 'LSTM 이 긴 문장에 강한 결정적인 이유는?',
          options: [
            '층이 더 많아서',
            '셀 상태가 곱해지지 않고 더해지며 흘러가서',
            '학습률이 커서',
            '단어를 거꾸로 읽어서',
          ],
          answer: 1,
          why: '곱셈이 반복되면 0으로 사그라듭니다. 덧셈으로 이어지는 통로를 따로 두어 먼 기억이 살아남습니다.',
        },
        {
          type: 'multi',
          q: 'LSTM 에 여전히 남아 있던 한계를 모두 고르세요.',
          options: [
            '순서대로 계산해야 해서 느리다',
            '아주 멀리 떨어진 낱말의 관계는 흐릿하다',
            '단어를 숫자로 바꿀 수 없다',
            '문장 길이가 다르면 쓸 수 없다',
          ],
          answer: [0, 1],
          why: '임베딩으로 숫자화는 잘 되고 길이가 달라도 처리됩니다. 문제는 속도와 장거리 관계였고, 그것을 Attention 이 풀었습니다.',
        },
      ]),
      nextHint('Attention 계산기로 →', () => ctx.go('context', 'attention')),
    );
    draw();
    drawDecay();
    return wrap;
  },
};

/* ══════════════════════════ 3. Attention 계산기 ═══════════════════════ */
const attnScreen = {
  id: 'attention',
  title: 'Attention 계산기',
  render(ctx) {
    const wrap = h('div');
    const state = { text: '카페 커피 향 진하다', dk: 4, mask: false, seed: 5, step: 'weights' };
    const out = h('div');
    const inp = input({ value: state.text, onInput: (v) => { state.text = v; draw(); } });

    const draw = () => {
      const toks = state.text.split(/\s+/).filter(Boolean).slice(0, 7);
      out.textContent = '';
      if (toks.length < 2) { out.appendChild(note('warn', '낱말을 두 개 이상 넣어 주세요.')); return; }
      const X = toks.map((t) => embed(t).slice(0, 8));
      const mask = state.mask ? causalMask(toks.length) : null;
      const A = selfAttention(X, { dk: state.dk, seed: state.seed, mask });

      const mk = (m, fmt, max) => {
        const hm = heatmap({
          matrix: m, rowLabels: toks, colLabels: toks, fmt, cellMax: max,
          height: 60 + toks.length * 32,
        });
        onResize(() => hm.redraw());
        return hm;
      };

      out.append(
        h('div.flow',
          ...[['raw', '① Q·Kᵀ 유사도'], ['scaled', `② ÷ √${state.dk}`], ['weights', '③ softmax'], ['out', '④ × V']]
            .map(([k, label], i, arr) => [
              h('button.flow-step' + (state.step === k ? '.on' : ''), {
                type: 'button', onclick: () => { state.step = k; draw(); },
              }, label),
              i < arr.length - 1 ? h('span.flow-arrow', '→') : null,
            ]).flat(),
        ),
      );

      if (state.step === 'raw') {
        out.append(
          h('p', b('① Query 와 Key 의 내적 '), h('span', { style: { color: 'var(--dim)' } },
            '— 「내가 찾는 것」과 「네가 가진 것」이 얼마나 맞는지 점수를 낸다')),
          mk(A.raw, (v) => v.toFixed(2), Math.max(...A.raw.flat().map(Math.abs))),
        );
      } else if (state.step === 'scaled') {
        out.append(
          h('p', b(`② √d_k = √${state.dk} = ${fx(Math.sqrt(state.dk), 2)} 로 나누기 `),
            h('span', { style: { color: 'var(--dim)' } }, '— 값이 너무 커지면 softmax 가 한 곳에만 몰려 학습이 막힌다')),
          mk(A.scaled, (v) => v.toFixed(2), Math.max(...A.scaled.flat().map(Math.abs))),
        );
      } else if (state.step === 'weights') {
        out.append(
          h('p', b('③ softmax — 줄마다 합이 1인 확률로 '),
            h('span', { style: { color: 'var(--dim)' } }, '— 「이 낱말을 이해하려면 어디를 얼마나 볼까」')),
          mk(A.weights, (v) => (v < 0.005 ? '' : v.toFixed(2)), 1),
          h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
            '가로 한 줄을 읽으세요. ', mono(toks[0]), ' 줄의 값들은 ', mono(toks[0]),
            ' 가 각 낱말에 얼마나 주목하는지를 나타내며, 모두 더하면 1 입니다.'),
          ...toks.map((t, i) => h('div', { style: { margin: '6px 0' } },
            h('span', { style: { display: 'inline-block', width: '92px', fontWeight: 700 } }, t, ' →'),
            h('span.tokens', { style: { display: 'inline-flex' } },
              ...A.weights[i].map((w, j) => h('span.tok', {
                style: { opacity: String(0.35 + w * 0.65), borderColor: w > 0.3 ? 'var(--accent)' : '' },
              }, toks[j], h('span.pos', pct(w, 0)))),
            ),
          )),
        );
      } else {
        out.append(
          h('p', b('④ 확률로 Value 를 섞기 '), h('span', { style: { color: 'var(--dim)' } },
            '— 결과는 「주변 정보가 스며든 새 벡터」')),
          table(['낱말', '들어온 벡터 (앞 4칸)', '나간 벡터 (문맥이 섞임)'], toks.map((t, i) => [
            h('b', t),
            h('span', ...X[i].slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 2)))),
            h('span', ...A.out[i].slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 2)))),
          ])),
          note('good', '들어올 때는 낱말마다 ', b('따로 떨어진'), ' 벡터였는데, ',
            '나갈 때는 ', b('주변 낱말의 정보가 섞인'), ' 벡터가 되었습니다. 이것이 「문맥을 반영한 표현」입니다.'),
        );
      }
    };

    wrap.append(
      screenHead('Self-Attention 계산기', '「이 낱말을 이해하려면 어디를 봐야 할까?」를 스스로 계산합니다.', '③ 맥락을 읽기'),

      card('도서관에서 책 찾기로 비유하면',
        table(['이름', '무엇인가', '도서관 비유'], [
          [h('b', 'Query (Q)'), '내가 찾는 것', '검색창에 친 검색어'],
          [h('b', 'Key (K)'), '내가 가진 것의 이름표', '책 제목·색인어'],
          [h('b', 'Value (V)'), '실제 내용', '책의 본문'],
        ]),
        h('p', b('검색 순서 — '), '검색어(Q)와 모든 이름표(K)를 맞춰 점수를 내고, 그 점수를 확률로 바꾼 뒤, ',
          '확률만큼 본문(V)을 섞어 가져옵니다.'),
        h('p', { style: { textAlign: 'center', fontSize: '1.05rem', margin: '14px 0' } },
          mono('Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V')),
        note('info', h('b', '왜 「Self」인가 — '),
          'Q·K·V 가 모두 ', b('같은 문장'), ' 에서 나오기 때문입니다. 문장이 자기 자신을 들여다보는 것입니다.'),
      ),

      card('직접 계산해 보기',
        h('div.pills', ...[
          '카페 커피 향 진하다',
          '이 카페는 조용하지 않아서 좋았다',
          '도서관 에서 빌린 책 을 반납 했다',
        ].map((t) => h('button.pill', { type: 'button', onclick: () => { state.text = t; inp.value = t; draw(); } }, t))),
        inp,
        out,
      ),

      card('설정 바꿔 보기',
        slider({ label: 'Q·K 벡터 차원 (d_k)', min: 2, max: 12, value: 4, onInput: (v) => { state.dk = v; draw(); } }),
        slider({ label: '가중치 초기값 씨앗', min: 1, max: 40, value: 5, onInput: (v) => { state.seed = v; draw(); } }),
        toggle('미래를 가리기 (인과 마스크 — GPT 방식)', false, (v) => { state.mask = v; draw(); }),
        h('div.honest',
          h('b', '가중치 행렬은 학습한 것이 아니라 고정된 무작위 값입니다. '),
          '진짜 모형은 이 행렬을 수십억 개의 문장으로 학습합니다. ',
          '여기서 확인할 것은 ', b('계산의 모양'), ' — 줄마다 합이 1이 되는지, 마스크를 켜면 오른쪽이 0이 되는지입니다.'),
      ),

      card('마스크를 켜면 무엇이 달라지나',
        h('p', '미래를 가리면 각 낱말은 ', b('자기 자신과 왼쪽'), ' 만 볼 수 있습니다. ',
          '표의 오른쪽 위 삼각형이 통째로 0이 됩니다.'),
        table(['질문자 \\ 참조 대상', '나는', '카페에', '갔다'], [
          [h('b', '나는'), '✅', '❌', '❌'],
          [h('b', '카페에'), '✅', '✅', '❌'],
          [h('b', '갔다'), '✅', '✅', '✅'],
        ], { compact: true }),
        terms([
          ['마스크 없음 (양방향)', '앞뒤를 모두 본다 → 글을 이해하는 데 강하다 → BERT 계열'],
          ['마스크 있음 (단방향)', '왼쪽만 본다 → 다음 낱말을 만들어 낼 수 있다 → GPT 계열'],
        ]),
        note('info', '위 스위치를 켜고 ③ softmax 표를 보세요. 첫 줄은 자기 자신에만 100%, ',
          '마지막 줄만 모든 낱말을 볼 수 있게 됩니다.'),
      ),

      deepDive('왜 √d_k 로 나누는가',
        h('p', '내적은 차원이 커질수록 값이 커집니다. 차원이 64면 값이 수십까지 올라갑니다. ',
          '그 상태로 softmax 를 씌우면 어떻게 될까요?'),
        code([
          'softmax([2, 1, 0])      → [0.67, 0.24, 0.09]   골고루',
          'softmax([20, 10, 0])    → [1.00, 0.00, 0.00]   한 곳에 몰림',
          '',
          '한 곳에 몰리면 나머지 낱말의 기울기가 0 이 되어 학습이 멈춘다.',
        ].join('\n'), { copy: false }),
        h('p', '그래서 ', mono('√d_k'), ' 로 나눠 값을 알맞은 크기로 눌러 줍니다. ',
          '위에서 d_k 를 12까지 올린 뒤 ①과 ②를 비교해 보세요.'),
        h('p', b('계산량 — '), '문장 길이가 n 이면 모든 낱말 쌍을 봐야 하므로 ', mono('n²'), ' 에 비례합니다. ',
          '길이가 2배면 계산이 4배입니다. 아주 긴 글에서 메모리가 부족해지는 이유이고, ',
          '이를 줄이려는 연구가 지금도 활발합니다.'),
      ),

      quizBlock('t3/attention', [
        {
          type: 'choice',
          q: 'Self-Attention 에서 softmax 를 거친 뒤 한 줄의 값을 모두 더하면?',
          options: ['0', '1', '낱말 개수', '정해져 있지 않다'],
          answer: 1,
          why: 'softmax 는 점수를 확률로 바꿉니다. 「어디를 얼마나 볼까」의 비율이므로 합이 1입니다.',
        },
        {
          type: 'choice',
          q: 'Query·Key·Value 를 도서관에 비유하면 Value 는 무엇인가요?',
          options: ['검색어', '책 제목', '책의 본문', '사서'],
          answer: 2,
          why: 'Q는 검색어, K는 제목(이름표), V는 실제 내용입니다. 점수는 Q와 K로 내고, 섞어 오는 것은 V입니다.',
        },
        {
          type: 'choice',
          q: 'LSTM 과 비교해 Self-Attention 의 결정적 장점은?',
          options: [
            '파라미터가 적다',
            '모든 낱말이 서로를 한 걸음에 참조하고, 동시에 계산할 수 있다',
            '문장이 짧아진다',
            '학습 자료가 적어도 된다',
          ],
          answer: 1,
          why: 'LSTM 은 t번째를 계산하려면 앞을 다 끝내야 했습니다. Attention 은 한 번에 계산하므로 GPU 를 제대로 씁니다.',
        },
        {
          type: 'choice',
          q: '인과 마스크(미래 가리기)를 씌우는 모형은?',
          options: ['BERT 계열', 'GPT 계열', 'Word2Vec', 'TF-IDF'],
          answer: 1,
          why: '글을 만들어 내려면 아직 쓰지 않은 뒷부분을 볼 수 없어야 합니다. 그래서 GPT 는 왼쪽만 봅니다.',
        },
      ]),
      nextHint('인코더 블록으로 →', () => ctx.go('context', 'transformer')),
    );

    draw();
    return wrap;
  },
};

/* ══════════════════════════ 4. 인코더 블록 ════════════════════════════ */
const encoderScreen = {
  id: 'transformer',
  title: '인코더 블록',
  render(ctx) {
    const wrap = h('div');

    // 위치 인코딩 히트맵
    const pe = positionalEncoding(12, 16);
    const peMap = heatmap({
      matrix: pe, rowLabels: pe.map((_, i) => `위치 ${i}`), colLabels: pe[0].map((_, i) => String(i)),
      cellMax: 1, fmt: () => '', height: 300,
      colorOf: (v) => `rgba(${v > 0 ? '96,165,250' : '244,114,182'},${(Math.abs(v) * 0.9 + 0.08).toFixed(2)})`,
    });
    onResize(() => peMap.redraw());

    // 순서를 바꿔도 결과가 같은지 실험
    const permBox = h('div');
    const runPerm = () => {
      const a = ['나는', '카페에', '갔다'];
      const bb = ['카페에', '나는', '갔다'];
      const Xa = a.map((t) => embed(t).slice(0, 8));
      const Xb = bb.map((t) => embed(t).slice(0, 8));
      const A = selfAttention(Xa, { dk: 4, seed: 9 });
      const B = selfAttention(Xb, { dk: 4, seed: 9 });
      // 「갔다」는 두 문장 모두 마지막 자리
      const va = A.out[2];
      const vb = B.out[2];
      const same = va.every((x, i) => Math.abs(x - vb[i]) < 1e-9);

      const withPE = (X) => X.map((row, i) => row.map((v, j) => v + positionalEncoding(X.length, X[0].length)[i][j] * 0.5));
      const A2 = selfAttention(withPE(Xa), { dk: 4, seed: 9 });
      const B2 = selfAttention(withPE(Xb), { dk: 4, seed: 9 });
      const same2 = A2.out[2].every((x, i) => Math.abs(x - B2.out[2][i]) < 1e-9);

      permBox.textContent = '';
      permBox.append(
        table(['문장', '「갔다」가 얻은 벡터 (앞 4칸)'], [
          [mono(a.join(' ')), h('span', ...va.slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 3))))],
          [mono(bb.join(' ')), h('span', ...vb.slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 3))))],
        ]),
        same
          ? note('bad', h('b', '두 값이 완전히 같습니다. '),
            'Attention 은 낱말의 ', b('자리'), ' 를 아예 보지 않기 때문입니다. ',
            '「나는 카페에 갔다」와 「카페에 나는 갔다」를 구별하지 못합니다.')
          : note('info', '값이 다릅니다.'),
        h('p', b('위치 정보를 더한 뒤 다시 계산하면')),
        table(['문장', '「갔다」가 얻은 벡터 (앞 4칸)'], [
          [mono(a.join(' ')), h('span', ...A2.out[2].slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 3))))],
          [mono(bb.join(' ')), h('span', ...B2.out[2].slice(0, 4).map((v) => heatCell(Math.abs(v), 1, fx(v, 3))))],
        ]),
        same2
          ? note('warn', '아직 같습니다.')
          : note('good', h('b', '이제 값이 달라졌습니다. '),
            '낱말 벡터에 「몇 번째 자리인가」를 더해 주었더니 순서를 구별하게 되었습니다.'),
      );
    };

    // 잔차 연결 & 층 정규화 예시
    const v = embed('커피').slice(0, 8);
    const attn = selfAttention([v, embed('향').slice(0, 8)], { dk: 4, seed: 3 }).out[0];
    const res = v.map((x, i) => x + attn[i]);
    const normed = layerNorm(res);

    wrap.append(
      screenHead('인코더 블록', 'Attention 하나만으로는 부족합니다. 자리 정보·잔차·작은 신경망을 함께 얹습니다.', '③ 맥락을 읽기'),

      card('블록 하나의 생김새',
        code([
          '     입력 임베딩  +  위치 인코딩',
          '              ↓',
          '  ┌─────────────────────────────┐',
          '  │ ① 여러 머리 Attention        │  ← 어디를 볼까',
          '  │ ② 원본 더하기 + 층 정규화     │  ← 안전장치',
          '  │                             │',
          '  │ ③ 자리별 작은 신경망(FFN)     │  ← 본 것을 어떻게 해석할까',
          '  │ ④ 원본 더하기 + 층 정규화     │',
          '  └─────────────────────────────┘',
          '              ↓  (이 블록을 N번 반복)',
          '           분류층 또는 다음 낱말 예측',
        ].join('\n'), { copy: false }),
        table(['부품', '하는 일', '왜 필요한가'], [
          ['여러 머리 Attention', '서로 다른 관점으로 동시에 본다', '한 관점만으로는 문법 관계·수식 관계·감정 관계를 한꺼번에 못 본다'],
          ['원본 더하기 (잔차 연결)', '변환 결과에 원래 입력을 더한다', '층을 깊이 쌓아도 신호가 사그라들지 않는다. 「수정본을 만들되 원본도 남긴다」'],
          ['층 정규화', '값의 평균 0, 표준편차 1 로 맞춘다', '값이 들쭉날쭉하면 학습이 흔들린다'],
          ['자리별 신경망 (FFN)', '각 자리에서 따로 비선형 변환', 'Attention 은 「어디를 볼까」만 정한다. 「본 것을 어떻게 쓸까」는 여기서'],
        ]),
      ),

      card('① 위치 인코딩 — 자리 번호 붙이기',
        h('p', 'Attention 은 모든 낱말을 ', b('동시에'), ' 봅니다. 편리하지만 대가가 있습니다 — ',
          b('순서를 모릅니다'), '. 직접 확인해 봅시다.'),
        button('순서를 바꿔 실험하기', runPerm),
        permBox,
        h('p', b('사인·코사인으로 만든 자리 무늬')),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '가로가 벡터의 칸, 세로가 자리(위치)입니다. 자리마다 무늬가 다릅니다. ',
          '이 무늬를 낱말 벡터에 ', b('더해'), ' 「몇 번째인가」를 알려 줍니다.'),
        peMap,
        code([
          'PE(자리, 2i)   = sin( 자리 / 10000^(2i/차원) )',
          'PE(자리, 2i+1) = cos( 자리 / 10000^(2i/차원) )',
          '',
          '짝수 칸은 sin, 홀수 칸은 cos.',
          '학습하지 않은 값이라 훈련 때보다 긴 문장에도 그대로 쓸 수 있다.',
        ].join('\n'), { copy: false }),
      ),

      card('② 원본 더하기와 층 정규화',
        table(['단계', '값 (앞 6칸)'], [
          ['원래 입력 x', h('span', ...v.slice(0, 6).map((x) => heatCell(Math.abs(x), 1, fx(x, 2))))],
          ['Attention 결과', h('span', ...attn.slice(0, 6).map((x) => heatCell(Math.abs(x), 1, fx(x, 2))))],
          ['x + Attention (잔차)', h('span', ...res.slice(0, 6).map((x) => heatCell(Math.abs(x), 2, fx(x, 2))))],
          ['층 정규화까지', h('span', ...normed.slice(0, 6).map((x) => heatCell(Math.abs(x), 2, fx(x, 2))))],
        ]),
        note('info', mono('출력 = 층정규화( x + Attention(x) )'), ' — ',
          '변환이 잘못돼도 원본 x 가 남아 있어 학습이 무너지지 않습니다. ',
          '이 아이디어 하나로 층을 12층, 96층까지 쌓을 수 있게 되었습니다.'),
      ),

      card('③ 머리를 여러 개 두는 까닭',
        code([
          '"그 배우가 출연한 액션 영화가 재미있었다"',
          '',
          '머리 1 (누가-무엇을):  배우가  ↔  출연한',
          '머리 2 (꾸미는 관계):  액션    ↔  영화가',
          '머리 3 (느낌):        영화가  ↔  재미있었다',
          '머리 4 (가리키는 말):  그      ↔  배우가',
        ].join('\n'), { copy: false }),
        h('p', '머리 하나로는 한 가지 관계만 잡힙니다. 그래서 차원을 나눠 여러 머리에 맡기고, ',
          '결과를 이어 붙입니다. ', mono('d_model=64, 머리 4개 → 머리마다 16차원'), '. ',
          '전체 계산량은 머리 하나일 때와 비슷합니다.'),
      ),

      card('갈림길 — 이해하는 쪽과 만들어 내는 쪽',
        code([
          '            Transformer (2017)',
          '           /                  \\',
          '   인코더만 쓰기            디코더만 쓰기',
          '   (앞뒤 모두 봄)          (왼쪽만 봄, 마스크)',
          '        ↓                        ↓',
          '   BERT 계열                 GPT 계열',
          '   분류·검색·정보 추출        대화·요약·글쓰기',
        ].join('\n'), { copy: false }),
        table(['', '인코더 (BERT 계열)', '디코더 (GPT 계열)'], [
          ['보는 방향', '양쪽 모두', '왼쪽만'],
          ['학습 방법', '빈칸 맞히기', '다음 낱말 맞히기'],
          ['잘하는 일', '문장 이해·분류·질의응답', '글 만들기·대화·요약'],
          ['문장 요약 방법', '특별한 토큰 하나를 씀', '마지막 자리의 출력을 씀'],
        ]),
        note('info', '요즘 널리 쓰는 도구(대화형 AI)는 대부분 ', b('디코더 계열'), ' 입니다. ',
          '모형이 충분히 커지자 분류 같은 일도 「물어보기만 해도」 잘하게 되었기 때문입니다. → 탭 ④'),
      ),

      deepDive('블록을 몇 층이나 쌓나',
        h('p', '처음 논문은 6층이었고, 이후 모형들은 12층·24층·96층까지 늘렸습니다. ',
          '층을 쌓으면 아래층은 「이 낱말이 무엇인가」 같은 가까운 관계를, ',
          '위층은 「이 문단의 주장이 무엇인가」 같은 먼 관계를 다루게 됩니다.'),
        h('p', b('그냥 쌓기만 하면 안 됩니다. '),
          '잔차 연결과 층 정규화가 없으면 깊은 신경망은 학습이 되지 않습니다. ',
          '「Attention 만 있으면 된다」는 제목과 달리, 실제로는 이 안전장치들이 함께 있어야 굴러갑니다.'),
        h('p', b('FFN 의 크기 — '), '관례적으로 안쪽을 d_model 의 4배로 넓혔다가 다시 줄입니다 ',
          '(64 → 256 → 64). 넓혔다 줄이는 사이에 비선형(ReLU)이 들어가면서 표현이 풍부해집니다. ',
          '파라미터의 상당수가 사실 이 FFN 에 있습니다.'),
      ),

      quizBlock('t3/transformer', [
        {
          type: 'choice',
          q: '위치 인코딩이 필요한 까닭은?',
          options: [
            'Attention 계산이 너무 느려서',
            'Attention 은 낱말의 순서를 아예 보지 않아서',
            '문장 길이를 맞추려고',
            '벡터 값이 너무 커져서',
          ],
          answer: 1,
          why: '모든 낱말을 동시에 보는 대신 순서 정보를 잃습니다. 그래서 자리마다 다른 무늬를 더해 알려 줍니다.',
        },
        {
          type: 'choice',
          q: '잔차 연결(원본 더하기)이 하는 일은?',
          options: [
            '계산량을 줄인다',
            '층을 깊게 쌓아도 신호가 사그라들지 않게 한다',
            '문장을 짧게 만든다',
            '확률의 합을 1로 만든다',
          ],
          answer: 1,
          why: '변환 결과에 원본을 더해 두면 기울기가 잘 흘러갑니다. 「수정본을 만들되 원본도 남긴다」고 생각하면 쉽습니다.',
        },
        {
          type: 'choice',
          q: '분류·검색에 강한 쪽과 글쓰기에 강한 쪽을 바르게 짝지은 것은?',
          options: [
            '인코더 = 글쓰기 / 디코더 = 분류',
            '인코더 = 분류 / 디코더 = 글쓰기',
            '둘 다 분류에만 쓴다',
            '둘 다 글쓰기에만 쓴다',
          ],
          answer: 1,
          why: '앞뒤를 다 보는 인코더는 이해에, 왼쪽만 보며 다음을 만들어 내는 디코더는 생성에 강합니다.',
        },
        {
          type: 'short',
          q: () => ['머리를 네 개 두고 d_model 이 64 라면, 머리 하나가 맡는 차원은 몇일까요? (숫자만)'],
          accept: ['16'],
          why: '64 ÷ 4 = 16 입니다. 차원을 나눠 맡기므로 전체 계산량은 머리 하나일 때와 비슷합니다.',
        },
      ]),
      nextHint('④ 말을 만들기로 →', () => ctx.go('llm', 'next')),
    );

    runPerm();
    return wrap;
  },
};

export default {
  id: 'context',
  num: 'Ⅲ',
  title: '맥락을 읽기',
  screens: [nnScreen, rnnScreen, attnScreen, encoderScreen],
};
