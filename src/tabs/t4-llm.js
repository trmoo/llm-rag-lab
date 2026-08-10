/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ④ 말을 만들기 — 다음 낱말을 고르는 모델, 그리고 그 모델과 대화하기
 *   1. 다음 낱말 예측기   확률 분포에서 하나를 고르는 네 가지 방법
 *   2. 토큰과 컨텍스트 창  모델이 한 번에 볼 수 있는 양과 비용
 *   3. 프롬프트 실험실     같은 일을 시키는 네 가지 말투
 *   4. 파이프라인 조립     프롬프트 | 모델 | 파서
 *   5. 기억하는 챗봇       모델에는 기억이 없다. 그럼 어떻게 기억하나
 * ========================================================================== */

import {
  h, card, note, deepDive, terms, table, screenHead, button, input, textarea,
  toggle, pillGroup, code, pyBox, mono, b, nextHint, fx, pct, slider, bar, onResize,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import { trainNgram, sample, generate, tokenize, countTokens, BOS, EOS } from '../lib/lm.js';
import { LM_SENTENCES, POSITIVE_REVIEWS, NEGATIVE_REVIEWS } from '../data/corpus.js';
import { makeVectorizer, logisticRegression, preprocess } from '../lib/nlp.js';
import { barChart, lineChart, PALETTE } from '../lib/chart.js';

const statBox = (k, v, kind = '') => h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));

/* 이 탭에서 함께 쓰는 작은 모델들 (한 번만 만든다) */
let NGRAM = null;
const getNgram = () => {
  if (!NGRAM) NGRAM = trainNgram(LM_SENTENCES.map((s) => s.split(' ')));
  return NGRAM;
};

let SENTI = null;
const getSentiment = () => {
  if (SENTI) return SENTI;
  const X = [...POSITIVE_REVIEWS, ...NEGATIVE_REVIEWS];
  const y = [...POSITIVE_REVIEWS.map(() => 1), ...NEGATIVE_REVIEWS.map(() => 0)];
  const vec = makeVectorizer({ min: 2, max: 4, minDf: 2 }).fit(X);
  const lr = logisticRegression(vec.transform(X), y, { epochs: 400, lr: 3, l2: 0.004 });
  SENTI = { vec, lr, judge: (t) => lr.predictProba(vec.transform([t]))[0] };
  return SENTI;
};

/* ══════════════════════════ 1. 다음 낱말 예측기 ═══════════════════════ */
const nextScreen = {
  id: 'next',
  title: '다음 낱말 예측기',
  render(ctx) {
    const wrap = h('div');
    const M = getNgram();
    const state = { words: ['도서관', '열람실은'], mode: 'topk', temperature: 0.9, k: 3, p: 0.9 };
    const out = h('div');

    const draw = () => {
      const { list, source } = M.distribution(state.words);
      const { pool } = sample(list, state, () => 0.5);
      const shown = pool.slice(0, 10);
      out.textContent = '';
      out.append(
        h('p', b('지금까지의 글 '), mono(state.words.join(' ') || '(비어 있음)'),
          h('span', { style: { color: 'var(--dim)', marginLeft: '10px', fontSize: '.85rem' } }, `(${source} 을 보고 셌습니다)`)),
        h('p', b('다음에 올 낱말의 확률')),
        ...shown.map((x) => h('div', {
          style: { display: 'flex', gap: '8px', alignItems: 'center', opacity: x.keep === false ? '.32' : '1' },
        },
        h('button.pill', {
          type: 'button',
          style: { width: '124px', textAlign: 'left' },
          onclick: () => { if (x.word !== EOS) { state.words.push(x.word); draw(); } },
        }, x.word),
        bar(x.q !== undefined ? x.q : x.p, {
          label: `원래 ${pct(x.p, 0)}${x.q !== undefined && state.mode !== 'greedy' ? ` → 고른 뒤 ${pct(x.q, 0)}` : ''}`,
          color: x.keep === false ? '#3b4a68' : undefined,
        }),
        )),
        shown.some((x) => x.keep === false)
          ? h('p', { style: { color: 'var(--dim)', fontSize: '.88rem' } },
            '흐린 줄은 이번 방법에서 ', b('아예 후보에서 빠진'), ' 낱말입니다.')
          : null,
        h('div.pills',
          button('무작위로 한 낱말 뽑기', () => {
            const r = sample(list, state, Math.random);
            if (r.picked !== EOS) state.words.push(r.picked);
            draw();
          }),
          button('한 낱말 되돌리기', () => { state.words.pop(); draw(); }, 'ghost'),
          button('처음부터', () => { state.words = []; draw(); }, 'ghost'),
        ),
      );
    };

    const genBox = h('div');
    const runGen = () => {
      genBox.textContent = '';
      const rows = [
        { label: '항상 1등만 (greedy)', opt: { mode: 'greedy' } },
        { label: '온도 0.3 — 안전하게', opt: { mode: 'temperature', temperature: 0.3 } },
        { label: '온도 1.0 — 보통', opt: { mode: 'temperature', temperature: 1.0 } },
        { label: '온도 1.8 — 자유롭게', opt: { mode: 'temperature', temperature: 1.8 } },
        { label: '상위 2개에서만 (top-k=2)', opt: { mode: 'topk', k: 2, temperature: 1 } },
        { label: '누적 확률 0.9까지 (top-p)', opt: { mode: 'topp', p: 0.9, temperature: 1 } },
      ];
      genBox.appendChild(table(['고르는 방법', '만들어진 문장 (같은 시작에서)'],
        rows.map((r, i) => [
          r.label,
          mono(generate(M, ['오늘'], { ...r.opt, seed: 20 + i, maxLen: 14 }).words.join(' ')),
        ])));
      genBox.appendChild(note('info',
        '모두 ', mono('오늘'), ' 에서 시작했는데 결과가 다릅니다. ',
        b('모델은 하나인데 뽑는 방법이 다르기 때문'), ' 입니다. ',
        '온도가 낮으면 늘 비슷한 말을, 높으면 뜻밖의 말을 하게 됩니다.'));
    };

    wrap.append(
      screenHead('다음 낱말 예측기', '거대 언어 모델이 하는 일은 결국 이것입니다 — 다음에 올 낱말의 확률 내놓기.', '④ 말을 만들기'),

      card('이것이 전부입니다',
        code([
          '입력:  "오늘 도서관에"',
          '출력:  가서 42%   갔다 28%   들렀다 17%   ...',
          '        ↓ 하나를 고른다',
          '입력:  "오늘 도서관에 가서"    ← 고른 낱말을 다시 붙여 넣고',
          '출력:  책을 55%   공부를 30%  ...',
          '        ↓ 되풀이',
          '"오늘 도서관에 가서 책을 빌렸다"',
        ].join('\n'), { copy: false }),
        h('p', '이렇게 자기가 만든 것을 다시 입력에 넣는 방식을 ', b('자기회귀(Autoregressive)'), ' 라고 합니다. ',
          '문장 전체를 한 번에 만들어 내는 것이 아닙니다.'),
        h('div.honest',
          h('b', '여기서 도는 모델은 문장 50개로 만든 아주 작은 것입니다. '),
          '앞의 두 낱말을 보고 다음 낱말이 몇 번 나왔는지 세었을 뿐입니다(n-gram). ',
          '실제 대형 모델은 인터넷 규모의 글로 학습하고 파라미터가 수천억 개입니다. ',
          '규모는 하늘과 땅 차이지만 ', b('「확률 분포를 내놓고 거기서 하나를 고른다」는 뼈대는 똑같습니다'), '. ',
          '그래서 여기서 온도를 올리면 진짜로 엉뚱해집니다.'),
      ),

      card('한 낱말씩 직접 골라 보기',
        h('div.pills', ...['도서관 열람실은', '오늘', '노트 종이가', '문서를'].map((t) => h('button.pill', {
          type: 'button', onclick: () => { state.words = t.split(' '); draw(); },
        }, mono(t)))),
        h('div', { style: { margin: '10px 0' } },
          pillGroup([
            { label: '항상 1등 (greedy)', value: 'greedy' },
            { label: '온도로 조절', value: 'temperature' },
            { label: '상위 k개', value: 'topk' },
            { label: '누적 확률 p', value: 'topp' },
          ], (v) => { state.mode = v; draw(); }, 'topk'),
        ),
        slider({ label: '온도 (temperature)', min: 0.1, max: 2, step: 0.05, value: 0.9, format: (v) => v.toFixed(2), onInput: (v) => { state.temperature = v; draw(); } }),
        slider({ label: '상위 몇 개까지 (k)', min: 1, max: 8, value: 3, onInput: (v) => { state.k = v; draw(); } }),
        slider({ label: '누적 확률 (p)', min: 0.1, max: 1, step: 0.05, value: 0.9, format: (v) => v.toFixed(2), onInput: (v) => { state.p = v; draw(); } }),
        out,
      ),

      card('네 가지 방법을 나란히',
        button('여섯 가지 설정으로 만들어 보기', runGen),
        genBox,
        table(['방법', '무엇을 하나', '느낌'], [
          [h('b', 'greedy'), '언제나 확률 1등만 고른다', '안전하지만 늘 같은 말을 한다'],
          [h('b', '온도'), '분포를 뾰족하게(낮음) 또는 평평하게(높음) 만든다', '0.2 정확 / 0.7 균형 / 1.2 이상 자유분방'],
          [h('b', 'top-k'), '상위 k개만 남기고 나머지는 버린 뒤 뽑는다', '엉뚱한 낱말이 나올 길을 원천 차단'],
          [h('b', 'top-p'), '확률을 더해 p 가 될 때까지만 남긴다', '상황에 따라 후보 수가 저절로 조절된다'],
        ]),
        note('info', h('b', '실무 권장값 — '),
          '코드 생성 ', mono('온도 0.2'), ' / 일반 대화 ', mono('온도 0.7'), ' / 창작 ', mono('온도 1.0'),
          '. 문서에 근거해 답해야 하는 RAG(탭 ⑤)에서는 ', mono('온도 0'), ' 을 씁니다 — 지어내면 안 되니까요.'),
      ),

      deepDive('온도는 수식에서 어디에 들어가나',
        code([
          'softmax(점수 / T)',
          '',
          '점수 = [3, 2, 1] 일 때',
          '  T = 0.3  →  [0.96, 0.03, 0.00]   뾰족  (1등 독차지)',
          '  T = 1.0  →  [0.67, 0.24, 0.09]   보통',
          '  T = 2.0  →  [0.51, 0.31, 0.19]   평평  (골고루)',
        ].join('\n'), { copy: false }),
        h('p', '점수를 T 로 나누기만 합니다. T 가 작으면 점수 차이가 벌어져 1등이 독차지하고, ',
          'T 가 크면 차이가 줄어 골고루 뽑힙니다. ', mono('T → 0'), ' 이면 greedy 와 같아집니다.'),
        h('p', b('왜 굳이 무작위로 뽑나 — '), '늘 1등만 고르면 ',
          '「그리고 그리고 그리고…」처럼 같은 말을 되풀이하는 함정에 빠지기 쉽습니다. ',
          '적당한 무작위가 오히려 자연스러운 글을 만듭니다.'),
        h('p', b('n-gram 모델의 한계 — '), '이 화면의 모델은 앞의 ', b('두 낱말'), ' 만 봅니다. ',
          '그래서 문장이 길어지면 앞뒤가 안 맞습니다. 대형 모델은 앞의 수천~수만 낱말을 ',
          'Attention 으로 한꺼번에 보기 때문에 훨씬 긴 맥락을 지킵니다. 그것이 탭 ③에서 배운 기술입니다.'),
      ),

      quizBlock('t4/next', [
        {
          type: 'choice',
          q: '온도(temperature)를 0에 가깝게 낮추면?',
          options: [
            '결과가 다양해진다',
            '확률이 가장 높은 낱말만 거의 늘 뽑힌다',
            '문장이 길어진다',
            '모델이 다시 학습된다',
          ],
          answer: 1,
          why: '분포가 뾰족해져 1등이 독차지합니다. 사실상 greedy 와 같아집니다.',
        },
        {
          type: 'choice',
          q: 'top-p (누적 확률) 방식이 top-k 와 다른 점은?',
          options: [
            '항상 더 많은 후보를 남긴다',
            '상황에 따라 남는 후보의 개수가 달라진다',
            '확률을 쓰지 않는다',
            '온도와 함께 쓸 수 없다',
          ],
          answer: 1,
          why: '1등이 90%면 후보가 하나만 남고, 고만고만하면 여러 개가 남습니다. 분포 모양에 맞춰 저절로 조절됩니다.',
        },
        {
          type: 'choice',
          q: '문서에 근거해 정확히 답해야 하는 챗봇의 온도는 어떻게 두는 것이 좋을까요?',
          options: ['0 또는 아주 낮게', '1.0 근처', '2.0 이상', '온도는 상관없다'],
          answer: 0,
          why: '사실 전달에는 창의성보다 일관성이 중요합니다. 온도를 올리면 근거에서 벗어날 위험이 커집니다.',
        },
      ]),
      nextHint('토큰과 컨텍스트 창으로 →', () => ctx.go('llm', 'tokens')),
    );
    draw();
    runGen();
    return wrap;
  },
};

/* ══════════════════════════ 2. 토큰과 컨텍스트 창 ═════════════════════ */
const tokenScreen = {
  id: 'tokens',
  title: '토큰과 컨텍스트 창',
  render(ctx) {
    const wrap = h('div');
    const state = { text: '안녕하세요. 오늘 도서관 이용 시간을 알고 싶습니다.' };
    const out = h('div');

    const draw = () => {
      const toks = tokenize(state.text);
      out.textContent = '';
      const colorOf = { ko: PALETTE[0], en: PALETTE[2], num: PALETTE[3], sym: '#6b7280' };
      out.append(
        h('div.stat-row',
          statBox('글자 수', String(state.text.length)),
          statBox('토큰 수', String(toks.length), 'accent'),
          statBox('글자당 토큰', fx(toks.length / Math.max(1, state.text.length), 2)),
        ),
        h('div.tokens', ...toks.map((t) => h('span.tok', {
          style: { borderColor: colorOf[t.kind], color: colorOf[t.kind] },
        }, t.text))),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '한국어는 대략 1~2글자, 영어는 3~4글자가 토큰 하나입니다. ',
          '같은 뜻이라도 한국어가 토큰을 더 많이 먹는 편이라, 같은 내용이면 비용이 더 나옵니다.'),
      );
    };

    // 대화가 길어질 때 토큰이 어떻게 쌓이나
    const growBox = h('div');
    const drawGrow = () => {
      const turns = 40;
      const perTurn = 60;
      const sysT = 120;
      const cum = [];
      const cumSum = [];
      let acc = 0;
      for (let i = 1; i <= turns; i++) {
        const thisCall = sysT + perTurn * (2 * i - 1); // 시스템 + 지금까지의 모든 대화
        cum.push([i, thisCall]);
        acc += thisCall;
        cumSum.push([i, acc]);
      }
      const cv = lineChart({
        height: 210,
        series: [
          { name: '한 번 부를 때 보내는 토큰', points: cum },
          { name: '지금까지 보낸 토큰 총합', points: cumSum, color: PALETTE[1] },
        ],
        xLabel: '주고받은 횟수', yMin: 0,
      });
      onResize(() => cv.redraw());
      growBox.textContent = '';
      growBox.append(cv,
        table(['턴', '이번에 보내는 토큰', '지금까지 총합'],
          [1, 5, 10, 20, 40].map((i) => [i, cum[i - 1][1].toLocaleString(), cumSum[i - 1][1].toLocaleString()]),
          { compact: true }),
        note('warn', h('b', '대화가 길어질수록 「한 번 부를 때」 보내는 양이 계속 늡니다. '),
          '앞 대화를 매번 통째로 다시 보내야 하기 때문입니다. ',
          '그래서 요약해서 줄이는 방법이 필요합니다 → 이 탭 마지막 화면.'));
    };

    wrap.append(
      screenHead('토큰과 컨텍스트 창', '모델은 글자가 아니라 「토큰」을 셉니다. 한 번에 볼 수 있는 양에는 끝이 있습니다.', '④ 말을 만들기'),

      card('토큰이란',
        h('p', '모델은 글을 글자 단위로도, 낱말 단위로도 다루지 않습니다. 그 중간쯤의 조각인 ',
          b('토큰'), ' 으로 다룹니다. 자주 나오는 덩어리는 통째로 한 토큰이 되고, 드문 낱말은 여러 조각으로 쪼개집니다.'),
        textarea({ value: state.text, rows: 3, onInput: (v) => { state.text = v; draw(); } }),
        h('div.pills', ...[
          '안녕하세요. 오늘 도서관 이용 시간을 알고 싶습니다.',
          'Hello, I would like to know the library hours today.',
          '주문번호 A-1234 배송 상태를 확인해 주세요',
        ].map((t) => h('button.pill', {
          type: 'button',
          onclick: () => { state.text = t; wrap.querySelector('textarea').value = t; draw(); },
        }, t.slice(0, 16) + '…'))),
        out,
        h('div.honest',
          h('b', '실제 모델의 토크나이저와는 다릅니다. '),
          '진짜는 자료에서 자주 붙어 나오는 조각을 통계로 배워 만듭니다(BPE 등). ',
          '여기서는 「한글 두 글자, 영어 네 글자쯤」이라는 널리 알려진 어림셈을 그대로 썼습니다. ',
          '감을 잡는 데는 충분하지만, 정확한 비용 계산에는 각 서비스가 주는 토큰 계산기를 쓰세요.'),
      ),

      card('컨텍스트 창 — 한 번에 볼 수 있는 양',
        code([
          '┌───────────────────── 컨텍스트 창 ─────────────────────┐',
          '│  [시스템 메시지]  +  [지금까지의 대화]  +  [이번 질문]  │',
          '│      역할 설정          앞의 주고받음        새 입력     │',
          '│                                                        │',
          '│  ← 이 안에 든 것만 모델이 「볼」 수 있다 →             │',
          '└────────────────────────────────────────────────────────┘',
        ].join('\n'), { copy: false }),
        h('p', b('창 밖에 있는 것은 모델에게 존재하지 않습니다. '),
          '앞 대화를 여기에 넣지 않으면, 모델은 그것을 알 방법이 아예 없습니다.'),
        table(['창 크기', '체감 분량', '무슨 뜻인가'], [
          ['4천 토큰', '짧은 보고서 하나', '초기 모델. 긴 대화가 금방 잘렸다'],
          ['12만 토큰', '책 몇 권 분량', '긴 문서를 통째로 넣을 수 있다'],
          ['100만 토큰', '책 수백 권 분량', '자료 전체를 넣는 것도 가능해졌다'],
        ]),
        note('info', h('b', '창이 커졌으니 RAG 가 필요 없어진 걸까요? '),
          '아닙니다. ① 넣는 만큼 돈과 시간이 듭니다 ② 넣은 것이 많을수록 정작 중요한 대목을 놓치기 쉽습니다 ',
          '③ 자료가 바뀔 때마다 전부 다시 넣어야 합니다. ',
          '필요한 조각만 골라 넣는 편이 싸고 정확합니다. 그것이 탭 ⑤의 RAG 입니다.'),
      ),

      card('대화가 길어지면 얼마나 늘어날까',
        h('p', { style: { color: 'var(--dim)' } },
          '한 번 주고받을 때 60토큰씩 쓰고, 시스템 메시지가 120토큰이라고 가정한 그림입니다.'),
        button('그려 보기', drawGrow),
        growBox,
      ),

      deepDive('왜 낱말이 아니라 토큰일까',
        h('p', b('낱말 단위로 하면 — '), '세상의 모든 낱말을 사전에 담아야 합니다. 새 낱말·오타·고유명사가 나오면 처리할 수 없습니다.'),
        h('p', b('글자 단위로 하면 — '), '사전은 작아지지만 문장이 아주 길어집니다. 「도서관」을 세 걸음에 나눠 처리해야 합니다.'),
        h('p', b('그래서 중간을 택했습니다. '), '자주 붙어 나오는 조각은 통째로 하나로 묶고, ',
          '드문 낱말은 조각으로 쪼갭니다. 사전 크기도 알맞고, 처음 보는 낱말도 조각을 이어 붙여 처리할 수 있습니다.'),
        code([
          '자주 나오는 낱말:  "도서관"      → [도서관]        1토큰',
          '드문 낱말:        "쿼드러플렛"   → [쿼, 드, 러, 플, 렛]  5토큰',
          '',
          '영어가 한국어보다 토큰이 적게 드는 까닭도 여기 있다.',
          '학습 자료에 영어가 훨씬 많아서, 영어 덩어리가 통째로 사전에 들어갔기 때문이다.',
        ].join('\n'), { copy: false }),
      ),

      quizBlock('t4/tokens', [
        {
          type: 'choice',
          q: '컨텍스트 창(context window)이란?',
          options: [
            '모델이 한 번 부를 때 볼 수 있는 최대 토큰 수',
            '모델이 저장하는 대화 기록',
            '모델의 파라미터 개수',
            '화면에 보이는 글자 수',
          ],
          answer: 0,
          why: '시스템 메시지 + 대화 이력 + 이번 질문이 모두 이 안에 들어가야 합니다. 넘치면 잘립니다.',
        },
        {
          type: 'choice',
          q: '같은 뜻의 문장인데 한국어가 영어보다 토큰을 더 많이 먹는 편인 까닭은?',
          options: [
            '한글이 더 복잡한 글자라서',
            '토크나이저가 배운 자료에 영어가 훨씬 많아, 영어 덩어리가 통째로 사전에 들어가서',
            '한국어에는 띄어쓰기가 없어서',
            '모델이 한국어를 못 해서',
          ],
          answer: 1,
          why: '토크나이저는 학습 자료에서 자주 붙어 나온 조각을 사전에 담습니다. 자료에 적은 언어일수록 잘게 쪼개집니다.',
        },
        {
          type: 'choice',
          q: '컨텍스트 창이 아주 커져도 RAG 가 여전히 쓸모 있는 까닭이 아닌 것은?',
          options: [
            '넣는 양만큼 비용과 시간이 든다',
            '많이 넣을수록 중요한 대목을 놓치기 쉽다',
            '자료가 바뀌면 필요한 조각만 다시 넣으면 된다',
            '창이 커지면 모델의 정확도가 반드시 떨어진다',
          ],
          answer: 3,
          why: '창이 커진다고 정확도가 반드시 떨어지지는 않습니다. 나머지 셋이 RAG 를 계속 쓰는 진짜 이유입니다.',
        },
      ]),
      nextHint('프롬프트 실험실로 →', () => ctx.go('llm', 'prompt')),
    );
    draw();
    drawGrow();
    return wrap;
  },
};

/* ══════════════════════════ 3. 프롬프트 실험실 ════════════════════════ */
const promptScreen = {
  id: 'prompt',
  title: '프롬프트 실험실',
  render(ctx) {
    const wrap = h('div');
    const S = getSentiment();
    const state = {
      text: '자리는 넓은데 음료가 식어서 나왔어요',
      role: false, fewshot: false, cot: false, format: false, style: 'zero',
    };
    const promptBox = h('div');
    const answerBox = h('div');

    const buildPrompt = () => {
      const lines = [];
      if (state.role) lines.push('당신은 고객 후기를 분류하는 상담 분석가입니다.');
      lines.push('다음 후기의 감정을 판단하세요.');
      if (state.format) lines.push('결과는 아래 형식의 JSON 으로만 답하세요.\n{"감정": "긍정|부정|중립", "근거": "..."}');
      if (state.cot) lines.push('먼저 근거가 되는 부분을 하나씩 짚은 뒤, 마지막 줄에 결론을 쓰세요.');
      if (state.fewshot) {
        lines.push([
          '[예시]',
          '후기: 배송이 빨라서 좋았어요  →  긍정',
          '후기: 포장이 엉망이라 실망했습니다  →  부정',
          '후기: 그냥 무난했어요  →  중립',
        ].join('\n'));
      }
      lines.push(`후기: ${state.text}`);
      return lines.join('\n\n');
    };

    const draw = () => {
      promptBox.textContent = '';
      promptBox.appendChild(code(buildPrompt(), { copy: true }));

      // ── 판단 자체는 탭 ②에서 학습시킨 진짜 분류기가 한다 ──
      const p = S.judge(state.text);
      const label = p > 0.62 ? '긍정' : p < 0.38 ? '부정' : '중립';
      const toks = preprocess(state.text);
      const posWords = toks.filter((w) => ['좋다', '넓다', '만족', '친절', '빠르다', '깨끗'].some((k) => w.includes(k)));
      const negWords = toks.filter((w) => ['식', '느리', '별로', '실망', '불편', '아쉽'].some((k) => w.includes(k)));

      let answer;
      if (state.format) {
        answer = JSON.stringify({
          감정: label,
          근거: (negWords[0] || posWords[0] || toks[0] || '') + ' 부분',
        }, null, 2);
      } else if (state.cot) {
        answer = [
          '1) 「자리는 넓은데」 — 좋은 점을 말한 부분',
          `2) 「${state.text.slice(-14)}」 — 아쉬운 점을 말한 부분`,
          '3) 좋은 점과 아쉬운 점이 함께 있음',
          `결론: ${label}`,
        ].join('\n');
      } else if (state.fewshot) {
        answer = label;
      } else {
        answer = `이 후기는 ${label}으로 보입니다. 자리에 대해서는 만족했지만 음료 상태에 대한 아쉬움이 함께 담겨 있어, 전체적으로는 그렇게 판단됩니다. 다만 표현이 강하지 않아 해석의 여지가 있습니다.`;
      }

      answerBox.textContent = '';
      answerBox.append(
        h('div.msg.msg-ai', h('span.msg-role', '모델'), h('pre.msg-text', { style: { whiteSpace: 'pre-wrap', fontFamily: 'inherit' } }, answer)),
        h('div.stat-row',
          statBox('판단', label, label === '긍정' ? 'good' : label === '부정' ? 'bad' : ''),
          statBox('긍정 확률', pct(p, 0), 'accent'),
          statBox('보낸 토큰', String(countTokens(buildPrompt()))),
          statBox('받은 토큰', String(countTokens(answer))),
        ),
      );
    };

    wrap.append(
      screenHead('프롬프트 실험실', '같은 일을 시켜도 어떻게 말하느냐에 따라 답의 모양이 달라집니다.', '④ 말을 만들기'),

      card('세 가지 방식',
        table(['방식', '어떻게 시키나', '언제 쓰나'], [
          [h('b', 'Zero-shot'), '예시 없이 그냥 시킨다', '간단한 일. 가장 먼저 해 볼 방법'],
          [h('b', 'Few-shot'), '입력–출력 예시를 두세 개 보여 준다', '출력 형식이 들쭉날쭉할 때'],
          [h('b', 'Chain-of-Thought'), '「단계별로 생각한 뒤 답하라」고 시킨다', '계산·추론이 필요한 어려운 일'],
        ]),
        h('p', b('먼저 프롬프트, 그래도 안 되면 예시, 그래도 안 되면 파인튜닝 — '),
          '이 순서로 갑니다. 파인튜닝은 자료 수백~수만 건과 학습 비용이 들기 때문입니다.'),
      ),

      card('스위치를 켜 보세요',
        input({ value: state.text, onInput: (v) => { state.text = v; draw(); } }),
        h('div.pills', ...[
          '자리는 넓은데 음료가 식어서 나왔어요',
          '직원분이 정말 친절하셔서 기분 좋게 다녀왔습니다',
          '그냥 뭐 무난했어요',
        ].map((t) => h('button.pill', {
          type: 'button',
          onclick: () => { state.text = t; wrap.querySelector('input.inp').value = t; draw(); },
        }, t.slice(0, 14) + '…'))),
        h('div', { style: { margin: '10px 0' } },
          toggle('역할 정해 주기', false, (v) => { state.role = v; draw(); }),
          toggle('예시 보여 주기 (few-shot)', false, (v) => { state.fewshot = v; draw(); }),
          toggle('단계별로 생각하게 하기 (CoT)', false, (v) => { state.cot = v; draw(); }),
          toggle('출력 형식 정해 주기 (JSON)', false, (v) => { state.format = v; draw(); }),
        ),
        h('p', b('완성된 프롬프트')),
        promptBox,
        h('p', b('모델의 답')),
        answerBox,
        h('div.honest',
          h('b', '반은 진짜, 반은 흉내입니다. '),
          '긍정·부정 ', b('판단'), ' 은 탭 ②에서 실제로 학습시킨 분류기가 합니다(확률도 진짜입니다). ',
          '다만 ', b('답의 문장을 짓는 부분'), ' 은 진짜 LLM 이 아니라, ',
          '「프롬프트를 바꾸면 출력 모양이 이렇게 달라진다」를 보여 주는 장치입니다.'),
      ),

      card('프롬프트를 잘 쓰는 다섯 가지',
        terms([
          ['구체적으로 지시하기', '「분석해 줘」 대신 「긍정·부정·중립 중 하나로 분류하고 근거를 한 문장으로 써 줘」'],
          ['역할 주기', '「당신은 고객 후기를 분류하는 분석가입니다」 — 말투와 관점이 정해진다'],
          ['형식 정해 주기', '「JSON 으로만 답하세요」 — 프로그램이 받아 쓰려면 필수'],
          ['예시 보여 주기', '설명 열 줄보다 예시 두 개가 낫다. 특히 「모르면 이렇게 답하라」는 예시가 강력하다'],
          ['해서는 안 될 일 적기', '「추측하지 마세요」 「문서에 없으면 없다고 하세요」 — 탭 ⑤에서 다시 나온다'],
        ]),
        note('warn', h('b', '주의 — '), '프롬프트 문장 안에 ', mono('{ }'), ' 를 그냥 쓰면 안 됩니다. ',
          '많은 도구가 중괄호를 「값이 들어갈 자리」로 읽기 때문에, ', mono('{이름} 고객님'),
          ' 이라고 쓰면 「이름이라는 값을 내놓으라」며 오류가 납니다.'),
      ),

      deepDive('Chain-of-Thought 는 왜 통하나',
        h('p', '모델은 토큰을 하나씩 만들어 냅니다. 곧, ', b('생각할 시간 = 만들어 내는 토큰 수'), ' 입니다. ',
          '바로 답만 내놓으라고 하면 몇 토큰 안에 결론을 내야 하지만, ',
          '「단계별로 쓰라」고 하면 중간 과정을 적으면서 스스로 정리할 여지가 생깁니다.'),
        code([
          '[바로 답하게 하면]',
          '  Q: 사과 3개를 2500원에 사서 1개당 1200원에 팔면 이익은?',
          '  A: 1100원          ← 틀림',
          '',
          '[단계별로 쓰게 하면]',
          '  1) 판 돈 = 3 × 1200 = 3600원',
          '  2) 산 돈 = 2500원',
          '  3) 이익 = 3600 − 2500 = 1100원',
          '  A: 1100원          ← 과정을 적으니 검산도 가능',
        ].join('\n'), { copy: false }),
        h('p', b('대가도 있습니다. '), '토큰을 많이 쓰니 느리고 비쌉니다. ',
          '분류처럼 단순한 일에는 오히려 방해가 되기도 합니다. 필요한 곳에만 씁니다.'),
        h('p', b('프롬프트 vs 파인튜닝 — '),
          '프롬프트는 자료 0건, 즉시 시작, 대신 일관성이 낮습니다. ',
          '파인튜닝은 자료 수백 건 이상과 학습 비용이 들지만 일관성이 높고 특정 과제에서 최고 성능을 냅니다. ',
          '대부분의 경우 프롬프트로 충분합니다.'),
      ),

      pyBox([
        'from langchain_core.prompts import ChatPromptTemplate',
        '',
        'prompt = ChatPromptTemplate.from_messages([',
        '    ("system", "당신은 고객 후기를 분류하는 분석가입니다. "',
        '               "결과는 긍정·부정·중립 중 하나로만 답하세요."),',
        '    ("human", "후기: {review}"),',
        '])',
        '',
        '# {review} 는 실행할 때 채워지는 빈칸입니다',
        'print(prompt.format(review="배송이 빨라서 좋았어요"))',
      ].join('\n')),

      think('「모르면 모른다고 답하라」는 지시가 왜 그렇게 중요할까요? ',
        '모델이 틀린 답을 자신 있게 말하는 것과, 모른다고 하는 것 중 어느 쪽이 사용자에게 더 나쁠까요?'),

      quizBlock('t4/prompt', [
        {
          type: 'choice',
          q: 'Few-shot 프롬프트란?',
          options: [
            '짧게 물어보는 것',
            '입력–출력 예시를 프롬프트에 몇 개 넣어 주는 것',
            '모델을 조금만 학습시키는 것',
            '온도를 낮추는 것',
          ],
          answer: 1,
          why: '가중치를 고치지 않고, 프롬프트 안의 예시만으로 원하는 패턴을 따르게 합니다.',
        },
        {
          type: 'choice',
          q: '프로그램이 모델의 답을 받아 처리하려면 무엇을 정해 주는 것이 가장 중요할까요?',
          options: ['온도', '출력 형식(JSON 등)', '모델 이름', '문장 길이'],
          answer: 1,
          why: '형식이 들쭉날쭉하면 뒤에서 파싱이 깨집니다. 「JSON 으로만 답하세요」가 실무에서 가장 자주 쓰는 지시입니다.',
        },
        {
          type: 'multi',
          q: 'Chain-of-Thought 에 대해 맞는 것을 모두 고르세요.',
          options: [
            '계산·추론이 필요한 문제에서 정확도가 오르는 경우가 많다',
            '토큰을 더 쓰므로 느리고 비싸진다',
            '모든 과제에서 항상 좋다',
            '모델을 다시 학습시키는 방법이다',
          ],
          answer: [0, 1],
          why: '단순 분류에서는 오히려 방해가 되기도 하고, 학습이 아니라 프롬프트 기법입니다.',
        },
      ]),
      nextHint('파이프라인 조립으로 →', () => ctx.go('llm', 'chain')),
    );
    draw();
    return wrap;
  },
};

/* ══════════════════════════ 4. 파이프라인 조립 ════════════════════════ */
const chainScreen = {
  id: 'chain',
  title: '파이프라인 조립',
  render(ctx) {
    const wrap = h('div');
    const S = getSentiment();
    const state = { parser: 'str', review: '포장이 꼼꼼해서 상자가 멀쩡했어요' };
    const out = h('div');

    const draw = () => {
      const template = '후기: {review}\n감정을 긍정·부정·중립 중 하나로 판단하세요.';
      const filled = template.replace('{review}', state.review);
      const p = S.judge(state.review);
      const label = p > 0.62 ? '긍정' : p < 0.38 ? '부정' : '중립';
      const rawOut = state.parser === 'json'
        ? `{"감정": "${label}", "확신": ${fx(Math.max(p, 1 - p), 2)}}`
        : `이 후기는 ${label}으로 보입니다.`;
      const parsed = state.parser === 'json'
        ? JSON.parse(rawOut)
        : rawOut;

      out.textContent = '';
      out.append(
        stage('① 입력', mono(JSON.stringify({ review: state.review }, null, 0)),
          '체인을 부를 때 넣는 값. 빈칸의 이름과 열쇠 이름이 같아야 한다.'),
        arrow(),
        stage('② 프롬프트 (틀 채우기)', code(filled, { copy: false }),
          '{review} 자리에 값이 들어가 완성된 프롬프트가 만들어진다.'),
        arrow(),
        stage('③ 모델', code(rawOut, { copy: false }),
          '완성된 프롬프트를 받아 글을 만들어 낸다. 결과는 아직 메시지 덩어리다.'),
        arrow(),
        stage('④ 파서', h('div',
          state.parser === 'json'
            ? h('div', mono(JSON.stringify(parsed)), h('p', { style: { color: 'var(--good)', margin: '4px 0 0' } },
              `→ 파이썬 딕셔너리가 되어 result["감정"] = "${parsed.감정}" 처럼 바로 쓸 수 있습니다`))
            : h('div', mono(`"${parsed}"`), h('p', { style: { color: 'var(--dim)', margin: '4px 0 0' } },
              '→ 그냥 문자열입니다. 프로그램이 쓰려면 다시 뜯어내야 합니다')),
        ), '메시지에서 필요한 것만 꺼낸다.'),
      );
    };

    wrap.append(
      screenHead('파이프라인 조립', '프롬프트 | 모델 | 파서 — 부품을 이어 붙여 하나의 흐름으로 만듭니다.', '④ 말을 만들기'),

      card('왜 부품으로 나누나',
        h('p', '모델을 그냥 부르면 이런 코드가 됩니다.'),
        code([
          '# 부품 없이',
          '프롬프트 = "후기: " + 리뷰 + "\\n감정을 판단하세요."',
          '응답 = 모델.호출(프롬프트)',
          '결과 = 응답.내용.strip().replace("감정:", "")   # 매번 손으로 뜯어냄',
        ].join('\n'), { copy: false }),
        h('p', '틀을 만들고 → 부르고 → 결과를 다듬는 일은 ', b('언제나 똑같습니다'), '. ',
          '그래서 이 셋을 부품으로 만들어 파이프(', mono('|'), ')로 잇습니다.'),
        h('p', { style: { textAlign: 'center', fontSize: '1.1rem', margin: '14px 0' } },
          mono('체인 = 프롬프트 | 모델 | 파서')),
        note('good', h('b', '가장 큰 이점 — 갈아 끼우기가 쉽습니다. '),
          '모델을 다른 것으로 바꾸고 싶으면 ', b('가운데 한 줄만'), ' 고치면 됩니다. ',
          '프롬프트와 파서는 그대로 둡니다.'),
      ),

      card('부품 사이로 값이 어떻게 흐르나',
        input({ value: state.review, onInput: (v) => { state.review = v; draw(); } }),
        pillGroup([
          { label: '문자열로 받기 (StrOutputParser)', value: 'str' },
          { label: 'JSON 으로 받기 (JsonOutputParser)', value: 'json' },
        ], (v) => { state.parser = v; draw(); }),
        out,
      ),

      card('실무에서 자주 쓰는 세 가지',
        terms([
          ['여러 건 한꺼번에 (batch)', '리뷰 100개를 한 줄로 처리한다. 하나씩 부르는 것보다 빠르다.'],
          ['여러 단계 잇기', '① 긴 글을 요약 → ② 요약을 다시 번역. 앞 단계의 결과가 뒤 단계의 입력이 된다.'],
          ['갈래로 나눠 동시에', '같은 입력으로 요약과 키워드 추출을 동시에 돌리고 결과를 합친다.'],
        ]),
        code([
          '# 여러 건 한꺼번에',
          'results = chain.batch([',
          '    {"review": "배송이 빨랐어요"},',
          '    {"review": "포장이 엉망이었습니다"},',
          '])',
          '',
          '# 여러 단계 잇기',
          '요약 = 요약체인.invoke({"text": 긴_글})',
          '번역 = 번역체인.invoke({"text": 요약})',
        ].join('\n'), { copy: false }),
      ),

      deepDive('파이프 기호는 무엇을 하는가',
        h('p', mono('a | b'), ' 는 「a 의 결과를 b 에 넣어라」는 뜻입니다. ',
          '파이썬에서 원래는 비트 연산 기호이지만, 라이브러리가 이 기호의 뜻을 새로 정해 두어 ',
          '흐름을 눈에 보이게 적을 수 있게 했습니다.'),
        code([
          'chain = prompt | llm | parser',
          '',
          '# 위 한 줄은 사실 이런 뜻이다',
          'def chain_invoke(입력):',
          '    x = prompt.invoke(입력)   # 틀 채우기',
          '    x = llm.invoke(x)         # 모델 호출',
          '    x = parser.invoke(x)      # 결과 다듬기',
          '    return x',
        ].join('\n'), { copy: false }),
        h('p', b('그래서 순서를 마음대로 바꿀 수 있습니다. '),
          '검색기를 앞에 끼워 넣으면 그대로 RAG 가 됩니다(탭 ⑤). ',
          '이 「부품을 잇는다」는 감각이 다음 두 탭의 바탕이 됩니다.'),
      ),

      pyBox([
        'from langchain_core.prompts import ChatPromptTemplate',
        'from langchain_core.output_parsers import StrOutputParser, JsonOutputParser',
        '',
        'prompt = ChatPromptTemplate.from_template(',
        '    "후기: {review}\\n감정을 긍정·부정·중립 중 하나로만 답하세요.")',
        '',
        'llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)',
        '#     ↑ 이 한 줄만 바꾸면 다른 모델로 교체됩니다',
        '',
        'chain = prompt | llm | StrOutputParser()',
        'print(chain.invoke({"review": "포장이 꼼꼼했어요"}))',
        '',
        '# JSON 으로 받고 싶다면 파서만 교체',
        'chain_json = prompt | llm | JsonOutputParser()',
      ].join('\n')),

      quizBlock('t4/chain', [
        {
          type: 'choice',
          q: () => [mono('chain = prompt | llm | parser'), ' 에서 파서가 하는 일은?'],
          options: [
            '프롬프트를 만든다',
            '모델을 부른다',
            '모델의 응답에서 필요한 형태만 꺼낸다',
            '토큰을 센다',
          ],
          answer: 2,
          why: '응답은 메시지 덩어리로 옵니다. 문자열만 뽑거나 JSON 으로 바꾸는 것이 파서의 일입니다.',
        },
        {
          type: 'choice',
          q: '모델을 다른 회사 것으로 바꾸려면 이 구조에서 무엇을 고쳐야 하나요?',
          options: ['전부 다시 짜야 한다', '가운데 모델 한 줄만', '파서만', '프롬프트만'],
          answer: 1,
          why: '부품으로 나눠 둔 가장 큰 이점입니다. 프롬프트와 파서는 그대로 두고 모델만 갈아 끼웁니다.',
        },
        {
          type: 'choice',
          q: '프롬프트 틀 안에 쓴 {review} 는 무엇인가요?',
          options: [
            '주석',
            '실행할 때 값이 채워지는 빈칸',
            'JSON 표시',
            '모델 이름',
          ],
          answer: 1,
          why: 'invoke({"review": "..."}) 로 넘긴 값이 그 자리에 들어갑니다. 그래서 프롬프트 문장에 중괄호를 함부로 쓰면 안 됩니다.',
        },
      ]),
      nextHint('기억하는 챗봇으로 →', () => ctx.go('llm', 'memory')),
    );
    draw();
    return wrap;
  },
};

function stage(title, body, hint) {
  return h('div', { style: { margin: '10px 0' } },
    h('p', { style: { margin: '0 0 4px' } }, b(title)),
    body,
    h('p', { style: { color: 'var(--dim)', fontSize: '.86rem', margin: '4px 0 0' } }, hint),
  );
}
const arrow = () => h('div', { style: { textAlign: 'center', color: 'var(--accent)', fontSize: '1.3rem' } }, '↓');

/* ══════════════════════════ 5. 기억하는 챗봇 ══════════════════════════ */
const memoryScreen = {
  id: 'memory',
  title: '기억하는 챗봇',
  render(ctx) {
    const wrap = h('div');
    const state = { useHistory: false, useSummary: false, session: 'A' };
    const sessions = { A: [], B: [] };
    const summaries = { A: '', B: '' };
    const out = h('div');
    const logBox = h('div');

    /* 아주 단순한 상담원 흉내 — 대화 이력에서 이름·주문번호를 찾아 쓴다 */
    const reply = (text, history, summary) => {
      const all = (summary ? summary + '\n' : '') + history.map((m) => m.text).join('\n');
      const name = (all.match(/(?:내 이름은|나는|저는)\s*([가-힣]{2,4})(?:이야|야|입니다|이에요|예요)/) || [])[1];
      const order = (all.match(/[A-Z]-?\d{3,4}/) || [])[0];

      if (/이름/.test(text) && /뭐|무엇|누구/.test(text)) {
        return name ? `${name} 님이라고 하셨습니다.` : '죄송합니다. 이름을 말씀해 주신 적이 없어서 알 수 없습니다.';
      }
      if (/주문|번호/.test(text) && /뭐|무엇|어디|얼마/.test(text)) {
        return order ? `말씀하신 주문번호는 ${order} 입니다.` : '죄송합니다. 주문번호를 아직 듣지 못했습니다.';
      }
      const hi = name ? `${name} 님, ` : '';
      if (/이름은|나는|저는/.test(text)) return `${hi}반갑습니다. 무엇을 도와드릴까요?`;
      if (/[A-Z]-?\d{3,4}/.test(text)) return `${hi}주문번호 확인했습니다. 어떤 점이 궁금하신가요?`;
      return `${hi}말씀 잘 들었습니다. 더 필요하신 것이 있을까요?`;
    };

    const send = (text) => {
      const s = sessions[state.session];
      const history = state.useHistory ? s : [];
      const answer = reply(text, history, state.useSummary ? summaries[state.session] : '');
      s.push({ role: 'human', text });
      s.push({ role: 'ai', text: answer });
      if (state.useSummary) {
        // 요약 체인 흉내 — 이름·주문번호 같은 핵심만 남긴다
        const all = s.map((m) => m.text).join(' ');
        const name = (all.match(/(?:내 이름은|나는|저는)\s*([가-힣]{2,4})(?:이야|야|입니다|이에요|예요)/) || [])[1];
        const order = (all.match(/[A-Z]-?\d{3,4}/) || [])[0];
        summaries[state.session] = [
          name ? `이름은 ${name}` : null,
          order ? `주문번호는 ${order}` : null,
          `주고받은 횟수 ${Math.ceil(s.length / 2)}회`,
        ].filter(Boolean).join(', ') + '.';
      }
      draw();
    };

    const draw = () => {
      const s = sessions[state.session];
      logBox.textContent = '';
      logBox.append(...s.map((m) => h(`div.msg.msg-${m.role}`,
        h('span.msg-role', m.role === 'human' ? '사용자' : '모델'),
        h('div.msg-text', m.text))));
      if (!s.length) logBox.appendChild(h('p', { style: { color: 'var(--dim)' } }, '아직 대화가 없습니다.'));

      const sysT = 40;
      const histT = state.useHistory && !state.useSummary ? s.reduce((a, m) => a + countTokens(m.text) + 4, 0) : 0;
      const sumT = state.useSummary ? countTokens(summaries[state.session]) : 0;

      out.textContent = '';
      out.append(
        h('p', b('다음에 모델에게 실제로 보내는 것')),
        h('div', { style: { border: '1px solid var(--line)', borderRadius: '10px', padding: '10px', background: '#0e1728' } },
          h('div.msg.msg-system', h('span.msg-role', '시스템'), h('div.msg-text', '당신은 친절한 상담원입니다.')),
          state.useSummary && summaries[state.session]
            ? h('div.msg.msg-system', h('span.msg-role', '요약'), h('div.msg-text', summaries[state.session]))
            : null,
          ...(state.useHistory && !state.useSummary
            ? s.map((m) => h(`div.msg.msg-${m.role}`, h('span.msg-role', m.role === 'human' ? '사용자' : '모델'), h('div.msg-text', m.text)))
            : []),
          h('div.msg.msg-human', h('span.msg-role', '사용자'), h('div.msg-text', h('span', { style: { color: 'var(--dim)' } }, '(다음에 입력할 말이 여기 들어갑니다)'))),
        ),
        h('div.stat-row',
          statBox('시스템', `${sysT} 토큰`),
          statBox('대화 이력', `${histT} 토큰`, histT > 200 ? 'bad' : ''),
          statBox('요약본', `${sumT} 토큰`, sumT ? 'good' : ''),
          statBox('합계', `${sysT + histT + sumT} 토큰`, 'accent'),
        ),
        !state.useHistory
          ? note('bad', h('b', '이력을 안 보내면 모델은 앞의 말을 아예 모릅니다. '),
            '아래에서 이름을 알려 준 뒤 「내 이름이 뭐야?」라고 물어보세요.')
          : state.useSummary
            ? note('good', h('b', '요약본만 보냅니다. '),
              '대화가 아무리 길어져도 보내는 양이 거의 늘지 않습니다. 대신 자잘한 내용은 사라집니다.')
            : note('info', '이력을 전부 보냅니다. 정확하지만 대화가 길어질수록 토큰이 계속 늡니다.'),
      );
    };

    const inp = input({
      value: '내 이름은 지민이야. 주문번호는 A-1204 야.',
      onEnter: (v) => { if (v.trim()) { send(v.trim()); inp.value = ''; } },
    });

    wrap.append(
      screenHead('기억하는 챗봇', '모델에는 기억이 없습니다. 그런데 왜 기억하는 것처럼 보일까요?', '④ 말을 만들기'),

      card('모델은 매번 백지에서 시작한다',
        code([
          '[1번째 호출]',
          '  보낸 것: "내 이름은 지민이야"',
          '  받은 것: "반갑습니다, 지민 님!"',
          '  ↑ 호출 끝. 서버는 이 대화를 어디에도 저장하지 않는다.',
          '',
          '[2번째 호출]',
          '  보낸 것: "내 이름이 뭐야?"        ← 이것 하나만 보냈다',
          '  받은 것: "죄송해요, 모르겠습니다."  ← 당연한 결과',
        ].join('\n'), { copy: false }),
        h('p', '이것을 ', b('무상태(Stateless)'), ' 라고 합니다. ',
          '그렇다면 대화형 서비스는 어떻게 기억하는 것처럼 보일까요? 답은 싱겁습니다 — ',
          b('앞 대화를 매번 통째로 다시 보냅니다.')),
      ),

      card('직접 확인하기',
        h('div', { style: { margin: '8px 0' } },
          toggle('앞 대화를 함께 보내기 (메모리 켜기)', false, (v) => { state.useHistory = v; draw(); }),
          toggle('전부 대신 요약본만 보내기', false, (v) => { state.useSummary = v; draw(); }),
        ),
        h('div.pills',
          pillGroup([{ label: '대화방 A', value: 'A' }, { label: '대화방 B', value: 'B' }],
            (v) => { state.session = v; draw(); }),
          button('이 대화방 비우기', () => { sessions[state.session] = []; summaries[state.session] = ''; draw(); }, 'ghost small'),
        ),
        h('div', { style: { display: 'flex', gap: '8px', margin: '10px 0' } },
          inp, button('보내기', () => { if (inp.value.trim()) { send(inp.value.trim()); inp.value = ''; } })),
        h('div.pills', ...[
          '내 이름은 지민이야. 주문번호는 A-1204 야.',
          '내 이름이 뭐야?',
          '내 주문번호가 뭐였지?',
          '고마워요',
        ].map((t) => h('button.pill', { type: 'button', onclick: () => send(t) }, t))),
        h('p', b('대화 기록')),
        logBox,
        out,
      ),

      card('대화방을 나누는 열쇠',
        h('p', '위의 ', b('대화방 A / B'), ' 를 오가며 확인해 보세요. ',
          'A 에서 이름을 알려 준 뒤 B 로 가서 「내 이름이 뭐야?」라고 물으면 모릅니다.'),
        code([
          'store = {}   # 대화방 번호별 이력 저장소',
          '',
          'def 이력_가져오기(대화방번호):',
          '    if 대화방번호 not in store:',
          '        store[대화방번호] = 새_이력_객체()',
          '    return store[대화방번호]',
          '',
          '# 부를 때 대화방 번호를 함께 넘긴다',
          'chain.invoke({"input": "안녕"}, config={"configurable": {"session_id": "손님-001"}})',
        ].join('\n'), { copy: false }),
        note('info', '실제 서비스에서는 로그인한 사용자의 ID 를 대화방 번호로 씁니다. ',
          '그래서 체인 객체 하나로 수천 명을 동시에 상대할 수 있습니다.'),
      ),

      deepDive('요약 전략과 그 대가',
        h('p', '대화가 50번 오가면 이력만 수천 토큰이 됩니다. 매번 그것을 다 보내면 느리고 비쌉니다. ',
          '그래서 오래된 대화를 ', b('요약본 한 문단'), ' 으로 눌러 두고, 그것만 보냅니다.'),
        code([
          '[원문 이력]',
          '  사용자: 내 이름은 지민이야',
          '  모델:   반갑습니다, 지민 님!',
          '  사용자: 주문번호는 A-1204 야',
          '  모델:   확인했습니다',
          '  ... (계속 쌓임)',
          '',
          '[요약본]',
          '  "이름은 지민, 주문번호는 A-1204. 주고받은 횟수 2회."',
          '  ← 이것 하나만 보낸다',
        ].join('\n'), { copy: false }),
        h('p', b('대가 — 자잘한 내용이 사라집니다. '),
          '요약에 안 담긴 것을 나중에 물으면 모릅니다. 그래서 ',
          '「이름·주문번호 같은 핵심 정보는 반드시 남기라」고 요약 프롬프트에 못을 박습니다.'),
        h('p', b('실무에서 쓰는 방법들 — '),
          '① 최근 몇 턴만 그대로 두고 나머지는 요약 ',
          '② 토큰 수가 한도를 넘을 때만 요약 ',
          '③ 중요한 사실은 따로 표에 저장하고 필요할 때만 꺼내기(이것이 사실상 RAG 입니다 → 탭 ⑤).'),
      ),

      pyBox([
        'from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder',
        'from langchain_core.runnables.history import RunnableWithMessageHistory',
        'from langchain_core.chat_history import InMemoryChatMessageHistory',
        '',
        'prompt = ChatPromptTemplate.from_messages([',
        '    ("system", "당신은 친절한 상담원입니다."),',
        '    MessagesPlaceholder(variable_name="history"),   # ← 이력이 들어갈 자리',
        '    ("human", "{input}"),',
        '])',
        'chain = prompt | llm',
        '',
        'store = {}',
        'def get_history(session_id):',
        '    if session_id not in store:',
        '        store[session_id] = InMemoryChatMessageHistory()',
        '    return store[session_id]',
        '',
        'chain_with_memory = RunnableWithMessageHistory(',
        '    chain, get_history,',
        '    input_messages_key="input", history_messages_key="history")',
        '',
        'chain_with_memory.invoke({"input": "내 이름은 지민이야"},',
        '    config={"configurable": {"session_id": "손님-001"}})',
      ].join('\n'), '체인은 그대로 두고 「이력을 넣고 빼주는 껍데기」만 씌운 것입니다'),

      quizBlock('t4/memory', [
        {
          type: 'choice',
          q: 'LLM 이 「무상태(stateless)」라는 말의 뜻은?',
          options: [
            '상태를 저장할 수 없는 하드웨어를 쓴다',
            '호출과 호출 사이에 아무것도 기억하지 않는다',
            '한 번에 한 사람만 상대할 수 있다',
            '학습을 하지 않는다',
          ],
          answer: 1,
          why: '매 호출이 독립입니다. 그래서 앞 대화를 매번 다시 넣어 주어야 「기억하는 것처럼」 보입니다.',
        },
        {
          type: 'choice',
          q: '대화 이력을 요약해서 보내면 좋은 점과 나쁜 점은?',
          options: [
            '좋은 점: 토큰 절약 / 나쁜 점: 자잘한 내용이 사라짐',
            '좋은 점: 정확도 상승 / 나쁜 점: 느려짐',
            '좋은 점: 모델이 학습됨 / 나쁜 점: 비용 증가',
            '좋은 점도 나쁜 점도 없음',
          ],
          answer: 0,
          why: '눌러 담는 만큼 잃는 것이 있습니다. 그래서 핵심 정보는 반드시 남기라고 요약 프롬프트에 적습니다.',
        },
        {
          type: 'choice',
          q: '여러 사용자의 대화를 섞이지 않게 하려면?',
          options: [
            '사용자마다 체인 객체를 따로 만든다',
            '대화방 번호(session_id)를 다르게 주어 이력 저장소를 나눈다',
            '온도를 0으로 둔다',
            '요약을 켠다',
          ],
          answer: 1,
          why: '체인은 하나로 두고 번호만 다르게 넘깁니다. 위 화면의 「대화방 A / B」가 바로 그것입니다.',
        },
      ]),
      nextHint('⑤ 내 문서로 답하기로 →', () => ctx.go('rag', 'search')),
    );
    draw();
    return wrap;
  },
};

export default {
  id: 'llm',
  num: 'Ⅳ',
  title: '말을 만들기',
  screens: [nextScreen, tokenScreen, promptScreen, chainScreen, memoryScreen],
};
