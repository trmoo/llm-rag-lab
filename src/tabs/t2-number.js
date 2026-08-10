/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ② 숫자로 바꾸기 — 낱말을 벡터로 만들어 「계산」할 수 있게 한다.
 *   1. 단어 주머니(BoW)   가장 단순한 수치화와 그 한계 셋
 *   2. TF-IDF 계산기      비율로 고르게 → 흔한 낱말 깎기
 *   3. 코사인 유사도      길이는 무시하고 방향만 본다
 *   4. 감성 분류기        진짜로 학습시켜 보고, 어디서 무너지는지 확인
 *   5. 단어 임베딩 지도   Skip-gram 을 직접 학습시킨다
 * ========================================================================== */

import {
  h, card, note, cols, deepDive, terms, table, screenHead, button, textarea, input,
  toggle, pillGroup, choicePicker, code, pyBox, mono, b, nextHint, fx, pct, heatCell, slider, bar,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import {
  vocabulary, bowMatrix, tf, idf, docFreq, tfidfMatrix, topTerms, cosine, euclid,
  similarityMatrix, preprocess, trainTestSplit, makeVectorizer, naiveBayes,
  logisticRegression, metrics, thresholdTable, counter,
} from '../lib/nlp.js';
import { trainWord2Vec } from '../lib/w2v.js';
import { pca2 } from '../lib/embed.js';
import { barChart, heatmap, lineChart, scatter, PALETTE } from '../lib/chart.js';
import {
  TOY_DOCS, SIM_DOCS, POSITIVE_REVIEWS, NEGATIVE_REVIEWS, TRICKY_REVIEWS, W2V_SENTENCES,
} from '../data/corpus.js';
import { onResize } from '../lib/ui.js';

/* 편집 가능한 손계산용 문서 상태 (화면 사이에서 공유하지 않고 각자 새로 만든다) */
const defaultDocs = () => TOY_DOCS.map((d) => d.words.join(' '));

/* ══════════════════════════ 1. 단어 주머니 ════════════════════════════ */
const bowScreen = {
  id: 'bow',
  title: '단어 주머니',
  render(ctx) {
    const wrap = h('div');
    const state = { lines: defaultDocs() };
    const out = h('div');

    const draw = () => {
      const docs = state.lines.map((l) => l.split(/\s+/).filter(Boolean)).filter((d) => d.length);
      const { vocab, matrix } = bowMatrix(docs);
      out.textContent = '';
      if (!vocab.length) { out.appendChild(note('warn', '문서를 한 줄 이상 적어 주세요.')); return; }

      out.append(
        h('p', b('① 어휘 만들기 '), ' — 모든 문서의 낱말을 모아 중복을 없애고 줄 세웁니다. 이것이 벡터의 ', b('축'), ' 이 됩니다.'),
        h('p', mono(`어휘 = [${vocab.map((w) => `'${w}'`).join(', ')}]   (${vocab.length}개 = ${vocab.length}차원)`)),
        h('p', b('② 문서마다 낱말이 몇 번 나왔는지 센다')),
        table(['', ...vocab], matrix.map((row, i) => [
          h('b', `문서${i + 1}`),
          ...row.map((v) => heatCell(v, Math.max(...matrix.flat()), String(v))),
        ])),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '이 표를 ', b('문서-단어 행렬'), ' 이라고 합니다. 행이 문서, 열이 낱말입니다. ',
          '이제 각 문서는 ', mono(`[${matrix[0].join(', ')}]`), ' 같은 ', b('숫자 배열'), ' 이 되었습니다.'),
      );
    };

    const ta = textarea({
      value: state.lines.join('\n'), rows: 6,
      onInput: (v) => { state.lines = v.split('\n'); draw(); },
    });

    wrap.append(
      screenHead('단어 주머니 (Bag of Words)', '가장 단순한 수치화 — 순서는 버리고 「몇 번 나왔나」만 셉니다.', '② 숫자로 바꾸기'),

      card('컴퓨터가 못 하는 일',
        h('p', '글자 상태에서 컴퓨터가 할 수 있는 것은 사실상 ', b('같냐 다르냐'), ' 하나뿐입니다.'),
        code([
          '리뷰A = "노트가 좋아요"',
          '리뷰B = "볼펜이 좋아요"',
          'print(리뷰A == 리뷰B)   # False',
          '# 「좋아요」가 겹친다는 것도, 얼마나 비슷한지도 모릅니다.',
        ].join('\n'), { copy: false }),
        h('p', '그런데 숫자 배열(벡터)로 바꾸면 갑자기 모든 계산이 가능해집니다 — 거리, 각도, 평균, 정렬.'),
      ),

      card('직접 만들어 보기',
        h('p', { style: { color: 'var(--dim)' } }, '한 줄이 문서 하나입니다. 낱말을 고쳐 보면 표가 바로 바뀝니다.'),
        ta,
        h('div.pills',
          button('처음으로 되돌리기', () => {
            state.lines = defaultDocs();
            ta.value = state.lines.join('\n');
            draw();
          }, 'ghost small'),
        ),
        out,
      ),

      card('단어 주머니가 못 하는 세 가지',
        h('div.cols-3',
          h('div',
            h('p', b('① 순서를 버린다')),
            mono('노트는 좋고 볼펜은 나쁘다'), h('br'),
            mono('볼펜은 좋고 노트는 나쁘다'),
            h('p', { style: { color: 'var(--bad)' } }, '두 문장의 표가 완전히 같습니다. 뜻은 정반대인데요.'),
          ),
          h('div',
            h('p', b('② 긴 글이 이긴다')),
            h('p', { style: { color: 'var(--dim)' } }, '짧은 글에서 「노트」 1번 vs 긴 글에서 2번. 횟수만 보면 긴 글이 더 「노트 글」처럼 보입니다.'),
            h('p', { style: { color: 'var(--good)' } }, '→ 다음 화면의 ', b('TF'), ' 가 해결합니다.'),
          ),
          h('div',
            h('p', b('③ 흔한 낱말이 이긴다')),
            h('p', { style: { color: 'var(--dim)' } }, '문구점 리뷰라면 「배송」은 거의 모든 글에 나옵니다. 많이 나온다고 중요한 것이 아닙니다.'),
            h('p', { style: { color: 'var(--good)' } }, '→ 다음 화면의 ', b('IDF'), ' 가 해결합니다.'),
          ),
        ),
        note('info', '①은 여기서 해결되지 않습니다. 순서와 문맥을 읽으려면 ',
          b('탭 ③ 맥락을 읽기'), ' 의 신경망·Attention 이 필요합니다. 그래서 그 기술이 나온 것입니다.'),
      ),

      deepDive('희소 행렬 — 거의 다 0인 표',
        h('p', '뉴스 기사 백만 건을 다루면 어휘가 수십만 개가 됩니다. 그러면 문서 하나가 ',
          b('수십만 차원 벡터'), ' 가 되는데, 그중 실제로 값이 있는 칸은 몇백 개뿐입니다. 나머지는 전부 0입니다.'),
        h('p', '이런 표를 ', b('희소 행렬(Sparse Matrix)'), ' 이라고 하고, 실제 도구는 0을 저장하지 않고 ',
          '「몇 번째 칸에 얼마」만 적어 두어 메모리를 아낍니다. 백만 × 십만 짜리 표를 그냥 만들면 메모리가 터집니다.'),
        h('p', b('반대말은 밀집 벡터(Dense) — '), '탭 ②의 마지막 화면에서 배울 임베딩은 ',
          '차원이 수십~수백 개뿐이지만 ', b('모든 칸에 뜻이 담긴'), ' 값이 들어갑니다.'),
      ),

      quizBlock('t2/bow', [
        {
          type: 'choice',
          q: () => ['문서 ', mono('"노트 노트 볼펜"'), ' 과 어휘 ', mono("['노트','볼펜','필기']"), ' 로 만든 벡터는?'],
          options: ['[1, 1, 0]', '[2, 1, 0]', '[2, 1, 1]', '[3, 0, 0]'],
          answer: 1,
          why: '노트가 2번, 볼펜이 1번, 필기가 0번입니다. 어휘 순서대로 [2, 1, 0].',
        },
        {
          type: 'choice',
          q: '단어 주머니가 구별하지 못하는 두 문장은?',
          options: [
            '「노트가 좋다」 와 「볼펜이 좋다」',
            '「노트는 좋고 볼펜은 나쁘다」 와 「볼펜은 좋고 노트는 나쁘다」',
            '「좋다」 와 「좋았다」',
            '「노트」 와 「노트 노트」',
          ],
          answer: 1,
          why: '쓰인 낱말과 횟수가 완전히 같아 표가 똑같아집니다. 순서를 버렸기 때문입니다.',
        },
      ]),
      nextHint('TF-IDF 계산기로 →', () => ctx.go('number', 'tfidf')),
    );
    draw();
    return wrap;
  },
};

/* ══════════════════════════ 2. TF-IDF 계산기 ══════════════════════════ */
const tfidfScreen = {
  id: 'tfidf',
  title: 'TF-IDF 계산기',
  render(ctx) {
    const wrap = h('div');
    const state = { lines: defaultDocs(), mode: 'plain', normalize: false, pick: { d: 0, w: null } };
    const out = h('div');
    const stepBox = h('div');

    const draw = () => {
      const docs = state.lines.map((l) => l.split(/\s+/).filter(Boolean)).filter((d) => d.length);
      const vocab = vocabulary(docs);
      const res = tfidfMatrix(docs, { mode: state.mode, normalize: state.normalize, vocab });
      out.textContent = '';
      if (!vocab.length) { out.appendChild(note('warn', '문서를 적어 주세요.')); return; }

      const dfs = vocab.map((w) => docFreq(w, docs));
      const maxAll = Math.max(...res.matrix.flat(), 1e-9);

      out.append(
        // ── TF ─────────────────────────────────────────
        h('p', b('① TF — 이 글에서 얼마나 자주 나왔나 '),
          h('span', { style: { color: 'var(--dim)' } }, '= 그 낱말 횟수 ÷ 그 글의 전체 낱말 수')),
        table(['', ...vocab], docs.map((d, i) => [
          h('b', `문서${i + 1} (${d.length}낱말)`),
          ...vocab.map((w) => {
            const v = tf(w, d);
            return h('span', { style: { color: v ? 'var(--fg)' : 'var(--dim)' } }, fx(v, 3));
          }),
        ]), { compact: true }),

        // ── IDF ────────────────────────────────────────
        h('p', b('② IDF — 전체에서 얼마나 드문가 '),
          h('span', { style: { color: 'var(--dim)' } },
            state.mode === 'plain' ? '= ln( 전체 문서 수 ÷ (등장 문서 수 + 1) )' : '= ln( (N+1) ÷ (df+1) ) + 1')),
        table(['낱말', '등장한 문서 수 (df)', 'IDF', ''], vocab.map((w, j) => [
          mono(w), `${dfs[j]} / ${docs.length}`, fx(res.idfs[j], 4),
          bar(res.idfs[j] / Math.max(...res.idfs, 1e-9), { small: true }),
        ]), { compact: true, rowClass: (r, j) => (res.idfs[j] < 0.01 ? 'dim' : '') }),
        res.idfs.some((v) => Math.abs(v) < 1e-9)
          ? note('warn', h('b', '「IDF = 0」인 낱말을 찾았습니다. '),
            '거의 모든 문서에 나오는 낱말입니다. 이런 낱말은 TF 가 아무리 커도 TF-IDF 가 0이 되어 ',
            b('문서를 구별하는 데 쓸모가 없습니다'), '. 불용어와 같은 취급을 받는 셈입니다.')
          : null,

        // ── TF-IDF ─────────────────────────────────────
        h('p', b('③ TF-IDF = TF × IDF '),
          h('span', { style: { color: 'var(--dim)' } }, ' — 칸을 누르면 계산 과정을 보여 줍니다')),
        table(['', ...vocab], res.matrix.map((row, i) => [
          h('b', `문서${i + 1}`),
          ...row.map((v, j) => {
            const cell = heatCell(v, maxAll, fx(v, 4));
            cell.style.cursor = 'pointer';
            cell.addEventListener('click', () => { state.pick = { d: i, w: j }; showStep(docs, vocab, res, dfs); });
            return cell;
          }),
        ])),
        stepBox,

        // ── 문서별 핵심어 ───────────────────────────────
        h('p', b('④ 문서마다 가장 값이 큰 낱말 = 그 문서의 핵심어')),
        table(['문서', '핵심어 (값이 큰 순서)'], res.matrix.map((row, i) => [
          h('b', `문서${i + 1}`),
          h('span', ...topTerms(row, vocab, 3).map((t) => h('span.tok', t.word, h('span.pos', fx(t.score, 3))))),
        ]), { compact: true }),
      );
      showStep(docs, vocab, res, dfs);
    };

    const showStep = (docs, vocab, res, dfs) => {
      stepBox.textContent = '';
      const { d, w } = state.pick;
      const wi = w === null ? vocab.indexOf(topTerms(res.matrix[d], vocab, 1)[0]?.word) : w;
      if (wi < 0 || !docs[d]) return;
      const word = vocab[wi];
      const doc = docs[d];
      const cnt = doc.filter((x) => x === word).length;
      const tfv = tf(word, doc);
      const idfv = res.idfs[wi];
      const raw = tfv * idfv;
      stepBox.appendChild(card(`손으로 따라 계산하기 — 문서${d + 1} 의 「${word}」`,
        table(['단계', '식', '값'], [
          ['TF', `${cnt} ÷ ${doc.length}`, fx(tfv, 4)],
          ['IDF', state.mode === 'plain'
            ? `ln( ${docs.length} ÷ (${dfs[wi]} + 1) ) = ln(${fx(docs.length / (dfs[wi] + 1), 3)})`
            : `ln( (${docs.length}+1) ÷ (${dfs[wi]}+1) ) + 1`, fx(idfv, 4)],
          ['TF × IDF', `${fx(tfv, 4)} × ${fx(idfv, 4)}`, h('b', fx(raw, 4))],
          state.normalize
            ? ['벡터 길이를 1로 맞춤', '이 줄의 값들을 모두 벡터 길이로 나눔', h('b', fx(res.matrix[d][wi], 4))]
            : null,
        ].filter(Boolean), { compact: true }),
      ));
    };

    const ta = textarea({
      value: state.lines.join('\n'), rows: 6,
      onInput: (v) => { state.lines = v.split('\n'); state.pick = { d: 0, w: null }; draw(); },
    });

    wrap.append(
      screenHead('TF-IDF 계산기', '자주 나오되 아무 데서나 나오지는 않는 낱말 — 그것이 그 글의 핵심어입니다.', '② 숫자로 바꾸기'),

      card('두 개의 물음',
        h('div.cols',
          h('div', h('p', b('TF (Term Frequency)')),
            h('p', '「', b('이 글 안에서'), ' 얼마나 자주 나왔나?」'),
            h('p', { style: { color: 'var(--dim)' } },
              '횟수를 그대로 쓰면 긴 글이 유리하니, ', b('전체 낱말 수로 나눠'), ' 비율로 만듭니다. 0 ~ 1 사이가 됩니다.')),
          h('div', h('p', b('IDF (Inverse Document Frequency)')),
            h('p', '「', b('전체 글 중에서'), ' 얼마나 드문가?」'),
            h('p', { style: { color: 'var(--dim)' } },
              '드문 낱말일수록 그 글의 주제를 잘 짚어 줍니다. 도서관에서 「사랑」이 든 책을 찾으면 만 권이 나오지만, ',
              '「양자컴퓨팅」이 든 책을 찾으면 몇 권으로 좁혀지는 것과 같습니다.')),
        ),
        h('p', { style: { textAlign: 'center', fontSize: '1.1rem', margin: '14px 0' } },
          mono('TF-IDF = TF × IDF'), ' — 두 물음의 답을 곱합니다.'),
      ),

      card('직접 계산하기',
        h('p', { style: { color: 'var(--dim)' } }, '한 줄이 문서 하나입니다. 낱말을 바꾸면 모든 표가 다시 계산됩니다.'),
        ta,
        h('div', { style: { display: 'flex', gap: '18px', flexWrap: 'wrap', alignItems: 'center', margin: '10px 0' } },
          h('span', { style: { color: 'var(--dim)', fontSize: '.9rem' } }, 'IDF 식:'),
          pillGroup([
            { label: '손계산식  ln(N/(df+1))', value: 'plain' },
            { label: '실제 도구식  ln((N+1)/(df+1))+1', value: 'sklearn' },
          ], (v) => { state.mode = v; state.normalize = v === 'sklearn'; draw(); }),
          toggle('벡터 길이를 1로 맞추기 (L2 정규화)', false, (v) => { state.normalize = v; draw(); }),
        ),
        out,
      ),

      card('네 가지 경우로 외우기',
        table(['TF', 'IDF', 'TF-IDF', '무슨 뜻인가'], [
          [h('b', '높음'), h('b', '높음'), h('span', { style: { color: 'var(--good)' } }, '★ 아주 높음'), '이 글의 핵심어! (예: 쫄면 리뷰의 「쫄깃」)'],
          [h('b', '높음'), '낮음', h('span', { style: { color: 'var(--dim)' } }, '낮음'), '자주 나오지만 아무 글에나 나온다 (예: 「배송」)'],
          ['낮음', h('b', '높음'), '중간', '드물게 나왔지만 등장 자체가 특별하다'],
          ['낮음', '낮음', h('span', { style: { color: 'var(--dim)' } }, '거의 0'), '아무 뜻 없음'],
        ]),
      ),

      deepDive('왜 하필 로그(log)를 쓰나',
        h('p', '낱말 빈도는 아주 고르지 않습니다. 몇 개가 전체의 대부분을 차지하고, 나머지는 아주 드물게 나옵니다 ',
          '(이것을 ', b('지프의 법칙'), ' 이라고 합니다).'),
        h('p', '로그를 안 쓰면 이렇게 됩니다.'),
        code([
          '흔한 낱말: 100만 문서 중 100만 문서에 등장 → N/df = 1',
          '드문 낱말: 100만 문서 중     100 문서에 등장 → N/df = 10,000  ← 값이 폭발',
          '',
          '로그를 씌우면',
          '  ln(1)      = 0.0',
          '  ln(10,000) = 9.21   ← 다룰 만한 크기로 눌러 준다',
        ].join('\n'), { copy: false }),
        h('p', '게다가 로그는 「몇 배 드문가」를 ', b('일정한 차이'), ' 로 바꿔 줍니다. ',
          '100→50 문서로 두 배 드물어질 때와 10→5 문서로 두 배 드물어질 때, 로그 값의 차이가 똑같이 0.69 입니다. ',
          '「두 배 드물다」는 개념이 값에 고르게 반영되는 것입니다.'),
        h('p', b('실제 도구는 왜 식이 다른가 — '),
          mono('ln((N+1)/(df+1)) + 1'), ' 에서 뒤의 ', mono('+1'), ' 은 ',
          '모든 문서에 나오는 낱말도 IDF 가 0이 아니라 1이 되게 합니다. 값이 통째로 사라지는 것을 막는 안전장치입니다. ',
          '위에서 두 식을 바꿔 가며 표가 어떻게 달라지는지 보세요.'),
      ),

      pyBox([
        'import math',
        '',
        'def TF(단어, 문서):',
        '    return 문서.count(단어) / len(문서)',
        '',
        'def IDF(단어, 문서들):',
        '    등장 = sum([1 for 문서 in 문서들 if 단어 in 문서])',
        '    return math.log(len(문서들) / (등장 + 1))',
        '',
        '문서들 = [["노트","노트","필기","좋다"], ["볼펜","필기","배송","빠르다"]]',
        'print(TF("노트", 문서들[0]) * IDF("노트", 문서들))',
        '',
        '# 실제 수업에서는 도구가 이 40줄을 3줄로 줄여 줍니다',
        '# vectorizer = TfidfVectorizer()',
        '# 행렬 = vectorizer.fit_transform(문서_문자열들)',
      ].join('\n'), '「파이썬 실습실」 탭에서 그대로 돌아갑니다'),

      think('위 표에서 ', mono('배송'), ' 의 IDF 가 0이 되었습니다. ',
        '그런데 문구점 리뷰만 모은 자료라면 원래 그럴 수밖에 없습니다. ',
        '이럴 때 「배송」이라는 낱말은 정말 쓸모가 없는 걸까요? 어떤 분석에서는 여전히 쓸모가 있을까요?'),

      quizBlock('t2/tfidf', [
        {
          type: 'choice',
          q: () => ['문서 5개 중 4개에 나오는 낱말의 IDF( ', mono('ln(N/(df+1))'), ' )는?'],
          options: ['ln(5/4) ≈ 0.223', 'ln(5/5) = 0', 'ln(4/5) ≈ −0.223', 'ln(5) ≈ 1.609'],
          answer: 1,
          why: 'df=4 이므로 분모는 4+1=5. ln(5/5)=ln(1)=0 입니다. 거의 모든 문서에 나오면 값이 0이 됩니다.',
        },
        {
          type: 'choice',
          q: 'TF 만 쓰고 IDF 를 안 쓰면 어떤 문제가 생기나요?',
          options: [
            '긴 문서가 유리해진다',
            '「배송」처럼 모든 문서에 나오는 흔한 낱말이 높은 점수를 받는다',
            '벡터의 길이가 달라진다',
            '숫자가 음수가 된다',
          ],
          answer: 1,
          why: 'TF 는 문서 길이 문제를 해결하지만, 흔한 낱말을 눌러 주지는 못합니다. 그것이 IDF 의 일입니다.',
        },
        {
          type: 'short',
          q: () => ['TF 를 「횟수」가 아니라 「비율」로 계산하는 까닭은 무엇을 없애기 위해서일까요? (○○ 편향, 두 글자)'],
          accept: ['길이', '문서길이', '문서 길이'],
          why: '문서 길이 편향입니다. 긴 글일수록 모든 낱말의 횟수가 늘어나므로, 전체 낱말 수로 나눠 고르게 만듭니다.',
        },
      ]),
      nextHint('코사인 유사도로 →', () => ctx.go('number', 'cosine')),
    );
    draw();
    return wrap;
  },
};

/* ══════════════════════════ 3. 코사인 유사도 ══════════════════════════ */
const cosineScreen = {
  id: 'cosine',
  title: '코사인 유사도',
  render(ctx) {
    const wrap = h('div');
    const pts = [
      { x: 0.5, y: 0.5, label: 'A', color: PALETTE[0], arrow: true },
      { x: 0.7, y: 0.15, label: 'B', color: PALETTE[1], arrow: true },
    ];
    const readout = h('div');

    const update = () => {
      const A = [pts[0].x, pts[0].y];
      const B = [pts[1].x, pts[1].y];
      const cs = cosine(A, B);
      const ang = (Math.acos(Math.max(-1, Math.min(1, cs))) * 180) / Math.PI;
      readout.textContent = '';
      readout.append(
        h('div.stat-row',
          statBox('A 벡터', `(${fx(A[0], 2)}, ${fx(A[1], 2)})`),
          statBox('B 벡터', `(${fx(B[0], 2)}, ${fx(B[1], 2)})`),
          statBox('사이 각도', `${ang.toFixed(1)}°`, 'accent'),
          statBox('코사인 유사도', fx(cs, 3), cs > 0.8 ? 'good' : cs < 0.3 ? 'bad' : ''),
          statBox('유클리드 거리', fx(euclid(A, B), 3)),
        ),
        h('p', { style: { color: 'var(--dim)' } },
          '두 화살표를 끌어 보세요. ', b('길이를 바꿔도'), ' 코사인 유사도는 그대로지만 ',
          b('거리는 바뀝니다'), '. 방향만 같으면 같은 내용으로 보는 것이 코사인 유사도입니다.'),
      );
    };

    const sc = scatter({
      height: 320, points: pts, xMin: -1, xMax: 1, yMin: -1, yMax: 1,
      onDrag: (i, x, y) => { pts[i].x = x; pts[i].y = y; update(); },
    });
    onResize(() => sc.redraw());

    // 문서 유사도 행렬
    const docs = SIM_DOCS.map((s) => preprocess(s));
    const { vocab, matrix } = (() => {
      const r = tfidfMatrix(docs, { mode: 'sklearn', normalize: true });
      return { vocab: r.vocab, matrix: r.matrix };
    })();
    const sim = similarityMatrix(matrix);
    const labels = SIM_DOCS.map((_, i) => `글${i + 1}`);
    const hm = heatmap({
      matrix: sim, rowLabels: labels, colLabels: labels, cellMax: 1,
      fmt: (v) => (v >= 0.995 ? '1' : v.toFixed(2)), height: 300,
    });
    onResize(() => hm.redraw());

    // 가장 닮은 글 찾기
    const pairs = [];
    for (let i = 0; i < sim.length; i++) {
      for (let j = i + 1; j < sim.length; j++) pairs.push({ i, j, v: sim[i][j] });
    }
    pairs.sort((a, b) => b.v - a.v);

    wrap.append(
      screenHead('코사인 유사도', '두 벡터가 얼마나 닮았나 — 길이는 무시하고 방향만 봅니다.', '② 숫자로 바꾸기'),

      card('화살표를 끌어 보세요',
        sc, readout,
        h('div.cols',
          h('div', h('p', b('같은 방향 (0°)')), mono('cos(0°) = 1.0'), h('p', { style: { color: 'var(--dim)' } }, '완전히 같은 내용으로 봅니다.')),
          h('div', h('p', b('직각 (90°)')), mono('cos(90°) = 0.0'), h('p', { style: { color: 'var(--dim)' } }, '겹치는 것이 하나도 없습니다.')),
        ),
      ),

      card('왜 거리가 아니라 각도인가',
        h('p', '같은 내용을 짧게 쓴 글과 길게 쓴 글이 있다고 합시다.'),
        code([
          '짧은 글:  [노트=1, 배송=1]',
          '긴  글:  [노트=3, 배송=3]     ← 내용은 같고 길이만 3배',
          '',
          '유클리드 거리 = √((3-1)² + (3-1)²) = 2.83   → 꽤 다른 글처럼 보인다  ❌',
          '코사인 유사도 = cos(0°) = 1.0              → 같은 글로 본다        ✅',
        ].join('\n'), { copy: false }),
        note('info', '이것이 텍스트에서 코사인 유사도를 쓰는 가장 큰 이유입니다. ',
          '글 길이는 내용과 상관없이 제각각이니까요.'),
      ),

      card('진짜 글로 재 보기',
        h('p', { style: { color: 'var(--dim)' } },
          '아래 여덟 개의 짧은 글을 TF-IDF 벡터로 만든 뒤, 모든 짝의 코사인 유사도를 계산한 표입니다. ',
          '대각선(자기 자신)은 언제나 1 입니다.'),
        hm,
        table(['닮은 정도', '글', '글'],
          pairs.slice(0, 4).map((p) => [
            h('b', fx(p.v, 3)),
            h('span', { style: { fontSize: '.88rem' } }, `${p.i + 1}. ${SIM_DOCS[p.i]}`),
            h('span', { style: { fontSize: '.88rem' } }, `${p.j + 1}. ${SIM_DOCS[p.j]}`),
          ]), { compact: true }),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '같은 주제(도서관끼리, 문구점끼리, 카페끼리)의 글이 위로 올라옵니다. ',
          '검색 엔진이 「이 글과 비슷한 글」을 찾는 방법이 바로 이것입니다.'),
      ),

      card('여기서 쓰이는 곳',
        terms([
          ['검색', '검색어를 벡터로 바꿔, 수백만 쪽 중 방향이 가장 비슷한 쪽을 위에 올린다'],
          ['표절 검사', '두 글의 유사도가 아주 높으면 표시한다'],
          ['뉴스 묶기', '오늘 기사 천 건에서 서로 닮은 것끼리 묶어 같은 사건으로 본다'],
          ['RAG', '질문 벡터와 가장 가까운 문서 조각을 찾아 모델에게 건네준다 (탭 ⑤)'],
        ]),
      ),

      deepDive('코사인 유사도가 속는 자리',
        h('p', b('「노트가 좋다」 와 「노트가 나쁘다」'), ' 를 재 보면 유사도가 꽤 높게 나옵니다. ',
          '「노트」와 「가」가 겹치고, TF-IDF 는 ', b('낱말이 있는지 없는지'), ' 만 볼 뿐 ',
          b('뜻'), ' 은 보지 않기 때문입니다.'),
        h('p', '반대로 ', b('「강아지가 뛰어놀아요」 와 「개가 달려요」'), ' 는 겹치는 낱말이 하나도 없어 유사도가 0 에 가깝습니다. ',
          '사람이 보기엔 거의 같은 말인데요.'),
        h('p', b('해결은 두 갈래입니다. '),
          '① 낱말의 뜻을 숫자로 담는 ', b('임베딩'), ' (다음 화면, 그리고 탭 ⑤) ',
          '② 문맥까지 읽는 ', b('Attention'), ' (탭 ③).'),
      ),

      quizBlock('t2/cosine', [
        {
          type: 'choice',
          q: () => ['벡터 ', mono('[1, 1]'), ' 과 ', mono('[5, 5]'), ' 의 코사인 유사도는?'],
          options: ['0.0', '0.5', '1.0', '5.0'],
          answer: 2,
          why: '방향이 완전히 같으므로 각도가 0°, cos(0°) = 1 입니다. 길이가 5배 차이나도 상관없습니다.',
        },
        {
          type: 'choice',
          q: '텍스트 분석에서 유클리드 거리보다 코사인 유사도를 즐겨 쓰는 까닭은?',
          options: [
            '계산이 더 빨라서',
            '글의 길이 차이에 흔들리지 않아서',
            '값이 항상 정수라서',
            '음수가 나오지 않아서',
          ],
          answer: 1,
          why: '같은 내용이라도 길게 쓰면 벡터가 커집니다. 코사인은 방향만 보므로 길이 차이를 무시합니다.',
        },
        {
          type: 'choice',
          q: '두 글의 코사인 유사도가 0.02 로 아주 낮게 나왔습니다. 반드시 참인 것은?',
          options: [
            '두 글의 뜻이 반대다',
            '두 글에 겹치는 낱말이 거의 없다',
            '두 글의 길이가 크게 다르다',
            '한 글이 다른 글을 베꼈다',
          ],
          answer: 1,
          why: 'TF-IDF 벡터는 낱말의 유무·빈도만 담습니다. 유사도가 낮다는 것은 「겹치는 낱말이 적다」는 뜻이지, 뜻이 반대라는 뜻이 아닙니다.',
        },
      ]),
      nextHint('감성 분류기로 →', () => ctx.go('number', 'classify')),
    );
    update();
    return wrap;
  },
};

function statBox(k, v, kind = '') {
  return h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));
}

/* ══════════════════════════ 4. 감성 분류기 ════════════════════════════ */
const classifyScreen = {
  id: 'classify',
  title: '감성 분류기',
  render(ctx) {
    const wrap = h('div');
    const X = [...POSITIVE_REVIEWS, ...NEGATIVE_REVIEWS];
    const y = [...POSITIVE_REVIEWS.map(() => 1), ...NEGATIVE_REVIEWS.map(() => 0)];
    const state = { seed: 7, minDf: 3, model: 'lr', th: 0.5 };

    const result = h('div');
    const tryBox = h('div');
    let trained = null;

    const train = () => {
      const sp = trainTestSplit(X, y, { testSize: 0.25, seed: state.seed });
      const vec = makeVectorizer({ min: 2, max: 4, minDf: state.minDf }).fit(sp.Xtrain);
      const nb = naiveBayes(vec.counts(sp.Xtrain), sp.ytrain);
      const lr = logisticRegression(vec.transform(sp.Xtrain), sp.ytrain, { epochs: 500, lr: 3, l2: 0.004 });
      const probs = lr.predictProba(vec.transform(sp.Xtest));
      trained = { sp, vec, nb, lr, probs };
      draw();
    };

    const draw = () => {
      const { sp, vec, nb, lr, probs } = trained;
      const predNB = nb.predict(vec.counts(sp.Xtest));
      const predLR = probs.map((p) => (p >= state.th ? 1 : 0));
      const mNB = metrics(sp.ytest, predNB);
      const mLR = metrics(sp.ytest, predLR);
      const m = state.model === 'nb' ? mNB : mLR;

      result.textContent = '';
      result.append(
        h('div.stat-row',
          statBox('훈련 자료', `${sp.Xtrain.length}개`),
          statBox('시험 자료', `${sp.Xtest.length}개`),
          statBox('특성(글자 조각)', `${vec.vocab.length}개`),
          statBox('정확도', pct(m.accuracy, 0), m.accuracy >= 0.7 ? 'good' : m.accuracy < 0.6 ? 'bad' : 'accent'),
          statBox('F1', fx(m.f1, 2)),
        ),

        h('p', b('혼동 행렬 '), h('span', { style: { color: 'var(--dim)' } }, '— 무엇을 맞히고 무엇을 틀렸나')),
        table(['', '부정이라 예측', '긍정이라 예측'], [
          [h('b', '실제 부정'),
            h('span', { style: { color: 'var(--good)' } }, `${m.tn} ✅`),
            h('span', { style: { color: 'var(--bad)' } }, `${m.fp} ❌ (불만인데 만족으로 봄)`)],
          [h('b', '실제 긍정'),
            h('span', { style: { color: 'var(--bad)' } }, `${m.fn} ❌ (만족인데 불만으로 봄)`),
            h('span', { style: { color: 'var(--good)' } }, `${m.tp} ✅`)],
        ]),
        table(['지표', '값', '무슨 뜻인가'], [
          ['정확도', pct(m.accuracy, 1), '전체 중 맞힌 비율'],
          ['정밀도', pct(m.precision, 1), '「긍정」이라 한 것 중 진짜 긍정 비율 — 헛짚음이 치명적일 때 본다'],
          ['재현율', pct(m.recall, 1), '진짜 긍정 중 찾아낸 비율 — 놓치면 치명적일 때 본다'],
          ['F1', fx(m.f1, 3), '두 지표의 균형'],
        ], { compact: true }),

        h('p', b('두 모형 겨루기')),
        table(['모형', '원리', '정확도', 'F1'], [
          ['나이브 베이즈', '낱말이 각 부류에서 나올 확률을 곱한다. 빠르고 자료가 적을 때 강하다.',
            pct(mNB.accuracy, 0), fx(mNB.f1, 2)],
          ['로지스틱 회귀', '낱말마다 가중치를 두고 점수를 더한다. 왜 그렇게 판단했는지 볼 수 있다.',
            pct(mLR.accuracy, 0), fx(mLR.f1, 2)],
        ], { compact: true }),
      );

      // 학습이 무엇을 배웠나 — 가중치
      const top = lr.weights.map((w, i) => ({ w, g: vec.vocab[i] })).sort((a, bb) => bb.w - a.w);
      const posBar = barChart({
        height: 190, horizontal: true,
        items: top.slice(0, 8).map((x) => ({ label: x.g, value: x.w, color: PALETTE[2] })),
        valueFmt: (v) => v.toFixed(2),
      });
      const negBar = barChart({
        height: 190, horizontal: true,
        items: top.slice(-8).reverse().map((x) => ({ label: x.g, value: -x.w, color: PALETTE[7] })),
        valueFmt: (v) => (-v).toFixed(2),
      });
      onResize(() => { posBar.redraw(); negBar.redraw(); });
      result.append(
        h('p', b('모형이 스스로 찾아낸 신호 '),
          h('span', { style: { color: 'var(--dim)' } }, '— 사람이 알려 준 적 없습니다. 자료에서 배운 것입니다.')),
        h('div.cols',
          h('div', h('p', { style: { color: 'var(--good)' } }, '긍정 쪽으로 미는 글자 조각'), posBar),
          h('div', h('p', { style: { color: 'var(--bad)' } }, '부정 쪽으로 미는 글자 조각'), negBar),
        ),
        note('good', '「만족」「친절」「훌륭」이 긍정 쪽에, 「별로」「실망」「불편」「답답」이 부정 쪽에 있다면 ',
          b('제대로 배운 것'), ' 입니다. 엉뚱한 조각이 위에 있다면 자료를 의심해야 합니다. ',
          '가중치 확인은 모형을 고칠 때 가장 먼저 하는 일입니다.'),
      );

      // 임계값 표
      const ths = thresholdTable(sp.ytest, probs, 7);
      result.append(
        h('p', b('기준선(임계값)을 옮기면 — '),
          h('span', { style: { color: 'var(--dim)' } }, '정밀도와 재현율은 한쪽을 올리면 다른 쪽이 내려갑니다')),
        table(['기준', '정확도', '정밀도', '재현율', 'F1'],
          ths.map((r) => [
            h('b', fx(r.th, 2)), pct(r.accuracy, 0), pct(r.precision, 0), pct(r.recall, 0), fx(r.f1, 2),
          ]), { compact: true, rowClass: (r, i) => (Math.abs(ths[i].th - state.th) < 0.06 ? 'hi' : '') }),
      );

      // 새 문장 예측
      tryBox.textContent = '';
      const inp = input({
        value: '자리가 넓고 직원분도 친절해서 아주 만족했어요',
        onEnter: () => runOne(),
      });
      const outP = h('div');
      const runOne = () => {
        const t = inp.value.trim();
        if (!t) return;
        const p = lr.predictProba(vec.transform([t]))[0];
        const pn = nb.predictProba(nb ? vec.counts([t]) : [])[0];
        outP.textContent = '';
        outP.append(
          h('div.stat-row',
            statBox('로지스틱 회귀', `${p >= 0.5 ? '긍정 😊' : '부정 😞'}`, p >= 0.5 ? 'good' : 'bad'),
            statBox('긍정 확률', pct(p, 0), 'accent'),
            statBox('나이브 베이즈', `${pn[1] >= 0.5 ? '긍정 😊' : '부정 😞'}`, pn[1] >= 0.5 ? 'good' : 'bad'),
          ),
          bar(p, { label: `긍정 ${pct(p, 0)}`, color: p >= 0.5 ? 'var(--good)' : 'var(--bad)' }),
          p > 0.4 && p < 0.6
            ? note('warn', '확률이 0.4 ~ 0.6 사이입니다. 모형이 ', b('확신하지 못하는'), ' 구간이에요. ',
              '실제 서비스에서는 이런 것만 따로 모아 사람이 확인합니다(Human-in-the-loop).')
            : null,
        );
      };
      tryBox.append(
        h('p', b('새 문장을 넣어 보세요')),
        h('div', { style: { display: 'flex', gap: '8px' } }, inp, button('판단', runOne)),
        outP,
      );
      runOne();
    };

    // 여러 번 나눠 보기
    const spreadBox = h('div');
    const runSpread = () => {
      const seeds = [1, 3, 7, 11, 42, 99, 2026];
      const rows = seeds.map((s) => {
        const sp = trainTestSplit(X, y, { testSize: 0.25, seed: s });
        const vec = makeVectorizer({ min: 2, max: 4, minDf: state.minDf }).fit(sp.Xtrain);
        const nb = naiveBayes(vec.counts(sp.Xtrain), sp.ytrain);
        const lr = logisticRegression(vec.transform(sp.Xtrain), sp.ytrain, { epochs: 500, lr: 3, l2: 0.004 });
        return {
          s,
          nb: metrics(sp.ytest, nb.predict(vec.counts(sp.Xtest))).accuracy,
          lr: metrics(sp.ytest, lr.predict(vec.transform(sp.Xtest))).accuracy,
        };
      });
      const avg = (k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
      spreadBox.textContent = '';
      spreadBox.append(
        table(['나누는 씨앗', '나이브 베이즈', '로지스틱 회귀'],
          rows.map((r) => [mono(String(r.s)), pct(r.nb, 0), pct(r.lr, 0)]), { compact: true }),
        h('div.stat-row',
          statBox('나이브 베이즈 평균', pct(avg('nb'), 0), 'accent'),
          statBox('로지스틱 평균', pct(avg('lr'), 0), 'accent'),
          statBox('가장 낮은 점수', pct(Math.min(...rows.flatMap((r) => [r.nb, r.lr])), 0), 'bad'),
          statBox('가장 높은 점수', pct(Math.max(...rows.flatMap((r) => [r.nb, r.lr])), 0), 'good'),
        ),
        note('warn',
          h('b', '같은 자료·같은 모형인데 점수가 크게 흔들립니다. '),
          '자료가 80개뿐이라 어떻게 나누느냐에 따라 시험지가 쉬워지기도 어려워지기도 하기 때문입니다. ',
          '그래서 실무에서는 ', b('한 번 나눠 본 점수를 믿지 않고'), ' 여러 번 나눠 평균을 냅니다(교차 검증). ',
          '자료를 늘리는 것이 가장 확실한 해결책입니다.'),
      );
    };

    wrap.append(
      screenHead('감성 분류기', '리뷰 80개로 「좋다 / 나쁘다」를 판단하는 모형을 브라우저에서 실제로 학습시킵니다.', '② 숫자로 바꾸기'),

      card('다섯 걸음',
        h('div.flow',
          h('span.flow-step', '① 전처리'), h('span.flow-arrow', '→'),
          h('span.flow-step', '② 훈련 / 시험 나누기'), h('span.flow-arrow', '→'),
          h('span.flow-step', '③ TF-IDF 로 숫자화'), h('span.flow-arrow', '→'),
          h('span.flow-step', '④ 학습'), h('span.flow-arrow', '→'),
          h('span.flow-step', '⑤ 성능 재기'),
        ),
        note('bad',
          h('b', '가장 흔한 실수 — 데이터 누수. '),
          '숫자화(③)를 ', b('나누기(②) 전에'), ' 전체 자료로 하면, 시험 문제의 정보가 훈련에 새어 들어갑니다. ',
          '그러면 점수가 실제보다 높게 나와 자신을 속이게 됩니다. ',
          '반드시 ', mono('훈련 자료로만 fit → 시험 자료는 transform 만'), ' 순서를 지킵니다.'),
      ),

      card('설정을 바꿔 다시 학습시키기',
        slider({
          label: '나누는 씨앗 (random_state)', min: 1, max: 100, value: state.seed,
          onInput: (v) => { state.seed = v; train(); },
        }),
        slider({
          label: '몇 개 이상 문서에 나온 조각만 쓸까', min: 1, max: 6, value: state.minDf,
          onInput: (v) => { state.minDf = v; train(); },
        }),
        slider({
          label: '긍정이라 판단할 기준선', min: 0.1, max: 0.9, step: 0.05, value: state.th,
          format: (v) => v.toFixed(2),
          onInput: (v) => { state.th = v; draw(); },
        }),
        pillGroup([{ label: '로지스틱 회귀로 보기', value: 'lr' }, { label: '나이브 베이즈로 보기', value: 'nb' }],
          (v) => { state.model = v; draw(); }),
        result,
      ),

      card('새 문장 판단해 보기', tryBox),

      card('한 번 나눈 점수를 믿어도 될까',
        h('p', { style: { color: 'var(--dim)' } },
          '자료를 일곱 가지 방식으로 나눠 각각 학습시켜 봅니다. 몇 초 걸립니다.'),
        button('일곱 번 나눠 보기', runSpread),
        spreadBox,
      ),

      card('모형이 무너지는 자리',
        h('p', '아래 문장들은 모두 모형에게 어려운 것들입니다. 직접 넣어 보고 확률을 확인해 보세요.'),
        table(['문장', '사람이 보는 정답', '왜 어려운가'],
          TRICKY_REVIEWS.map((t) => [
            mono(t.text),
            t.label === null ? h('span', { style: { color: 'var(--dim)' } }, '없음(중립)')
              : t.label ? h('span', { style: { color: 'var(--good)' } }, '긍정')
                : h('span', { style: { color: 'var(--bad)' } }, '부정'),
            t.note,
          ]), { compact: true }),
        note('warn',
          h('b', '근본 원인은 하나입니다 — '),
          '이 방식은 ', b('낱말(글자 조각)이 있는지'), ' 만 셀 뿐 ', b('어디에 있는지'), ' 는 보지 않습니다. ',
          mono('조용하지 않아서 좋았어요'), ' 에서 「않다」가 어느 낱말에 걸리는지 알 수가 없습니다. ',
          '이것이 탭 ③에서 배울 ', b('순서와 문맥을 읽는 신경망'), ' 이 필요한 이유입니다.'),
      ),

      deepDive('두 모형은 안에서 무엇이 다른가',
        h('p', b('나이브 베이즈'), ' — 「각 부류에서 이 낱말이 나올 확률」을 세어 두고, 새 글이 오면 그 확률들을 ',
          b('곱합니다'), '. 「순진하다(naive)」는 이름은, 낱말들이 서로 아무 상관 없다고 ',
          b('일부러 단순하게 가정'), ' 하기 때문입니다. 현실과 맞지 않지만 계산이 아주 빠르고, 놀랍게도 잘 맞습니다.'),
        code([
          '패턴      긍정에서 확률   부정에서 확률',
          '"만족"        0.80          0.05',
          '"실망"        0.03          0.70',
          '',
          '새 글 "만족" 이 들어오면',
          '  P(긍정) ∝ 0.80,  P(부정) ∝ 0.05  → 긍정',
        ].join('\n'), { copy: false }),
        h('p', b('로지스틱 회귀'), ' — 낱말마다 ', b('가중치'), ' 를 두고 모두 더해 점수를 냅니다. ',
          '그 점수를 0~1 사이 확률로 눌러 주는 것이 ', b('시그모이드'), ' 함수입니다. ',
          '가중치를 꺼내 볼 수 있어서 ', b('왜 그렇게 판단했는지'), ' 설명할 수 있다는 것이 큰 장점입니다.'),
        h('p', b('실무에서는 — '), '고민하지 말고 둘 다 돌려 보고 좋은 것을 씁니다. ',
          '자료가 적을 때는 단순한 모형이 이기는 일이 아주 흔합니다.'),
      ),

      pyBox([
        'from sklearn.feature_extraction.text import TfidfVectorizer',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.naive_bayes import MultinomialNB',
        'from sklearn.metrics import classification_report',
        '',
        'X = 긍정_리뷰 + 부정_리뷰',
        'y = [1]*len(긍정_리뷰) + [0]*len(부정_리뷰)',
        '',
        '# ② 먼저 나눈다 — 이 순서를 어기면 데이터 누수',
        'X_train, X_test, y_train, y_test = train_test_split(',
        '    X, y, test_size=0.25, random_state=42)',
        '',
        '# ③ 훈련 자료로만 fit',
        '변환기 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3)',
        'X_train_vec = 변환기.fit_transform(X_train)   # 학습 + 변환',
        'X_test_vec  = 변환기.transform(X_test)        # 변환만!',
        '',
        '# ④ 학습  ⑤ 평가',
        '모델 = MultinomialNB().fit(X_train_vec, y_train)',
        'print(classification_report(y_test, 모델.predict(X_test_vec)))',
      ].join('\n'), 'char_wb 는 글자 2~4개 묶음을 특성으로 씁니다 — 한국어에 잘 통합니다'),

      quizBlock('t2/classify', [
        {
          type: 'choice',
          q: '시험 자료에 fit() 을 쓰면 안 되는 까닭은?',
          options: [
            '계산이 느려져서',
            '시험 자료의 정보가 학습에 섞여 성능이 실제보다 좋게 나와서',
            '시험 자료는 개수가 적어서',
            '사이킷런이 오류를 내서',
          ],
          answer: 1,
          why: '이것이 데이터 누수(Data Leakage)입니다. 시험 문제를 미리 보고 공부한 셈이라, 점수를 믿을 수 없게 됩니다.',
        },
        {
          type: 'choice',
          q: '리뷰 100개 중 긍정이 95개인 자료에서, 모두 「긍정」이라고만 답하는 모형의 정확도는?',
          options: ['5%', '50%', '95%', '0%'],
          answer: 2,
          why: '95%입니다. 높아 보이지만 부정을 하나도 못 잡습니다. 이래서 정확도만 봐서는 안 되고 정밀도·재현율을 함께 봅니다.',
        },
        {
          type: 'choice',
          q: '스팸 메일 필터에서 특히 중요한 지표는?',
          options: ['정확도', '정밀도', '재현율', '학습 속도'],
          answer: 1,
          why: '정상 메일을 스팸으로 잘못 보내면 큰일입니다(헛짚음이 치명적). 그래서 정밀도를 중시합니다. 반대로 병 진단은 놓치면 안 되므로 재현율을 봅니다.',
        },
        {
          type: 'choice',
          q: () => [mono('조용하지 않아서 좋았어요'), ' 를 이 모형이 자주 틀리는 근본 원인은?'],
          options: [
            '문장이 너무 짧아서',
            '「않다」가 어느 낱말에 걸리는지 순서를 보지 않아서',
            '한글이라서',
            '학습 자료에 이 문장이 없어서',
          ],
          answer: 1,
          why: '글자 조각을 세는 방식은 위치를 버립니다. 부정이 어디에 걸리는지 알 수 없으니 뜻이 뒤집혀도 눈치채지 못합니다.',
        },
      ]),
      nextHint('단어 임베딩 지도로 →', () => ctx.go('number', 'w2v')),
    );

    train();
    return wrap;
  },
};

/* ══════════════════════════ 5. 단어 임베딩 지도 ═══════════════════════ */
const w2vScreen = {
  id: 'w2v',
  title: '단어 임베딩 지도',
  render(ctx) {
    const wrap = h('div');
    const state = { dim: 24, window: 3, epochs: 150, extra: '' };
    let model = null;
    const out = h('div');
    const mapBox = h('div');

    const train = () => {
      const extra = state.extra.split('\n').map((l) => l.split(/\s+/).filter(Boolean)).filter((a) => a.length >= 2);
      const sentences = [...W2V_SENTENCES, ...extra];
      const t0 = performance.now();
      model = trainWord2Vec(sentences, {
        vectorSize: state.dim, window: state.window, epochs: state.epochs, minCount: 1, seed: 42,
      });
      draw(Math.round(performance.now() - t0), sentences.length);
    };

    const draw = (ms, nSent) => {
      out.textContent = '';
      mapBox.textContent = '';
      if (!model.vocab.length) { out.appendChild(note('warn', '문장이 필요합니다.')); return; }

      const loss = lineChart({
        height: 170,
        series: [{ name: '손실', points: model.history.map((x) => [x.epoch, x.loss]) }],
        xLabel: '학습 횟수(epoch)', yMin: 0,
      });
      onResize(() => loss.redraw());

      out.append(
        h('div.stat-row',
          statBox('문장', `${nSent}개`),
          statBox('어휘', `${model.vocab.length}개`),
          statBox('벡터 차원', `${model.dim}`),
          statBox('걸린 시간', `${ms}ms`, 'accent'),
          statBox('손실', `${fx(model.history[0].loss, 2)} → ${fx(model.history.at(-1).loss, 2)}`, 'good'),
        ),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '손실이 내려간다는 것은 「중심 낱말로 주변 낱말을 맞히는 문제」를 점점 잘 풀고 있다는 뜻입니다. ',
          '그 과정에서 벡터가 저절로 정돈됩니다.'),
        loss,
      );

      // 비슷한 낱말
      const probes = ['도서관', '커피', '노트', '실망', '좋다', '배송'].filter((w) => model.has(w));
      out.append(
        h('p', b('비슷한 낱말 찾기 '), h('span', { style: { color: 'var(--dim)' } }, '— 사람이 묶어 준 적이 없습니다')),
        h('div.cols',
          ...probes.map((w) => h('div',
            h('p', mono(w), ' 와 가까운 낱말'),
            ...model.mostSimilar(w, 5).map((s) => h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } },
              h('span', { style: { width: '72px', fontSize: '.9rem' } }, s.word),
              bar(Math.max(0, s.sim), { small: true, label: fx(s.sim, 3) }),
            )),
          )),
        ),
      );

      // 두 낱말 유사도 직접 재 보기
      const w1 = input({ value: '도서관', size: 8 });
      const w2 = input({ value: '커피', size: 8 });
      const simOut = h('span');
      const calc = () => {
        const s = model.similarity(w1.value.trim(), w2.value.trim());
        simOut.textContent = '';
        simOut.appendChild(s === null
          ? h('span', { style: { color: 'var(--warn)' } }, '어휘에 없는 낱말입니다 (OOV)')
          : h('b', { style: { color: s > 0.6 ? 'var(--good)' : s < 0.2 ? 'var(--bad)' : 'var(--fg)' } }, fx(s, 4)));
      };
      w1.addEventListener('input', calc);
      w2.addEventListener('input', calc);
      out.append(
        h('p', b('두 낱말의 닮은 정도 직접 재 보기')),
        h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } },
          w1, h('span', '↔'), w2, h('span', '='), simOut),
      );
      calc();

      // 2차원 지도
      const words = model.vocab.slice(0, 46);
      const vecs = words.map((w) => model.getUnit(w));
      const { points } = pca2(vecs);
      const groupOf = (w) => {
        if (['도서관', '열람실', '자료실', '책', '대출', '반납', '기한', '연체', '회원증', '신간', '사서', '전자책', '사물함'].includes(w)) return 0;
        if (['문구점', '노트', '볼펜', '종이', '잉크', '형광펜', '지우개', '문구', '주문', '배송', '포장', '가격', '재고', '교환', '환불'].includes(w)) return 1;
        if (['카페', '커피', '라떼', '디저트', '향', '맛', '자리', '창가', '콘센트', '매장', '직원'].includes(w)) return 2;
        if (['좋다', '만족', '추천', '최고', '훌륭하다', '친절하다', '깨끗하다', '쾌적하다', '편하다', '빠르다', '고맙다', '기분'].includes(w)) return 3;
        return 4;
      };
      const sc = scatter({
        height: 380,
        points: points.map((p, i) => ({
          x: p[0], y: p[1], label: words[i], color: PALETTE[groupOf(words[i])], r: 4,
        })),
        xMin: -1, xMax: 1, yMin: -1, yMax: 1,
      });
      onResize(() => sc.redraw());
      mapBox.append(
        h('p', { style: { color: 'var(--dim)' } },
          `${model.dim}차원은 사람이 볼 수 없으니, 가장 많이 흩어지는 두 방향만 남겨 평면에 그렸습니다(PCA). `,
          '색은 사람이 나중에 칠한 것이고, ', b('자리는 모형이 스스로 정한 것'), ' 입니다.'),
        h('div.pills',
          ...['도서관', '문구점', '카페', '좋고 나쁨', '그 밖'].map((n, i) =>
            h('span.pill', { style: { cursor: 'default', borderColor: PALETTE[i], color: PALETTE[i] } }, n)),
        ),
        sc,
      );
    };

    wrap.append(
      screenHead('단어 임베딩 지도', '「함께 나오는 낱말은 뜻도 가깝다」 — 이 한 문장으로 낱말의 뜻을 숫자로 만듭니다.', '② 숫자로 바꾸기'),

      card('TF-IDF 가 못 하던 일',
        code([
          'TF-IDF 로 재면',
          '  "책을 빌렸어요"  ↔  "도서를 대출했습니다"     → 0.0   같은 뜻인데 낱말이 안 겹침 ❌',
          '  "커피가 맛있다"  ↔  "커피가 맛없다"          → 0.8   정반대인데 낱말이 겹침    ❌',
        ].join('\n'), { copy: false }),
        h('p', '낱말을 ', b('있다 / 없다'), ' 로만 세었기 때문입니다. 낱말의 ', b('뜻'), ' 자체를 숫자에 담을 수는 없을까요?'),
        note('info',
          h('b', '분포 가설 — '), '「비슷한 자리에 함께 나오는 낱말은 비슷한 뜻이다.」 ',
          '「커피」와 「라떼」는 늘 「향·맛·마시다」 옆에 나옵니다. 그렇다면 둘의 벡터를 가깝게 만들면 되지 않을까? ',
          '이 생각을 신경망으로 구현한 것이 ', b('Word2Vec'), ' 입니다.'),
      ),

      card('브라우저에서 진짜로 학습시키기',
        h('div.honest',
          h('b', '이 화면은 흉내가 아닙니다. '),
          '아래 [다시 학습] 을 누르면 Skip-gram + 네거티브 샘플링으로 ', b('실제 학습'), ' 이 일어납니다. ',
          '손실이 내려가는 것을 그래프로 확인할 수 있습니다.'),
        slider({ label: '벡터 차원 (vector_size)', min: 4, max: 60, value: state.dim, onInput: (v) => { state.dim = v; } }),
        slider({ label: '문맥 범위 (window)', min: 1, max: 6, value: state.window, onInput: (v) => { state.window = v; } }),
        slider({ label: '학습 횟수 (epochs)', min: 20, max: 400, step: 10, value: state.epochs, onInput: (v) => { state.epochs = v; } }),
        h('p', b('문장 더 넣기 '), h('span', { style: { color: 'var(--dim)' } }, '— 한 줄에 한 문장, 낱말은 공백으로 띄웁니다')),
        textarea({
          rows: 3, placeholder: '도서관 조용하다 공부 집중 좋다\n카페 음악 시끄럽다 집중 어렵다',
          onInput: (v) => { state.extra = v; },
        }),
        button('다시 학습시키기', train),
        out,
      ),

      card('낱말 지도', mapBox),

      card('OOV — 배운 적 없는 낱말',
        h('p', 'Word2Vec 의 가장 큰 약점입니다. 학습 자료에 없던 낱말은 ', b('벡터 자체가 없습니다'), '. ',
          '위 「두 낱말 닮은 정도」 칸에 ', mono('우주선'), ' 같은 낱말을 넣어 보세요.'),
        code([
          'if "우주선" in model.wv:',
          '    v = model.wv["우주선"]',
          'else:',
          '    print("어휘에 없는 낱말")   # 이 확인을 빠뜨리면 KeyError',
        ].join('\n'), { copy: false }),
        h('p', { style: { color: 'var(--dim)' } },
          '해결책은 ① 자료를 더 모으기 ② 낱말을 더 작은 조각으로 쪼개 처리하기 ',
          '③ 문맥까지 보는 최신 모형 쓰기 — 이 순서로 생각합니다.'),
      ),

      deepDive('Skip-gram 은 정확히 무엇을 푸는가',
        h('p', '학습은 아주 단순한 문제 하나를 수없이 반복하는 것입니다.'),
        code([
          '문장: [카페, 커피, 향, 진하다, 좋다]     window=2',
          '',
          '중심 낱말 "향" 으로 주변 낱말을 맞혀라',
          '   맞혀야 할 것: 카페, 커피, 진하다, 좋다',
          '   틀린 보기(무작위로 뽑음): 노트, 반납, 지우개   ← 네거티브 샘플',
          '',
          '맞히면 두 벡터를 가까이,  틀린 보기와는 멀리 밀어낸다',
        ].join('\n'), { copy: false }),
        h('p', '이 밀고 당김을 수만 번 반복하면, 같은 자리에 나오던 낱말들끼리 저절로 모입니다. ',
          b('아무도 「커피와 라떼는 비슷하다」고 알려 주지 않았습니다.')),
        terms([
          ['vector_size', '벡터 차원. 자료가 적은데 너무 크면 배울 것이 부족해 오히려 나빠진다(차원의 저주). 실습은 20~50.'],
          ['window', '앞뒤로 몇 낱말까지 볼지. 작으면 문법 관계, 크면 주제 관계를 더 배운다.'],
          ['epochs', '같은 자료를 몇 번 반복할지. 자료가 적을수록 많이 돌린다.'],
          ['min_count', '몇 번 미만 나온 낱말은 버릴지. 실습에서는 1(다 쓰기).'],
          ['seed', '무작위 초기값을 고정해 매번 같은 결과가 나오게 한다.'],
        ]),
        h('p', b('CBOW 와 Skip-gram — '),
          'CBOW 는 반대로 「주변 낱말들로 중심 낱말 맞히기」입니다. 빠르지만 드문 낱말에 약합니다. ',
          '자료가 적은 실습에서는 Skip-gram 이 대개 낫습니다.'),
        h('p', b('그래도 못 하는 것 — '),
          '한 낱말에 벡터가 하나뿐이라 ', mono('배가 고프다'), ' 의 배와 ', mono('배를 타다'), ' 의 배가 ',
          b('같은 벡터'), ' 를 씁니다. 문맥에 따라 벡터가 달라지려면 탭 ③의 Attention 이 필요합니다.'),
      ),

      pyBox([
        'from gensim.models import Word2Vec',
        '',
        '문장들 = [["카페","커피","향","진하다","좋다"],',
        '          ["도서관","열람실","조용하다","자리","넓다"]]',
        '',
        'model = Word2Vec(sentences=문장들, vector_size=30, window=3,',
        '                 min_count=1, epochs=200, sg=1, seed=42)   # sg=1 이 Skip-gram',
        '',
        'print(model.wv.most_similar("커피", topn=5))',
        'print(model.wv.similarity("커피", "라떼"))',
        '',
        'if "우주선" in model.wv:      # OOV 확인은 습관처럼',
        '    print(model.wv["우주선"])',
      ].join('\n')),

      quizBlock('t2/w2v', [
        {
          type: 'choice',
          q: 'Word2Vec 이 기대는 「분포 가설」은 무엇인가요?',
          options: [
            '자주 나오는 낱말이 중요하다',
            '비슷한 자리에 함께 나오는 낱말은 뜻이 비슷하다',
            '문장이 길수록 뜻이 풍부하다',
            '낱말은 알파벳 순서로 정렬해야 한다',
          ],
          answer: 1,
          why: '1950년대 언어학에서 나온 생각을 2013년에 신경망으로 구현한 것이 Word2Vec 입니다.',
        },
        {
          type: 'choice',
          q: '학습 자료에 없던 낱말을 model.wv[...] 로 꺼내면 어떻게 되나요?',
          options: ['0 벡터가 나온다', '가장 비슷한 낱말이 나온다', '오류가 난다 (OOV)', '자동으로 학습된다'],
          answer: 2,
          why: '벡터 자체가 없으므로 KeyError 가 납니다. 꺼내기 전에 「in model.wv」로 확인하는 것이 습관입니다.',
        },
        {
          type: 'multi',
          q: 'TF-IDF 대신 단어 임베딩을 쓰면 나아지는 점을 모두 고르세요.',
          options: [
            '「책」과 「도서」처럼 다른 낱말의 뜻이 가깝다는 것을 안다',
            '벡터 차원이 수만 개에서 수십 개로 줄어든다',
            '「맛있다」와 「맛없다」를 확실히 구별한다',
            '한 낱말이 문맥에 따라 다른 뜻이 되는 것을 처리한다',
          ],
          answer: [0, 1],
          why: '반의어는 비슷한 자리에 나와서 오히려 가깝게 나오기도 합니다. 문맥에 따라 벡터가 달라지는 것은 Transformer(탭 ③)의 몫입니다.',
        },
      ]),
      nextHint('③ 맥락을 읽기로 →', () => ctx.go('context', 'nn')),
    );

    train();
    return wrap;
  },
};

export default {
  id: 'number',
  num: 'Ⅱ',
  title: '숫자로 바꾸기',
  screens: [bowScreen, tfidfScreen, cosineScreen, classifyScreen, w2vScreen],
};
