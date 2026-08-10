/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ⑤ 내 문서로 답하기 (RAG) — 찾아서 · 넣고 · 답한다
 *   1. 임베딩 검색     뜻으로 찾기 vs 낱말로 찾기
 *   2. 문서 자르기     조각 크기와 겹침이 검색을 좌우한다
 *   3. RAG 5단계       문서를 안 주면 지어내고, 주면 맞힌다
 *   4. 검색 품질 올리기 하이브리드 · MMR · 꼬리표 · 기준 점수
 *   5. 근거와 출처     환각을 막는 문장들, 그리고 재는 법
 * ========================================================================== */

import {
  h, card, note, deepDive, terms, table, screenHead, button, input, textarea,
  toggle, pillGroup, code, pyBox, mono, b, nextHint, fx, pct, slider, bar, heatCell, onResize,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import {
  splitByChars, splitByHeadings, buildIndex, retrieve, buildPrompt,
  answerFromContext, answerWithoutContext, expandQuery, PROMPT_PARTS,
} from '../lib/rag.js';
import { embed, pca2, topAxes, AXIS_NAMES } from '../lib/embed.js';
import { cosine, preprocess, bm25Search } from '../lib/nlp.js';
import { scatter, heatmap, barChart, PALETTE } from '../lib/chart.js';
import {
  LIBRARY_DOC, STORE_DOC, SAMPLE_QUESTIONS, NAIVE_GUESSES, DOC_FACTS, NO_ANSWER_QUESTIONS,
} from '../data/docs.js';
import { EMBED_DEMO, SEARCH_PAIRS } from '../data/corpus.js';

const statBox = (k, v, kind = '') => h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));

/* 섹션 단위로 자른 안내문 색인 — 이 탭 여러 화면에서 함께 쓴다 */
let INDEX = null;
const getIndex = () => {
  if (!INDEX) INDEX = buildIndex(splitByHeadings(LIBRARY_DOC));
  return INDEX;
};

/* ══════════════════════════ 1. 임베딩 검색 ════════════════════════════ */
const searchScreen = {
  id: 'search',
  title: '임베딩 검색',
  render(ctx) {
    const wrap = h('div');

    /* ── 뜻 지도 ── */
    const vecs = EMBED_DEMO.map((d) => embed(d.text));
    const { points, project } = pca2(vecs);
    const groups = [...new Set(EMBED_DEMO.map((d) => d.group))];
    const qState = { q: '반려견과 산책해도 되나요' };
    const mapBox = h('div');
    const drawMap = () => {
      const qv = embed(qState.q);
      const qp = project(qv);
      const pts = [
        ...points.map((p, i) => ({
          x: p[0], y: p[1], label: EMBED_DEMO[i].text.slice(0, 10),
          color: PALETTE[groups.indexOf(EMBED_DEMO[i].group)], r: 5,
        })),
        { x: qp[0], y: qp[1], label: '❓ 질문', color: '#fff', r: 8 },
      ];
      const sc = scatter({ height: 360, points: pts, xMin: -1.1, xMax: 1.1, yMin: -1.1, yMax: 1.1 });
      onResize(() => sc.redraw());

      const ranked = EMBED_DEMO
        .map((d, i) => ({ ...d, sim: cosine(qv, vecs[i]) }))
        .sort((a, bb) => bb.sim - a.sim);

      mapBox.textContent = '';
      mapBox.append(
        h('div.pills', ...groups.map((g, i) =>
          h('span.pill', { style: { cursor: 'default', borderColor: PALETTE[i], color: PALETTE[i] } }, g))),
        sc,
        h('p', b('질문과 가까운 순서')),
        table(['닮은 정도', '문장', '갈래'], ranked.slice(0, 5).map((r) => [
          h('span', bar(Math.max(0, r.sim), { small: true, label: fx(r.sim, 3) })),
          r.text, r.group,
        ]), { compact: true }),
      );
    };

    /* ── 뜻으로 찾기 vs 낱말로 찾기 ── */
    const cmpBox = h('div');
    const runCompare = (q) => {
      const idx = getIndex();
      const qv = embed(q);
      const qTok = preprocess(q);
      const vecScores = idx.docs.map((d) => ({ d, s: cosine(qv, d.vector) })).sort((a, bb) => bb.s - a.s);
      const bm = bm25Search(qTok, idx.docs.map((d) => d.tokens), { topK: idx.docs.length });
      cmpBox.textContent = '';
      cmpBox.append(
        h('p', b('질문 '), mono(q), h('span', { style: { color: 'var(--dim)' } },
          `  →  낱말로 쪼개면 [${qTok.join(', ')}]`)),
        h('div.cols',
          h('div',
            h('p', b('뜻으로 찾기 (임베딩)')),
            ...vecScores.slice(0, 4).map((r) => h('div', { style: { margin: '4px 0' } },
              h('span', { style: { display: 'inline-block', width: '150px', fontSize: '.88rem' } }, r.d.metadata.섹션),
              bar(Math.max(0, r.s), { small: true, label: fx(r.s, 3) }))),
          ),
          h('div',
            h('p', b('낱말로 찾기 (BM25)')),
            ...bm.slice(0, 4).map((r) => h('div', { style: { margin: '4px 0' } },
              h('span', { style: { display: 'inline-block', width: '150px', fontSize: '.88rem' } }, idx.docs[r.i].metadata.섹션),
              bar(r.score / Math.max(1e-9, bm[0].score), { small: true, label: fx(r.score, 2) }))),
            bm[0].score === 0
              ? note('bad', h('b', '낱말이 하나도 겹치지 않아 아무것도 못 찾았습니다. '),
                '이것이 낱말 검색의 약점입니다.')
              : null,
          ),
        ),
      );
    };

    wrap.append(
      screenHead('임베딩 검색', '「강아지」로 물었는데 「반려견」이 적힌 글을 찾아내려면 어떻게 해야 할까요?', '⑤ 내 문서로 답하기'),

      card('낱말이 안 겹치면 못 찾는다',
        code([
          '문장 A: "강아지가 공원에서 뛰어놀아요"',
          '문장 B: "개가 야외에서 달리고 있어요"      ← 사람이 보면 같은 말',
          '문장 C: "오늘 파이썬 코드에서 오류가 났어요"',
          '',
          '겹치는 낱말로만 재면:  A ↔ B = 0     A ↔ C = 0',
          '                        ↑ 같은 말인데 0. 구별이 안 된다.',
        ].join('\n'), { copy: false }),
        h('p', '낱말을 ', b('있다 / 없다'), ' 로 세는 한 이 벽을 넘을 수 없습니다. ',
          '낱말의 ', b('뜻'), ' 자체를 좌표로 만들어야 합니다. 그것이 ', b('임베딩'), ' 입니다.'),
        h('div.stat-row',
          statBox('강아지 ↔ 개', fx(cosine(embed('강아지'), embed('개')), 3), 'good'),
          statBox('강아지 ↔ 고양이', fx(cosine(embed('강아지'), embed('고양이')), 3)),
          statBox('강아지 ↔ 코드', fx(cosine(embed('강아지'), embed('코드')), 3), 'bad'),
        ),
      ),

      card('뜻 지도 위에서 찾기',
        h('p', { style: { color: 'var(--dim)' } },
          '문장을 벡터로 만든 뒤 가장 많이 흩어지는 두 방향만 남겨 평면에 그렸습니다. ',
          '질문(흰 점)이 어느 무리 쪽으로 가는지 보세요.'),
        input({ value: qState.q, onInput: (v) => { qState.q = v; drawMap(); } }),
        h('div.pills', ...[
          '반려견과 산책해도 되나요',
          '책을 빌리고 싶어요',
          '필기구를 주문했어요',
          '프로그램이 안 돌아가요',
        ].map((t) => h('button.pill', {
          type: 'button',
          onclick: () => { qState.q = t; wrap.querySelector('input.inp').value = t; drawMap(); },
        }, t))),
        mapBox,
        h('div.honest',
          h('b', '이 앱의 임베딩은 「교실용」입니다. '),
          '진짜 임베딩 모델은 수억 개의 문장을 학습해 768차원 같은 벡터를 만듭니다. ',
          '여기서는 인터넷 없이 돌리려고, 뜻이 가까운 낱말을 스무 갈래로 미리 묶어 24차원 벡터를 만들었습니다. ',
          b('원리(가까우면 비슷하다)는 같고, 규모와 정교함이 다릅니다.')),
      ),

      card('두 검색을 나란히 놓고 보기',
        h('p', { style: { color: 'var(--dim)' } }, '아래 질문을 눌러 「푸른숲 도서관 안내문」에서 찾아 봅니다.'),
        h('div.pills', ...[
          '책을 빌리는 방법이 궁금해요',
          '연체하면 어떻게 되나요',
          '자리를 미리 잡을 수 있나요',
          '전자책',
        ].map((t) => h('button.pill', { type: 'button', onclick: () => runCompare(t) }, t))),
        cmpBox,
        table(['', '뜻으로 찾기 (임베딩)', '낱말로 찾기 (BM25)'], [
          ['원리', '벡터 방향이 얼마나 가까운가', '같은 낱말이 몇 번 나오는가'],
          ['강한 곳', '동의어·비슷한 표현', '정확한 낱말·고유명사·번호'],
          ['약한 곳', '「A-1204」 같은 식별자를 잘 못 찾음', '「빌리다 ↔ 대출」처럼 낱말이 다르면 못 찾음'],
          ['속도', '벡터를 미리 만들어 둬야 함', '빠름'],
        ]),
        note('good', h('b', '그래서 실무에서는 둘을 섞어 씁니다. '),
          '서로의 약점을 메워 주기 때문입니다. 이것을 ', b('하이브리드 검색'), ' 이라 하고, ',
          '이 탭의 네 번째 화면에서 직접 비중을 조절해 봅니다.'),
      ),

      card('벡터 저장소가 하는 일',
        code([
          '┌──────────────── 벡터 저장소 ────────────────┐',
          '│  번호 │ 원문                │ 벡터           │',
          '│ ──────┼────────────────────┼─────────────── │',
          '│   0   │ "대출 기간은 21일…"  │ [0.23, -0.81…] │',
          '│   1   │ "연체하면 정지…"     │ [0.11,  0.44…] │',
          '│   2   │ "자리는 3시간…"      │ [-0.55, 0.32…] │',
          '└─────────────────────────────────────────────┘',
          '',
          '질문 "며칠까지 빌려요?"',
          '   → 질문도 벡터로 바꾸고',
          '   → 가장 가까운 벡터를 찾아',
          '   → 0번 원문을 돌려준다',
        ].join('\n'), { copy: false }),
        h('p', '보통의 데이터베이스는 ', b('정확히 같은 값'), ' 을 찾는 데 최적화되어 있습니다. ',
          '벡터 저장소는 ', b('가까운 값'), ' 을 찾는 데 최적화되어 있습니다. 이것이 결정적인 차이입니다.'),
        note('info', '실습에서는 가벼운 것을 쓰고, 규모가 커지면 전문 서비스로 옮깁니다. ',
          '다행히 도구들이 인터페이스를 맞춰 두어서 ', b('클래스 이름만 바꾸면'), ' 되는 경우가 많습니다.'),
      ),

      deepDive('임베딩 벡터의 숫자는 무슨 뜻인가',
        h('p', '솔직히 말하면 ', b('사람은 알 수 없습니다'), '. ',
          '모델이 대량의 글을 학습하면서 스스로 만들어 낸 표현이라, 3번째 칸이 무엇을 뜻하는지 아무도 정해 주지 않았습니다.'),
        h('p', '이 앱의 교실용 임베딩은 예외적으로 각 칸에 이름이 있습니다. 사람이 직접 갈래를 정했기 때문입니다.'),
        (() => {
          const v = embed('도서관에서 책을 빌렸어요');
          return table(['가장 세게 켜진 갈래', '값'],
            topAxes(v, 6).map((a) => [a.name, fx(a.v, 3)]), { compact: true });
        })(),
        h('p', b('중요한 것은 숫자의 뜻이 아니라 「가깝다」는 사실입니다. '),
          '뜻이 비슷하면 벡터도 가깝다 — 이것만 성립하면 검색은 잘 돌아갑니다.'),
        h('p', b('차원은 몇이 좋은가 — '), '높을수록 담을 수 있는 뜻이 많지만 저장 공간과 계산 시간이 늘어납니다. ',
          '실제 모델은 보통 수백~수천 차원을 씁니다.'),
      ),

      pyBox([
        'from langchain_google_genai import GoogleGenerativeAIEmbeddings',
        'from langchain_chroma import Chroma',
        '',
        '# ① 임베딩 모델 준비',
        'embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")',
        '',
        '# ② 문장을 벡터로 바꿔 저장 (인덱싱)',
        'texts = ["대출 기간은 21일입니다", "연체하면 대출이 정지됩니다"]',
        'vectorstore = Chroma.from_texts(texts=texts, embedding=embedding)',
        '',
        '# ③ 가장 비슷한 것 찾기',
        'for doc in vectorstore.similarity_search("며칠까지 빌릴 수 있나요?", k=2):',
        '    print(doc.page_content)',
      ].join('\n')),

      quizBlock('t5/search', [
        {
          type: 'choice',
          q: '임베딩의 핵심 성질은 무엇인가요?',
          options: [
            '벡터의 길이가 모두 같다',
            '뜻이 비슷한 글은 벡터도 가깝다',
            '숫자가 모두 0과 1 사이다',
            '낱말 수만큼 차원이 생긴다',
          ],
          answer: 1,
          why: '이 성질 하나로 「비슷한 글 찾기」가 거리 계산 문제로 바뀝니다.',
        },
        {
          type: 'choice',
          q: '주문번호 「A-1204」로 문서를 찾을 때 더 잘 통하는 방법은?',
          options: ['뜻으로 찾기(임베딩)', '낱말로 찾기(BM25)', '둘 다 못 찾는다', '차이 없다'],
          answer: 1,
          why: '임베딩은 「비슷한 뜻」을 찾지 「정확히 그 글자」를 찾지 않습니다. 식별자에는 낱말 검색이 강합니다.',
        },
        {
          type: 'choice',
          q: '보통의 데이터베이스와 벡터 저장소의 결정적 차이는?',
          options: [
            '벡터 저장소가 더 빠르다',
            '보통 DB 는 정확히 같은 값을, 벡터 저장소는 가까운 값을 찾는다',
            '벡터 저장소는 글자를 저장하지 않는다',
            '보통 DB 는 한국어를 못 쓴다',
          ],
          answer: 1,
          why: '「강아지가 든 행」이 아니라 「강아지와 뜻이 가까운 것」을 찾는 것이 벡터 저장소의 일입니다.',
        },
      ]),
      nextHint('문서 자르기로 →', () => ctx.go('rag', 'chunk')),
    );
    drawMap();
    runCompare('책을 빌리는 방법이 궁금해요');
    return wrap;
  },
};

/* ══════════════════════════ 2. 문서 자르기 ════════════════════════════ */
const chunkScreen = {
  id: 'chunk',
  title: '문서 자르기',
  render(ctx) {
    const wrap = h('div');
    const state = { size: 300, overlap: 40, mode: 'chars' };
    const out = h('div');
    const testBox = h('div');

    const draw = () => {
      const chunks = state.mode === 'chars'
        ? splitByChars(LIBRARY_DOC, { chunkSize: state.size, overlap: state.overlap })
        : splitByHeadings(LIBRARY_DOC).map((c) => c.text);
      out.textContent = '';
      out.append(
        h('div.stat-row',
          statBox('원문 길이', `${LIBRARY_DOC.length}자`),
          statBox('조각 수', String(chunks.length), 'accent'),
          statBox('평균 길이', `${Math.round(chunks.reduce((a, c) => a + c.length, 0) / chunks.length)}자`),
          statBox('가장 긴 조각', `${Math.max(...chunks.map((c) => c.length))}자`),
        ),
        h('div.scroll-box',
          ...chunks.map((c, i) => h('div', {
            style: {
              margin: '0 0 8px', padding: '8px 10px', borderRadius: '8px',
              background: i % 2 ? '#14203a' : '#172745',
              borderLeft: `3px solid ${PALETTE[i % PALETTE.length]}`,
            },
          },
          h('div', { style: { fontSize: '.76rem', color: 'var(--dim)' } }, `조각 ${i + 1} · ${c.length}자`),
          h('div', { style: { fontSize: '.9rem' } }, c),
          )),
        ),
      );
    };

    const runTest = () => {
      const rows = [];
      for (const cfg of [
        { label: '아주 잘게 (60자, 겹침 없음)', size: 60, overlap: 0 },
        { label: '잘게 (120자)', size: 120, overlap: 10 },
        { label: '작게 (200자)', size: 200, overlap: 20 },
        { label: '보통 (300자)', size: 300, overlap: 40 },
        { label: '크게 (600자)', size: 600, overlap: 60 },
        { label: '아주 크게 (1200자)', size: 1200, overlap: 100 },
        { label: '제목 구조대로', byHead: true },
      ]) {
        const chunks = cfg.byHead
          ? splitByHeadings(LIBRARY_DOC)
          : splitByChars(LIBRARY_DOC, { chunkSize: cfg.size, overlap: cfg.overlap });
        const ix = buildIndex(chunks);
        let hit = 0;
        let chars = 0;
        for (const f of DOC_FACTS) {
          const r = retrieve(ix, f.question, { k: 1, mode: 'hybrid' });
          const joined = r.hits.map((hh) => hh.doc.text).join(' ');
          chars += joined.length;
          if (f.accept.some((a) => joined.includes(a))) hit += 1;
        }
        const rate = hit / DOC_FACTS.length;
        rows.push([
          cfg.label, String(chunks.length),
          h('span', { style: { color: rate === 1 ? 'var(--good)' : rate < 0.8 ? 'var(--bad)' : '' } },
            `${hit} / ${DOC_FACTS.length}`),
          bar(rate, { small: true, label: pct(rate, 0) }),
          h('span', { style: { color: chars / DOC_FACTS.length > 500 ? 'var(--bad)' : 'var(--fg)' } },
            `${Math.round(chars / DOC_FACTS.length)}자`),
        ]);
      }
      testBox.textContent = '';
      testBox.append(
        table(['자르는 방식', '조각 수', '1등 조각에 정답이 있었나', '', '프롬프트에 들어가는 평균 글자'], rows),
        note('info', h('b', `질문–정답 세트 ${DOC_FACTS.length}개로, 「1등 조각 하나만」 써서 재 본 것입니다. `),
          '보아야 할 것은 ', b('두 열의 맞바꿈'), ' 입니다. ',
          '조각이 너무 작으면 답이 두 조각에 걸려 잘려서 놓치고, ',
          '조각이 크면 놓치지는 않지만 ', b('쓸데없는 내용까지 프롬프트에 함께 들어가'), ' 비용이 오르고 초점이 흐려집니다. ',
          '「정답을 놓치지 않으면서 가장 짧은」 지점이 좋은 설정입니다.'),
        note('warn', h('b', '「느낌」으로 정하지 마세요. '),
          '이렇게 재 보지 않으면 설정을 바꾼 것이 개선인지 개악인지 알 수 없습니다.'),
      );
    };

    wrap.append(
      screenHead('문서 자르기', '검색 품질의 절반은 「어떻게 잘랐는가」에서 결정됩니다.', '⑤ 내 문서로 답하기'),

      card('왜 통째로 넣으면 안 되나',
        code([
          '[안내문 전체를 벡터 하나로]',
          '  문제 ① 여러 주제가 한 벡터에 뭉개져 검색이 뭉툭해진다',
          '  문제 ② 찾아도 전체가 돌아오니 프롬프트에 다 넣을 수 없다',
          '',
          '[작은 조각으로 나누면]',
          '  질문 "며칠까지 빌려요?"  →  「대출」 조각만 골라서 쓴다',
        ].join('\n'), { copy: false }),
      ),

      card('직접 잘라 보기',
        pillGroup([
          { label: '글자 수로 자르기', value: 'chars' },
          { label: '제목 구조대로 자르기', value: 'head' },
        ], (v) => { state.mode = v; draw(); }),
        slider({ label: '조각 크기 (chunk_size)', min: 100, max: 1200, step: 20, value: 300, onInput: (v) => { state.size = v; if (state.mode === 'chars') draw(); } }),
        slider({ label: '겹치는 길이 (chunk_overlap)', min: 0, max: 200, step: 10, value: 40, onInput: (v) => { state.overlap = v; if (state.mode === 'chars') draw(); } }),
        out,
      ),

      card('겹치기(overlap)는 왜 필요한가',
        code([
          '[원문] "…배송비는 사는 사람이 냅니다. 뜯어 쓴 필기구는 교환할 수 없습니다…"',
          '',
          '겹침 없음',
          '  [조각1 …사는 사람이 냅니다.]  [조각2 뜯어 쓴 필기구는…]',
          '                                ↑ 연결이 뚝 끊긴다',
          '',
          '겹침 40자',
          '  [조각1 …사는 사람이 냅니다. 뜯어 쓴 필기구는]',
          '                    [조각2 냅니다. 뜯어 쓴 필기구는 교환할 수…]',
          '                     └─ 겹치는 부분이 앞뒤를 이어 준다',
        ].join('\n'), { copy: false }),
        h('p', '답이 조각 경계에 걸쳐 있어도, 겹침이 있으면 ', b('어느 한 조각에는 온전히 담깁니다'), '.'),
        note('warn', '겹침을 너무 크게 하면 같은 내용이 여러 조각에 중복되어 저장 공간과 검색 시간이 늘고, ',
          '검색 결과가 서로 비슷해집니다. 보통 조각 크기의 10~20% 정도로 둡니다.'),
      ),

      card('제목 구조대로 자르기',
        h('p', '안내문처럼 ', mono('## 대출'), ' 같은 제목이 있다면, 글자 수로 자르는 것보다 ',
          b('제목을 따라 자르는 편'), ' 이 훨씬 낫습니다.'),
        h('div.cols',
          h('div', h('p', b('글자 수로 자르면')),
            h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
              '「대출」 이야기 끝부분과 「반납」 시작이 한 조각에 섞일 수 있습니다.')),
          h('div', h('p', b('제목대로 자르면')),
            h('p', { style: { color: 'var(--good)', fontSize: '.9rem' } },
              '조각 하나가 곧 한 주제가 됩니다. 게다가 제목이 ', b('꼬리표'),
              ' 로 함께 저장되어 「대출 부분만 검색」이 가능해집니다(다음다음 화면).')),
        ),
        table(['문서 성격', '권장 조각 크기', '왜'], [
          ['문답·정책 (짧은 항목)', '200 ~ 400자', '한 항목이 곧 한 조각이 되게'],
          ['일반 글·기사', '500 ~ 800자', '문단 단위 맥락을 지키게'],
          ['기술 문서·논문', '800 ~ 1200자', '표·수식·긴 설명이 끊기지 않게'],
          ['대화 기록', '발화 단위', '말하는 사람이 바뀌는 곳에서 끊기게'],
        ]),
        h('p', h('b', '원칙은 하나입니다 — '), '조각 하나가 ', b('하나의 완결된 뜻'), ' 을 담도록.'),
      ),

      card('설정을 바꾸면 정말 좋아질까 — 숫자로 재기',
        h('p', { style: { color: 'var(--dim)' } },
          '안내문에서 답을 아는 질문 7개를 미리 만들어 두고, 자르는 방식마다 ',
          '「검색 결과 안에 정답 값이 들어 있었는가」를 셉니다.'),
        button('여섯 가지 방식으로 재 보기', runTest),
        testBox,
      ),

      deepDive('자연스러운 경계부터 찾는 자르기',
        h('p', '글자 수로 뚝 자르면 낱말 한가운데가 잘립니다. 그래서 실제 도구는 ',
          b('문단 → 문장 → 낱말'), ' 순서로 「자를 만한 자리」를 먼저 찾습니다.'),
        code([
          '1) 먼저 빈 줄(문단)에서 자를 수 있는지 본다',
          '2) 안 되면 마침표(문장)에서 자를 수 있는지 본다',
          '3) 안 되면 공백(낱말)에서 자른다',
          '4) 그래도 안 되면 어쩔 수 없이 글자 수로 자른다',
        ].join('\n'), { copy: false }),
        h('p', b('조각 크기를 바꾸면 벡터 저장소를 다시 만들어야 합니다. '),
          '조각이 달라지면 벡터도 전부 달라지기 때문입니다. ',
          '그래서 자르는 방식은 ', b('처음에 잘 정해 두는 것'), ' 이 좋습니다. 나중에 바꾸면 전부 다시 계산해야 합니다.'),
      ),

      pyBox([
        'from langchain_text_splitters import RecursiveCharacterTextSplitter',
        '',
        'splitter = RecursiveCharacterTextSplitter(',
        '    chunk_size=300,      # 조각 하나의 최대 길이',
        '    chunk_overlap=40,    # 이웃 조각과 겹치는 길이',
        ')',
        'chunks = splitter.split_text(안내문)',
        'print(len(chunks), "조각")',
        '',
        '# 제목 구조대로 자르기 (제목이 꼬리표로 함께 저장된다)',
        'from langchain_text_splitters import MarkdownHeaderTextSplitter',
        'splitter2 = MarkdownHeaderTextSplitter(',
        '    headers_to_split_on=[("##", "섹션"), ("###", "소제목")])',
      ].join('\n')),

      quizBlock('t5/chunk', [
        {
          type: 'choice',
          q: 'chunk_overlap 을 두는 까닭은?',
          options: [
            '조각 수를 늘리려고',
            '답이 조각 경계에 걸려 잘리는 것을 막으려고',
            '검색 속도를 높이려고',
            '저장 공간을 아끼려고',
          ],
          answer: 1,
          why: '겹침이 있으면 경계에 걸친 내용이 어느 한 조각에는 온전히 담깁니다. 대신 저장량은 늘어납니다.',
        },
        {
          type: 'choice',
          q: '조각을 너무 크게 잡으면 생기는 문제는?',
          options: [
            '문맥이 끊긴다',
            '한 조각에 여러 주제가 섞여 검색이 뭉툭해진다',
            '겹침을 쓸 수 없다',
            '벡터를 만들 수 없다',
          ],
          answer: 1,
          why: '반대로 너무 작으면 문맥이 끊깁니다. 「하나의 완결된 뜻」이 담기는 크기를 찾는 것이 목표입니다.',
        },
        {
          type: 'choice',
          q: '조각 크기를 바꾸면 무엇을 다시 해야 하나요?',
          options: [
            '아무것도 안 해도 된다',
            '프롬프트만 고치면 된다',
            '벡터를 다시 만들어 저장해야 한다(재인덱싱)',
            '모델을 다시 학습시켜야 한다',
          ],
          answer: 2,
          why: '조각이 달라지면 벡터도 전부 달라집니다. 그래서 자르는 방식은 처음에 잘 정해 두는 편이 좋습니다.',
        },
      ]),
      nextHint('RAG 5단계로 →', () => ctx.go('rag', 'pipeline')),
    );
    draw();
    runTest();
    return wrap;
  },
};

/* ══════════════════════════ 3. RAG 5단계 ══════════════════════════════ */
const pipelineScreen = {
  id: 'pipeline',
  title: 'RAG 5단계',
  render(ctx) {
    const wrap = h('div');
    const state = { q: SAMPLE_QUESTIONS[0].q, k: 3, strict: true };
    const out = h('div');

    const draw = () => {
      const idx = getIndex();
      const r = retrieve(idx, state.q, { k: state.k, mode: 'hybrid' });
      const a = answerFromContext(state.q, r.hits, { strict: state.strict, cite: true });
      const naive = answerWithoutContext(state.q, NAIVE_GUESSES);
      const prompt = buildPrompt(state.q, r.hits, { role: true, only: true, refuse: state.strict, short: true });

      out.textContent = '';
      out.append(
        /* ── 비교 ── */
        h('div.cols',
          h('div.card', { style: { margin: 0, borderColor: 'var(--bad)' } },
            h('h3.card-title', { style: { background: '#2c1717' } }, '문서를 안 주고 물으면'),
            h('div.card-body',
              h('div.msg.msg-ai', h('span.msg-role', '모델'), h('div.msg-text', naive.text)),
              naive.wrong
                ? note('bad', h('b', '그럴듯하지만 실제 안내문과 다릅니다. '),
                  '이것이 ', b('환각(Hallucination)'), ' 입니다. 모델은 「다음에 올 법한 말」을 만들 뿐, ',
                  '사실인지 확인하는 장치가 없습니다.')
                : null,
            ),
          ),
          h('div.card', { style: { margin: 0, borderColor: 'var(--good)' } },
            h('h3.card-title', { style: { background: '#10291f' } }, '문서를 찾아서 주면'),
            h('div.card-body',
              h('div.msg.msg-ai', h('span.msg-role', '모델'),
                h('pre.msg-text', { style: { whiteSpace: 'pre-wrap', fontFamily: 'inherit' } }, a.text)),
              a.supported
                ? note('good', '안내문에 실제로 적힌 내용으로 답했습니다. 근거를 확인할 수 있습니다.')
                : note('info', '안내문에 없는 내용이라 「없다」고 답했습니다. ',
                  b('모른다고 말하는 것이 틀린 답보다 낫습니다.')),
            ),
          ),
        ),

        /* ── 5단계 ── */
        h('p', b('그 사이에 무슨 일이 있었나')),
        stepCard('① 자르기 (Load & Split)',
          `안내문을 제목 구조대로 ${idx.docs.length}조각으로 나눴습니다.`,
          h('div.pills', ...idx.docs.map((d) => h('span.pill', { style: { cursor: 'default' } }, d.metadata.섹션)))),
        stepCard('② 벡터로 만들어 저장 (Embed & Store)',
          '조각마다 벡터를 만들어 두었습니다. 여기까지는 질문이 오기 전에 미리 해 둡니다.',
          mono(`${idx.docs.length}개 × ${idx.docs[0].vector.length}차원`)),
        stepCard('③ 찾기 (Retrieve)',
          `질문을 벡터로 바꿔 가장 가까운 ${state.k}조각을 골랐습니다.`,
          h('div',
            a.expanded && a.expanded.length
              ? h('p', { style: { fontSize: '.88rem', color: 'var(--dim)' } },
                '질문 넓히기: ', ...a.expanded.map((x) => h('span.tok', `${x.from} → ${x.to}`)))
              : null,
            table(['순위', '조각', '점수'], r.hits.map((hh, i) => [
              i + 1, h('b', hh.doc.metadata.섹션), bar(Math.max(0, hh.score), { small: true, label: fx(hh.score, 3) }),
            ]), { compact: true }),
          )),
        stepCard('④ 프롬프트에 넣기 (Augment)',
          '찾은 조각을 [컨텍스트] 자리에 끼워 넣고, 규칙을 함께 적습니다.',
          code(prompt.length > 900 ? prompt.slice(0, 900) + '\n…(줄임)' : prompt, { copy: true })),
        stepCard('⑤ 답 만들기 (Generate)',
          '그 프롬프트를 모델에게 보내 답을 받습니다. 이때 온도는 0 으로 둡니다 — 지어내면 안 되니까요.',
          h('div.msg.msg-ai', h('span.msg-role', '모델'),
            h('pre.msg-text', { style: { whiteSpace: 'pre-wrap', fontFamily: 'inherit' } }, a.text))),
      );
    };

    wrap.append(
      screenHead('RAG 5단계', 'Retrieval(찾아서) · Augmented(넣고) · Generation(답한다) — 세 낱말이 곧 순서입니다.', '⑤ 내 문서로 답하기'),

      card('오픈북 시험에 비유하면',
        h('div.cols',
          h('div', h('p', b('닫힌 시험 = 그냥 물어보기')),
            h('p', { style: { color: 'var(--dim)' } },
              '머릿속 기억만으로 답을 씁니다. 기억이 틀렸거나 오래됐으면 그대로 틀립니다. ',
              '게다가 ', b('틀린 줄도 모르고 자신 있게'), ' 씁니다.')),
          h('div', h('p', b('오픈북 시험 = RAG')),
            h('p', { style: { color: 'var(--dim)' } },
              '관련 자료를 펴 놓고(검색) 보면서 답을 씁니다. 자료에 있는 대로 답하게 됩니다.')),
        ),
        note('good', h('b', '핵심 — 모델을 다시 학습시키지 않습니다. '),
          '모델은 그대로 두고 프롬프트에 지식을 얹어 줄 뿐이라 빠르고 쌉니다. ',
          '게다가 ', b('문서만 바꾸면 곧바로 최신 내용이 반영됩니다'), '. 재학습은 며칠이 걸리지만 문서 교체는 몇 초입니다.'),
        table(['얻는 것', '설명'], [
          ['환각 줄이기', '지어내지 않고 찾은 문서를 근거로 답한다'],
          ['최신 정보', '모델이 학습한 시점 이후의 내용도 문서만 넣으면 된다'],
          ['우리만의 자료', '모델이 본 적 없는 내부 문서로 답할 수 있다'],
          ['확인 가능', '어느 문서에서 나온 답인지 추적할 수 있다'],
        ]),
      ),

      card('직접 돌려 보기',
        h('div.pills', ...SAMPLE_QUESTIONS.map((q) => h('button.pill', {
          type: 'button',
          onclick: () => { state.q = q.q; wrap.querySelector('input.inp').value = q.q; draw(); },
        }, q.q, h('span.pos', q.tag)))),
        input({ value: state.q, onInput: (v) => { state.q = v; draw(); } }),
        h('div', { style: { margin: '8px 0' } },
          toggle('「없으면 없다고 답하라」 규칙 켜기', true, (v) => { state.strict = v; draw(); }),
        ),
        slider({ label: '몇 조각을 가져올까 (k)', min: 1, max: 6, value: 3, onInput: (v) => { state.k = v; draw(); } }),
        out,
        h('div.honest',
          h('b', '⑤의 「답 만들기」는 진짜 LLM 이 아닙니다. '),
          '찾은 조각에서 질문과 가장 맞는 문장을 골라 답을 조립합니다. ',
          '그래서 ', b('문서를 안 주면 못 답하고, 규칙을 끄면 지어냅니다'), ' — RAG 수업에서 봐야 할 성질은 그대로 재현됩니다. ',
          '①~④는 모두 진짜 계산입니다.'),
      ),

      card('규칙 한 줄의 힘',
        h('p', '위의 스위치를 껐다 켜 보세요. ', mono('주차장은 무료인가요?'), ' 같은 ',
          b('안내문에 없는 질문'), ' 에서 차이가 확 드러납니다.'),
        code([
          '[규칙 없음]',
          '  "일반적으로 두 시간까지 무료인 곳이 많습니다."   ← 지어냄',
          '',
          '[규칙 있음]',
          '  "컨텍스트에 없는 내용은 추측하지 말고,',
          '   \'안내되어 있지 않습니다\'라고 답하세요."',
          '  → "제공된 안내문에는 그 내용이 나와 있지 않습니다."   ← 정직',
        ].join('\n'), { copy: false }),
        note('warn', h('b', '「모른다」고 답하게 만드는 것이 오답보다 훨씬 낫습니다. '),
          '사용자가 틀린 정보를 믿고 행동하는 것이 가장 나쁜 결과이기 때문입니다.'),
      ),

      deepDive('RAG 는 언제 실패하는가',
        h('p', 'RAG 의 답은 결국 ', b('검색 품질'), ' 에 달려 있습니다. 엉뚱한 조각을 가져오면 답도 엉뚱해집니다.'),
        table(['어디서 무너지나', '무슨 일이 일어나나', '어떻게 고치나'], [
          ['검색', '관련 없는 조각이 들어가 답이 엉뚱해진다', '하이브리드 검색·리랭킹 (다음 화면)'],
          ['자르기', '조각이 부적절해 문맥이 뭉툭하거나 잘린다', '조각 크기 조절·구조 기반 자르기 (앞 화면)'],
          ['문서 부재', '아예 관련 문서가 없다', '「없다」고 답하게 하기·기준 점수 두기'],
          ['프롬프트', '지시가 약해 규칙을 무시하고 지어낸다', '규칙 강화·예시 넣기 (마지막 화면)'],
        ]),
        h('p', b('Garbage In, Garbage Out — '), '들어간 것이 쓰레기면 나오는 것도 쓰레기입니다. ',
          'RAG 를 개선한다는 말은 대개 ', b('검색을 개선한다'), ' 는 뜻입니다.'),
      ),

      pyBox([
        'from langchain_core.runnables import RunnablePassthrough',
        'from langchain_core.output_parsers import StrOutputParser',
        '',
        '# ① 자르기  ② 벡터 저장  ③ 검색기',
        'chunks = splitter.split_text(안내문)',
        'vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding)',
        'retriever = vectorstore.as_retriever(search_kwargs={"k": 3})',
        '',
        '# ④ 프롬프트',
        'prompt = ChatPromptTemplate.from_template(',
        '    "아래 컨텍스트에만 근거해 답하세요. 없으면 \'모른다\'고 답하세요.\\n\\n"',
        '    "[컨텍스트]\\n{context}\\n\\n[질문]\\n{question}")',
        '',
        '# 찾은 여러 조각을 하나의 문자열로 합치기',
        'def format_docs(docs):',
        '    return "\\n\\n".join(d.page_content for d in docs)',
        '',
        '# ⑤ 이어 붙이기',
        'rag_chain = (',
        '    {"context": retriever | format_docs, "question": RunnablePassthrough()}',
        '    | prompt | llm | StrOutputParser()',
        ')',
        'print(rag_chain.invoke("책은 며칠까지 빌릴 수 있나요?"))',
      ].join('\n'), 'RunnablePassthrough() 는 「입력을 그대로 통과시켜라」는 뜻입니다'),

      quizBlock('t5/pipeline', [
        {
          type: 'choice',
          q: 'RAG 의 세 낱말을 순서대로 바르게 늘어놓은 것은?',
          options: [
            '생성 → 검색 → 증강',
            '검색 → 증강 → 생성',
            '증강 → 생성 → 검색',
            '검색 → 생성 → 증강',
          ],
          answer: 1,
          why: '관련 문서를 찾고(Retrieval), 프롬프트에 넣고(Augmented), 그것을 근거로 답합니다(Generation).',
        },
        {
          type: 'choice',
          q: 'RAG 를 쓸 때 모델을 다시 학습시켜야 하나요?',
          options: [
            '그렇다. 문서를 넣으려면 재학습이 필요하다',
            '아니다. 프롬프트에 문서를 얹어 줄 뿐이다',
            '문서가 100개 넘으면 필요하다',
            '한국어일 때만 필요하다',
          ],
          answer: 1,
          why: '이것이 RAG 의 가장 큰 장점입니다. 문서만 바꾸면 몇 초 만에 최신 내용이 반영됩니다.',
        },
        {
          type: 'choice',
          q: '「미리 해 두는 일(오프라인)」과 「질문마다 하는 일(실행)」을 바르게 나눈 것은?',
          options: [
            '미리: 자르기·벡터 저장 / 질문마다: 검색·증강·생성',
            '미리: 검색·생성 / 질문마다: 자르기·저장',
            '전부 질문마다 한다',
            '전부 미리 해 둔다',
          ],
          answer: 0,
          why: '문서가 바뀌지 않는 한 자르기와 저장은 한 번만 하면 됩니다. 검색부터가 매 질문 반복됩니다.',
        },
        {
          type: 'short',
          q: () => ['모델이 사실이 아닌 내용을 자신 있게 만들어 내는 현상을 무엇이라고 하나요? (세 글자)'],
          accept: ['환각', 'hallucination', '할루시네이션'],
          why: '환각(Hallucination)입니다. 모델은 「다음에 올 법한 말」을 만들 뿐 사실 여부를 확인하지 않습니다.',
        },
      ]),
      nextHint('검색 품질 올리기로 →', () => ctx.go('rag', 'improve')),
    );
    draw();
    return wrap;
  },
};

function stepCard(title, lead, body) {
  return h('div', { style: { borderLeft: '3px solid var(--accent)', paddingLeft: '14px', margin: '14px 0' } },
    h('p', { style: { margin: '0 0 2px' } }, b(title)),
    h('p', { style: { color: 'var(--dim)', fontSize: '.9rem', margin: '0 0 6px' } }, lead),
    body,
  );
}

/* ══════════════════════════ 4. 검색 품질 올리기 ═══════════════════════ */
const improveScreen = {
  id: 'improve',
  title: '검색 품질 올리기',
  render(ctx) {
    const wrap = h('div');
    const state = {
      q: '연체하면 어떻게 되나요', mode: 'hybrid', wv: 0.6, mmr: false,
      lambda: 0.5, filter: '', threshold: 0,
    };
    const out = h('div');
    const idx = getIndex();
    const sections = [...new Set(idx.docs.map((d) => d.metadata.섹션))];

    const draw = () => {
      const r = retrieve(idx, state.q, {
        k: 3, mode: state.mode, weightVector: state.wv, mmr: state.mmr,
        lambda: state.lambda, fetchK: 6,
        filter: state.filter ? { 섹션: state.filter } : null,
        threshold: state.threshold || undefined,
      });
      out.textContent = '';
      out.append(
        h('p', b('고른 조각')),
        r.hits.length
          ? table(['순위', '조각', '뜻 점수', '낱말 점수', '최종'], r.hits.map((hh, i) => [
            i + 1, h('b', hh.doc.metadata.섹션), fx(hh.vec, 3), fx(hh.kw, 3),
            bar(Math.max(0, hh.score), { small: true, label: fx(hh.score, 3) }),
          ]))
          : note('warn', r.note || '조건에 맞는 조각이 없습니다.'),
        r.mmrTrace
          ? h('div',
            h('p', b('MMR 이 고르는 과정')),
            table(['차례', '고른 조각', '질문과의 관련', '이미 고른 것과 겹침', 'MMR 점수'],
              r.mmrTrace.map((t, i) => [
                i + 1, idx.docs[t.i].metadata.섹션,
                fx(r.scores[t.i]?.score ?? 0, 3), fx(t.redundancy, 3), fx(t.score, 3),
              ]), { compact: true }))
          : null,
        h('p', b('모든 조각의 점수')),
        table(['조각', '뜻으로', '낱말로', '섞은 점수'],
          r.scores.map((s) => [
            idx.docs[s.id].metadata.섹션,
            bar(Math.max(0, s.vec), { small: true, label: fx(s.vec, 2) }),
            bar(Math.max(0, s.kw), { small: true, label: fx(s.kw, 2) }),
            bar(Math.max(0, s.score), { small: true, label: fx(s.score, 2) }),
          ]), { compact: true }),
      );
    };

    wrap.append(
      screenHead('검색 품질 올리기', '증상마다 손잡이가 따로 있습니다. 문제가 되는 것만 하나씩 돌리세요.', '⑤ 내 문서로 답하기'),

      card('증상별 손잡이 지도',
        table(['이런 증상이면', '이 손잡이를 돌린다', '어느 단계'], [
          ['조각이 뭉툭하거나 조각조각 잘린다', '자르는 방식 (앞 화면)', '색인'],
          ['고유명사·번호를 못 찾는다', '하이브리드 검색', '검색'],
          ['비슷한 조각만 세 개 나온다', 'MMR (다양성)', '검색'],
          ['엉뚱한 범위까지 뒤진다', '꼬리표(메타데이터) 거르기', '검색'],
          ['관련 없는 조각도 억지로 가져온다', '기준 점수(임계값)', '검색'],
          ['근거 없이 지어낸다', '프롬프트 강화 (다음 화면)', '생성'],
        ]),
        note('warn', h('b', '전부 다 넣을 필요는 없습니다. '),
          '넣을수록 느려지고 비싸집니다. ', b('한 번에 하나씩 넣고, 좋아졌는지 숫자로 확인한 뒤에만'), ' 남깁니다.'),
      ),

      card('손잡이를 직접 돌려 보기',
        input({ value: state.q, onInput: (v) => { state.q = v; draw(); } }),
        h('div.pills', ...[
          '연체하면 어떻게 되나요',
          '자리 예약과 이용 시간이 궁금해요',
          '전자책',
          '책을 잃어버렸어요',
        ].map((t) => h('button.pill', {
          type: 'button',
          onclick: () => { state.q = t; wrap.querySelector('input.inp').value = t; draw(); },
        }, t))),
        h('p', b('① 무엇으로 찾을까')),
        pillGroup([
          { label: '뜻으로만', value: 'vector' },
          { label: '낱말로만', value: 'keyword' },
          { label: '둘을 섞어 (하이브리드)', value: 'hybrid' },
        ], (v) => { state.mode = v; draw(); }, 'hybrid'),
        slider({
          label: '섞는 비중 — 뜻 쪽', min: 0, max: 1, step: 0.05, value: 0.6,
          format: (v) => `뜻 ${v.toFixed(2)} : 낱말 ${(1 - v).toFixed(2)}`,
          onInput: (v) => { state.wv = v; draw(); },
        }),
        h('p', b('② 겹치지 않게 고를까 (MMR)')),
        toggle('MMR 켜기 — 관련은 있으면서 서로 다른 조각 고르기', false, (v) => { state.mmr = v; draw(); }),
        slider({
          label: '관련성 ↔ 다양성', min: 0, max: 1, step: 0.05, value: 0.5,
          format: (v) => (v > 0.75 ? `관련성 우선 ${v.toFixed(2)}` : v < 0.25 ? `다양성 우선 ${v.toFixed(2)}` : `균형 ${v.toFixed(2)}`),
          onInput: (v) => { state.lambda = v; draw(); },
        }),
        h('p', b('③ 범위를 좁힐까 (꼬리표)')),
        pillGroup([{ label: '전체', value: '' }, ...sections.map((s) => ({ label: s, value: s }))],
          (v) => { state.filter = v; draw(); }),
        h('p', b('④ 점수가 낮으면 버릴까 (기준 점수)')),
        slider({
          label: '이 점수 아래는 안 가져오기', min: 0, max: 0.8, step: 0.05, value: 0,
          format: (v) => (v === 0 ? '끔' : v.toFixed(2)),
          onInput: (v) => { state.threshold = v; draw(); },
        }),
        out,
      ),

      card('각 손잡이가 하는 일',
        h('p', b('① 하이브리드 검색 — 서로의 약점 메우기')),
        code([
          '질문: "주문번호 A-1204 배송 상태"',
          '',
          '  뜻으로 찾기   : "배송 상태"의 뜻은 잘 잡지만 A-1204 는 놓친다',
          '  낱말로 찾기   : A-1204 는 정확히 잡지만 "배송 상태"의 동의어는 놓친다',
          '        ↓ 두 점수를 가중치로 섞는다',
          '  둘 다 잡는다',
        ].join('\n'), { copy: false }),
        h('p', '고유명사·코드가 중요한 자료라면 낱말 쪽 비중을, 자연스러운 문장 질문이 많다면 뜻 쪽 비중을 높입니다. ',
          mono('0.5 : 0.5'), ' 에서 시작해 조금씩 옮기며 재 보는 것이 정석입니다.'),

        h('p', b('② MMR — 같은 말만 세 개 나오는 것을 막기')),
        code([
          '보통 검색 (k=3):  [대출 A] [대출 A′] [대출 A″]   ← 사실상 같은 내용',
          '                   → 정작 필요한 「연체」나 「연장」은 빠진다',
          '',
          'MMR (k=3):        [대출] [연체] [연장]           ← 관련 있으면서 서로 다름',
        ].join('\n'), { copy: false }),
        h('p', mono('λ'), ' 가 1에 가까우면 보통 검색과 같아지고, 0에 가까우면 다양성만 좇습니다. ',
          '보통 0.5 에서 시작합니다.'),

        h('p', b('③ 꼬리표 거르기 — 아예 범위를 좁히기')),
        h('p', { style: { color: 'var(--dim)' } },
          '조각을 저장할 때 「이건 대출 부분」이라는 표를 함께 붙여 두면, ',
          '나중에 「대출 부분에서만 찾아라」가 가능해집니다. ',
          '제목 구조대로 자르면 이 표가 ', b('저절로'), ' 붙습니다(앞 화면).'),

        h('p', b('④ 기준 점수 — 없으면 없다고 하기')),
        h('p', { style: { color: 'var(--dim)' } },
          '기준 점수를 두면 관련 없는 조각을 아예 안 돌려줍니다. ',
          '가져올 것이 없으면 모델도 답할 거리가 없으니 ', b('억지 답변을 막는'), ' 효과가 있습니다. ',
          '위에서 ', mono('전자책'), ' 을 검색하고 기준 점수를 0.6까지 올려 보세요.'),
      ),

      deepDive('리랭킹 — 넓게 찾고 정밀하게 고르기',
        h('p', '벡터 검색은 빠르지만 거칩니다. 반대로 「이 조각이 이 질문에 정말 맞는가」를 ',
          '꼼꼼히 따지는 모델은 정확하지만 느립니다. 그래서 두 단계로 나눕니다.'),
        code([
          '1단계 — 넓게 (놓치지 않는 것이 목표)',
          '   질문 → 벡터 검색 → 후보 20개      빠르지만 거칠다',
          '',
          '2단계 — 정밀하게 (정확한 것이 목표)',
          '   후보 20개 → 재채점 → 상위 3개만 통과',
          '        ↓',
          '   진짜 관련 있는 3개만 모델에게 준다',
        ].join('\n'), { copy: false }),
        h('p', b('대가는 속도와 비용입니다. '), '후보마다 추가 계산이 들어갑니다. ',
          '간단한 사내 챗봇이라면 ', b('좋은 자르기 + 하이브리드'), ' 만으로 충분한 경우가 많습니다. ',
          '정확도가 서비스의 생명일 때만 도입하세요.'),
        h('p', b('트레이드오프를 기억할 것 — '),
          '정확도·속도·비용은 서로 잡아당깁니다. k 를 늘리면 놓칠 확률은 줄지만 프롬프트가 길어져 비싸집니다. ',
          '「서비스에 필요한 만큼만」이 정답입니다.'),
      ),

      pyBox([
        '# ① 하이브리드 검색 (낱말 + 뜻)',
        'from langchain_community.retrievers import BM25Retriever',
        'from langchain.retrievers import EnsembleRetriever',
        '',
        'bm25 = BM25Retriever.from_texts(chunks); bm25.k = 3',
        'vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})',
        'ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever],',
        '                             weights=[0.4, 0.6])',
        '',
        '# ② MMR — 겹치지 않게',
        'mmr = vectorstore.as_retriever(search_type="mmr",',
        '        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})',
        '',
        '# ③ 꼬리표로 범위 좁히기',
        'filtered = vectorstore.as_retriever(',
        '        search_kwargs={"k": 3, "filter": {"섹션": "대출"}})',
        '',
        '# ④ 기준 점수',
        'thresholded = vectorstore.as_retriever(',
        '        search_type="similarity_score_threshold",',
        '        search_kwargs={"score_threshold": 0.5, "k": 3})',
      ].join('\n')),

      quizBlock('t5/improve', [
        {
          type: 'choice',
          q: '하이브리드 검색이 필요한 대표적인 상황은?',
          options: [
            '문서가 아주 짧을 때',
            '제품 코드·주문번호 같은 고유명사가 중요한 자료일 때',
            '한국어 문서일 때',
            '문서가 하나뿐일 때',
          ],
          answer: 1,
          why: '임베딩은 「비슷한 뜻」을 찾지 「정확히 그 글자」를 찾지 않습니다. 그 약점을 낱말 검색이 메웁니다.',
        },
        {
          type: 'choice',
          q: 'MMR 이 해결하는 문제는?',
          options: [
            '검색이 느린 문제',
            '거의 같은 내용의 조각만 여러 개 뽑히는 문제',
            '문서가 너무 긴 문제',
            '모델이 지어내는 문제',
          ],
          answer: 1,
          why: '「관련은 있으면서 서로 다른」 조각을 뽑아, 프롬프트에 같은 정보가 세 번 들어가는 낭비를 막습니다.',
        },
        {
          type: 'choice',
          q: 'MMR 의 λ 를 1에 가깝게 두면?',
          options: [
            '다양성만 좇는다',
            '보통 검색과 거의 같아진다',
            '아무것도 안 뽑힌다',
            '검색이 느려진다',
          ],
          answer: 1,
          why: 'λ 는 관련성의 비중입니다. 1이면 다양성 항이 사라져 그냥 관련성 순으로 뽑게 됩니다.',
        },
        {
          type: 'choice',
          q: '고도화 기법을 도입할 때의 원칙으로 가장 알맞은 것은?',
          options: [
            '한꺼번에 다 넣는다',
            '증상에 맞는 것을 하나씩 넣고, 전후를 숫자로 비교해 좋아진 것만 남긴다',
            '비싼 기법부터 넣는다',
            '남들이 쓰는 것을 그대로 쓴다',
          ],
          answer: 1,
          why: '기법을 넣었는데 오히려 나빠지는 일도 흔합니다. 재지 않으면 개선인지 알 수 없습니다.',
        },
      ]),
      nextHint('근거와 출처로 →', () => ctx.go('rag', 'evidence')),
    );
    draw();
    return wrap;
  },
};

/* ══════════════════════════ 5. 근거와 출처 ════════════════════════════ */
const evidenceScreen = {
  id: 'evidence',
  title: '근거와 출처',
  render(ctx) {
    const wrap = h('div');
    const parts = { role: true, only: true, refuse: true, short: false, cite: true, fewshot: false };
    const state = { q: '주차장은 무료인가요?' };
    const promptBox = h('div');
    const ansBox = h('div');
    const idx = getIndex();

    const draw = () => {
      const r = retrieve(idx, state.q, { k: 3, mode: 'hybrid' });
      const a = answerFromContext(state.q, r.hits, {
        strict: parts.only && parts.refuse, cite: parts.cite, short: parts.short,
        guess: answerWithoutContext(state.q, NAIVE_GUESSES).text,
      });
      promptBox.textContent = '';
      promptBox.appendChild(code(buildPrompt(state.q, r.hits, parts), { copy: true }));
      ansBox.textContent = '';
      ansBox.append(
        h('div.msg.msg-ai', h('span.msg-role', '모델'),
          h('pre.msg-text', { style: { whiteSpace: 'pre-wrap', fontFamily: 'inherit' } }, a.text)),
        a.hallucinated
          ? note('bad', h('b', '지어냈습니다. '), '「없으면 없다고 답하라」 규칙이 꺼져 있기 때문입니다.')
          : a.supported
            ? note('good', '찾은 문서에 실제로 적힌 내용으로 답했습니다.')
            : note('info', '문서에 없어서 「없다」고 답했습니다. 이것이 옳은 행동입니다.'),
        parts.cite && a.used?.length
          ? h('p', { style: { fontSize: '.88rem', color: 'var(--dim)' } },
            '📎 근거로 쓴 조각: ', ...a.used.map((d) => h('span.tok', d.metadata.섹션 || `조각 ${d.id + 1}`)))
          : null,
      );
    };

    /* 정답 세트로 재기 */
    const evalBox = h('div');
    const runEval = () => {
      const rows = DOC_FACTS.map((f) => {
        const r = retrieve(idx, f.question, { k: 3, mode: 'hybrid' });
        const a = answerFromContext(f.question, r.hits, { strict: true });
        const ok = f.accept.some((x) => a.text.includes(x));
        return { f, a, ok };
      });
      const noAns = NO_ANSWER_QUESTIONS.map((q) => {
        const r = retrieve(idx, q, { k: 3, mode: 'hybrid' });
        const a = answerFromContext(q, r.hits, { strict: true });
        return { q, refused: !a.supported, text: a.text };
      });
      const hit = rows.filter((r) => r.ok).length;
      evalBox.textContent = '';
      evalBox.append(
        h('div.stat-row',
          statBox('답이 있는 질문', `${hit} / ${rows.length}`, hit === rows.length ? 'good' : 'accent'),
          statBox('답이 없는 질문에서 「모른다」', `${noAns.filter((n) => n.refused).length} / ${noAns.length}`,
            noAns.every((n) => n.refused) ? 'good' : 'bad'),
        ),
        table(['질문', '안내문에 적힌 값', '모델의 답', ''], rows.map((r) => [
          r.f.question, mono(r.f.value),
          h('span', { style: { fontSize: '.88rem' } }, r.a.text.slice(0, 60) + (r.a.text.length > 60 ? '…' : '')),
          r.ok ? h('span', { style: { color: 'var(--good)' } }, '⭕') : h('span', { style: { color: 'var(--bad)' } }, '❌'),
        ]), { compact: true }),
        table(['안내문에 없는 질문', '모델의 답', ''], noAns.map((n) => [
          n.q, h('span', { style: { fontSize: '.88rem' } }, n.text),
          n.refused ? h('span', { style: { color: 'var(--good)' } }, '⭕ 모른다고 함')
            : h('span', { style: { color: 'var(--bad)' } }, '❌ 지어냄'),
        ]), { compact: true }),
      );
    };

    wrap.append(
      screenHead('근거와 출처', '검색을 아무리 잘해도 프롬프트가 약하면 다시 지어냅니다.', '⑤ 내 문서로 답하기'),

      card('프롬프트 조각을 켜고 끄기',
        h('div.pills', ...['주차장은 무료인가요?', '책은 며칠까지 빌릴 수 있나요?', '연체하면 돈을 내야 하나요?']
          .map((q) => h('button.pill', {
            type: 'button',
            onclick: () => { state.q = q; wrap.querySelector('input.inp').value = q; draw(); },
          }, q))),
        input({ value: state.q, onInput: (v) => { state.q = v; draw(); } }),
        h('div', { style: { margin: '10px 0' } },
          ...Object.entries(PROMPT_PARTS).map(([key, p]) =>
            toggle(p.label, parts[key], (v) => { parts[key] = v; draw(); })),
        ),
        h('p', b('완성된 프롬프트')),
        promptBox,
        h('p', b('돌아온 답')),
        ansBox,
        note('warn', h('b', '「없으면 없다고 답하라」를 꺼 보세요. '),
          '안내문에 없는 질문에서 곧바로 지어내기 시작합니다. ',
          '이 한 줄이 환각을 막는 가장 큰 방어선입니다.'),
      ),

      card('환각을 막는 문장들',
        table(['넣는 문장', '무엇을 막나'], [
          ['「반드시 아래 컨텍스트에 있는 내용만으로 답하세요」', '모델이 자기 기억으로 답하는 것'],
          ['「없는 내용은 추측하지 말고 안내되어 있지 않다고 하세요」', '억지로 답을 만들어 내는 것'],
          ['「답변은 세 문장 이내로」', '길게 늘어놓다가 근거에서 벗어나는 것'],
          ['「근거가 된 부분의 제목을 함께 적으세요」', '확인할 수 없는 답'],
          ['모르는 질문에 대한 예시 하나', '「모른다」고 말하는 방법 자체를 모르는 것'],
        ]),
        h('p', b('예시 하나가 설명 열 줄보다 강합니다.')),
        code([
          '[예시]',
          '컨텍스트: (대출·반납 안내만 있음)',
          '질문: 주차는 무료인가요?',
          '답변: 제공된 안내문에는 주차에 관한 내용이 없습니다.',
        ].join('\n'), { copy: false }),
      ),

      card('출처를 함께 돌려주기',
        h('p', 'RAG 의 큰 장점 하나는 ', b('어디서 나온 답인지 확인할 수 있다'), ' 는 것입니다. ',
          '사람이 검증할 수 있으면 신뢰가 올라갑니다.'),
        code([
          '답변: 대출 기간은 21일이며, 한 번에 한해 7일 연장할 수 있습니다.',
          '출처: [대출] 섹션                    ← 직접 확인 가능',
        ].join('\n'), { copy: false }),
        h('p', { style: { color: 'var(--dim)' } },
          '조각을 저장할 때 붙여 둔 꼬리표(파일 이름·쪽 번호·섹션 제목)를 답과 함께 내보내면 됩니다.'),
      ),

      card('좋아졌는지 어떻게 아나 — 재 보기',
        h('p', h('b', '가장 중요한 원칙입니다. '), '「느낌」으로 튜닝하면 안 됩니다.'),
        h('ol',
          h('li', '실제 사용자가 물을 법한 ', b('질문–정답 세트'), ' 를 20~30개 만든다'),
          h('li', '기법을 넣기 ', b('전과 후'), ' 의 답을 비교한다'),
          h('li', '좋아진 것이 ', b('숫자로 확인된 뒤에만'), ' 채택한다'),
        ),
        h('p', { style: { color: 'var(--dim)' } },
          '아래는 이 안내문으로 만든 작은 정답 세트입니다. ',
          '답이 있는 질문 7개와, 답이 ', b('없는'), ' 질문 2개를 함께 넣었습니다. ',
          '없는 질문에 「모른다」고 답하는지도 반드시 재야 합니다.'),
        button('정답 세트로 재 보기', runEval),
        evalBox,
      ),

      deepDive('그 다음은 어디로 가나',
        terms([
          ['평가 자동화', '질문–정답 세트를 늘리고, 답이 근거에 실제로 들어 있는지를 자동으로 채점한다. 사람이 매번 읽을 수 없기 때문이다.'],
          ['대화형 RAG', '「그럼 연장은?」처럼 앞 대화에 기대는 질문을 처리하려면, 검색하기 전에 질문을 완전한 문장으로 다시 써야 한다.'],
          ['에이전트로 확장', '검색할지, 계산할지, 다른 도구를 부를지를 모델이 스스로 판단하게 한다 → 탭 ⑥'],
        ]),
        h('p', b('대화형 RAG 의 함정 — '),
          '「그럼 연장은?」을 그대로 검색하면 아무것도 안 나옵니다. ',
          '앞 대화를 보고 「대출 연장은 어떻게 하나요?」로 바꾼 뒤 검색해야 합니다. ',
          '이 「질문 다시 쓰기」 단계를 빠뜨려서 대화형 RAG 가 망가지는 일이 아주 흔합니다.'),
      ),

      pyBox([
        'from langchain_core.runnables import RunnableParallel, RunnablePassthrough',
        '',
        'prompt = ChatPromptTemplate.from_template(',
        '    "당신은 도서관 안내 도우미입니다.\\n"',
        '    "규칙:\\n"',
        '    "1) 반드시 아래 [컨텍스트]에 있는 내용만으로 답하세요.\\n"',
        '    "2) 없는 내용은 추측하지 말고 \'안내되어 있지 않습니다\'라고 답하세요.\\n"',
        '    "3) 답변은 세 문장 이내로 간결하게 하세요.\\n\\n"',
        '    "[컨텍스트]\\n{context}\\n\\n[질문]\\n{question}")',
        '',
        '# 답변과 출처를 함께 돌려받기',
        'answer_chain = (RunnablePassthrough.assign(',
        '        context=lambda x: format_docs(x["context"]))',
        '    | prompt | llm | StrOutputParser())',
        '',
        'rag_with_source = RunnableParallel(',
        '    context=retriever, question=RunnablePassthrough()',
        ").assign(answer=answer_chain)",
        '',
        'result = rag_with_source.invoke("반납은 며칠 이내인가요?")',
        'print(result["answer"])',
        'for doc in result["context"]:',
        '    print("출처:", doc.metadata)',
      ].join('\n')),

      quizBlock('t5/evidence', [
        {
          type: 'choice',
          q: '환각을 막는 데 가장 큰 몫을 하는 프롬프트 문장은?',
          options: [
            '「친절하게 답하세요」',
            '「없는 내용은 추측하지 말고 안내되어 있지 않다고 답하세요」',
            '「한국어로 답하세요」',
            '「빠르게 답하세요」',
          ],
          answer: 1,
          why: '「모른다」고 말할 길을 열어 주는 것이 핵심입니다. 이 길이 없으면 모델은 무엇이든 만들어 냅니다.',
        },
        {
          type: 'choice',
          q: '고도화 기법을 넣기 전후로 반드시 해야 하는 일은?',
          options: [
            '문서를 다시 쓴다',
            '질문–정답 세트로 재서 좋아졌는지 확인한다',
            '모델을 바꾼다',
            '온도를 올린다',
          ],
          answer: 1,
          why: '재지 않으면 개선인지 개악인지 알 수 없습니다. 기법을 넣었는데 나빠지는 일도 흔합니다.',
        },
        {
          type: 'choice',
          q: '정답 세트를 만들 때 「답이 없는 질문」도 넣어야 하는 까닭은?',
          options: [
            '문항 수를 늘리려고',
            '모르는 질문에 「모른다」고 답하는지 재야 하니까',
            '검색 속도를 재려고',
            '넣을 필요 없다',
          ],
          answer: 1,
          why: '답을 잘 맞히는 것만큼 「모를 때 모른다고 하는 것」이 중요합니다. 그것도 성능의 일부입니다.',
        },
      ]),
      nextHint('⑥ 스스로 판단하기로 →', () => ctx.go('agent', 'graph')),
    );
    draw();
    runEval();
    return wrap;
  },
};

export default {
  id: 'rag',
  num: 'Ⅴ',
  title: '내 문서로 답하기',
  screens: [searchScreen, chunkScreen, pipelineScreen, improveScreen, evidenceScreen],
};
