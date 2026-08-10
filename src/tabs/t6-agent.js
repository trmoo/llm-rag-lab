/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ⑥ 스스로 판단하기 — 갈림길과 되돌아오기가 있는 흐름
 *   1. 상태·노드·엣지   그래프의 세 부품과 「덮어쓰기 vs 이어붙이기」
 *   2. 갈림길과 루프     조건 분기·반복, 그리고 무한 루프를 막는 안전장치
 *   3. 도구를 쓰는 에이전트  생각 → 도구 → 다시 생각
 *   4. 멀티턴과 멀티에이전트 시간축(기억)과 역할축(분담)
 * ========================================================================== */

import {
  h, card, note, deepDive, terms, table, screenHead, button, input,
  toggle, pillGroup, code, pyBox, mono, b, nextHint, fx, slider, onResize,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import { stateGraph, memorySaver, addReducer, addMessages, START, END, toMermaid } from '../lib/graph.js';
import { retrieve, answerFromContext, buildIndex, splitByHeadings } from '../lib/rag.js';
import { LIBRARY_DOC } from '../data/docs.js';

const statBox = (k, v, kind = '') => h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));

/** 실행 기록을 표로 */
function traceTable(trace, keys) {
  return table(['걸음', '어디를 지났나', ...keys], trace.map((t, i) => [
    i,
    t.node === '⟨판단⟩'
      ? h('span', { style: { color: 'var(--accent2)' } }, `판단: ${t.router} → 「${t.label}」 → ${t.to}`)
      : h('b', t.node),
    ...keys.map((k) => {
      const v = t.state?.[k];
      return h('span', { style: { fontFamily: 'var(--mono)', fontSize: '.86rem' } },
        Array.isArray(v) ? `[${v.map((x) => (typeof x === 'object' ? x.text ?? JSON.stringify(x) : x)).join(', ')}]` : String(v ?? ''));
    }),
  ]), { compact: true });
}

/* ══════════════════════════ 1. 상태·노드·엣지 ═════════════════════════ */
const graphScreen = {
  id: 'graph',
  title: '상태·노드·엣지',
  render(ctx) {
    const wrap = h('div');
    const state = { start: 3, reducer: true };
    const out = h('div');

    const run = () => {
      const g = stateGraph({ log: state.reducer ? addReducer : undefined })
        .addNode('하나 더하기', (s) => ({ number: s.number + 1, log: ['+1 을 했다'] }))
        .addNode('두 배 하기', (s) => ({ number: s.number * 2, log: ['×2 를 했다'] }))
        .addEdge(START, '하나 더하기')
        .addEdge('하나 더하기', '두 배 하기')
        .addEdge('두 배 하기', END)
        .compile();
      const r = g.invoke({ number: state.start, log: [] });
      out.textContent = '';
      out.append(
        h('div.flow',
          h('span.flow-step', '시작'), h('span.flow-arrow', '→'),
          h('span.flow-step.done', '하나 더하기'), h('span.flow-arrow', '→'),
          h('span.flow-step.done', '두 배 하기'), h('span.flow-arrow', '→'),
          h('span.flow-step', '끝'),
        ),
        traceTable(r.trace, ['number', 'log']),
        h('p', b('결과 '), mono(JSON.stringify(r.state))),
        state.reducer
          ? note('good', h('b', '이어붙이기 규칙이 켜져 있습니다. '),
            'log 칸에 두 노드의 기록이 모두 쌓였습니다.')
          : note('warn', h('b', '규칙이 없으면 덮어씁니다. '),
            'log 칸에 마지막 노드의 기록만 남고 앞의 것은 사라졌습니다. ',
            '여러 노드가 같은 칸을 건드릴 때 반드시 주의해야 할 부분입니다.'),
      );
    };

    wrap.append(
      screenHead('상태·노드·엣지', '직선으로만 흐르던 파이프라인에 갈림길과 되돌아오기를 넣으려면 순서도가 필요합니다.', '⑥ 스스로 판단하기'),

      card('왜 직선으로는 부족한가',
        code([
          '[지금까지] 입력 → 검색 → 모델 → 출력          갈림길도 되돌아옴도 없다',
          '',
          '[필요한 것]',
          '   갈림길 : 질문 종류에 따라 다른 길로',
          '   되돌아옴: 조건이 될 때까지 같은 일을 반복',
          '   공유 기억: 여러 단계가 같은 메모장을 보며 조금씩 고쳐 감',
        ].join('\n'), { copy: false }),
        h('p', '이런 것을 표현하려면 직선이 아니라 ', b('순서도(그래프)'), ' 가 필요합니다.'),
      ),

      card('세 부품이 전부입니다',
        table(['부품', '무엇인가', '비유'], [
          [h('b', '상태 (State)'), '모든 노드가 함께 보는 공유 데이터', '📋 다 같이 보는 메모장'],
          [h('b', '노드 (Node)'), '상태를 받아 「바뀔 부분만」 돌려주는 함수', '🧑‍💼 일하는 담당자'],
          [h('b', '엣지 (Edge)'), '노드에서 노드로 가는 화살표', '➡️ 흐름선'],
        ]),
        note('info', h('b', '노드는 전체 상태를 돌려줄 필요가 없습니다. '),
          '바뀌는 칸만 돌려주면 나머지는 그대로 유지됩니다. 그래서 코드가 짧아집니다.'),
      ),

      card('직접 돌려 보기',
        h('p', { style: { color: 'var(--dim)' } },
          '노드 두 개짜리 그래프입니다. 숫자가 노드를 지나며 어떻게 바뀌는지 표로 따라가 보세요.'),
        slider({ label: '시작 숫자', min: 0, max: 20, value: 3, onInput: (v) => { state.start = v; run(); } }),
        toggle('log 칸에 「이어붙이기」 규칙 주기', true, (v) => { state.reducer = v; run(); }),
        out,
      ),

      card('덮어쓰기 vs 이어붙이기 (Reducer)',
        h('p', '이 부분이 가장 헷갈립니다. 노드가 어떤 칸의 값을 돌려주면 ', b('기본은 덮어쓰기'), ' 입니다.'),
        code([
          '■ 덮어쓰기 (기본)',
          '   A: 기록 = ["A 실행"]  →  B: 기록 = ["B 실행"]  →  결과 ["B 실행"]',
          '                                                     (A 가 사라짐 ❌)',
          '',
          '■ 이어붙이기 (규칙을 붙이면)',
          '   A: 기록 = ["A 실행"]  →  B: 기록 = ["B 실행"]  →  결과 ["A 실행", "B 실행"]',
          '                                                     (쌓임 ✅)',
        ].join('\n'), { copy: false }),
        h('p', '대화 기록처럼 ', b('쌓여야 하는 칸'), ' 에는 반드시 이어붙이기 규칙을 붙입니다. ',
          '덮어쓰면 앞 대화가 통째로 날아갑니다.'),
      ),

      deepDive('상태 설계가 그래프 설계다',
        h('p', '그래프를 만들 때 가장 먼저 정하는 것은 노드가 아니라 ', b('상태'), ' 입니다. ',
          '「이 흐름이 무슨 데이터를 들고 다니는가」가 곧 설계도이기 때문입니다.'),
        code([
          'class 상태(TypedDict):',
          '    messages: Annotated[list, add_messages]  # 대화 — 이어붙이기',
          '    next: str                                # 다음 담당자 — 덮어쓰기',
          '    검색결과: list                            # 이번 턴에만 쓰는 값 — 덮어쓰기',
        ].join('\n'), { copy: false }),
        h('p', b('칸마다 규칙이 다를 수 있습니다. '),
          '대화는 쌓고, 「다음에 갈 곳」은 덮어씁니다. 이 구분을 잘못하면 ',
          '대화가 사라지거나 옛 결정이 계속 남아 흐름이 엉킵니다.'),
        h('p', b('그래프를 그림으로 확인하기 — '),
          '복잡해질수록 「내가 만든 그래프가 정말 이렇게 생겼나」를 눈으로 봐야 합니다. ',
          '대부분의 도구가 그래프 구조를 그림으로 뽑아 주는 기능을 제공합니다.'),
      ),

      pyBox([
        'from typing import Annotated',
        'from typing_extensions import TypedDict',
        'from langgraph.graph import StateGraph, START, END',
        'import operator',
        '',
        '# ① 상태 — 이어붙일 칸에는 규칙(reducer)을 붙인다',
        'class 상태(TypedDict):',
        '    number: int',
        '    log: Annotated[list, operator.add]',
        '',
        '# ② 노드 — 바뀔 부분만 돌려준다',
        'def 하나_더하기(s): return {"number": s["number"] + 1, "log": ["+1"]}',
        'def 두_배(s):       return {"number": s["number"] * 2, "log": ["x2"]}',
        '',
        '# ③ 조립',
        'builder = StateGraph(상태)',
        'builder.add_node("add", 하나_더하기)',
        'builder.add_node("double", 두_배)',
        'builder.add_edge(START, "add")',
        'builder.add_edge("add", "double")',
        'builder.add_edge("double", END)',
        'graph = builder.compile()',
        '',
        'print(graph.invoke({"number": 3, "log": []}))',
        '# {\'number\': 8, \'log\': [\'+1\', \'x2\']}',
      ].join('\n')),

      quizBlock('t6/graph', [
        {
          type: 'choice',
          q: '노드가 상태의 일부만 돌려주면 나머지 칸은 어떻게 되나요?',
          options: ['모두 지워진다', '그대로 유지된다', '오류가 난다', '0 이 된다'],
          answer: 1,
          why: '바뀌는 칸만 돌려주면 됩니다. 나머지는 그대로 남아 코드가 짧아집니다.',
        },
        {
          type: 'choice',
          q: '대화 기록을 담는 칸에 이어붙이기 규칙을 안 붙이면?',
          options: [
            '오류가 난다',
            '새 메시지가 앞 대화를 덮어써서 기록이 사라진다',
            '메시지가 두 번씩 쌓인다',
            '아무 문제 없다',
          ],
          answer: 1,
          why: '기본이 덮어쓰기입니다. 쌓여야 하는 칸에는 반드시 규칙을 붙여야 합니다.',
        },
        {
          type: 'short',
          q: () => ['그래프에서 「모든 노드가 함께 보는 공유 데이터」를 무엇이라고 하나요? (두 글자)'],
          accept: ['상태', 'state'],
          why: '상태(State)입니다. 노드는 이 메모장을 읽고 고쳐 씁니다.',
        },
      ]),
      nextHint('갈림길과 루프로 →', () => ctx.go('agent', 'branch')),
    );
    run();
    return wrap;
  },
};

/* ══════════════════════════ 2. 갈림길과 루프 ══════════════════════════ */
const branchScreen = {
  id: 'branch',
  title: '갈림길과 루프',
  render(ctx) {
    const wrap = h('div');
    const bState = { n: 7 };
    const lState = { target: 5, limit: 25, broken: false };
    const bOut = h('div');
    const lOut = h('div');

    const runBranch = () => {
      const g = stateGraph({})
        .addNode('짝수 처리', () => ({ result: '짝수입니다' }))
        .addNode('홀수 처리', () => ({ result: '홀수입니다' }))
        .addNode('판단', (s) => ({ 판단중: s.number }))
        .addEdge(START, '판단')
        .addConditionalEdges('판단', (s) => (s.number % 2 === 0 ? '짝수' : '홀수'),
          { 짝수: '짝수 처리', 홀수: '홀수 처리' })
        .addEdge('짝수 처리', END)
        .addEdge('홀수 처리', END)
        .compile();
      const r = g.invoke({ number: bState.n, result: '' });
      bOut.textContent = '';
      bOut.append(
        h('div.flow',
          h('span.flow-step', '시작'), h('span.flow-arrow', '→'),
          h('span.flow-step.done', '판단'), h('span.flow-arrow', '→'),
          h('span.flow-step' + (bState.n % 2 === 0 ? '.on' : ''), '짝수 처리'),
          h('span', { style: { color: 'var(--dim)' } }, ' 또는 '),
          h('span.flow-step' + (bState.n % 2 !== 0 ? '.on' : ''), '홀수 처리'),
        ),
        traceTable(r.trace, ['number', 'result']),
      );
    };

    const runLoop = () => {
      const g = stateGraph({})
        .addNode('하나씩 세기', (s) => ({ count: s.count + 1 }))
        .addEdge(START, '하나씩 세기')
        .addConditionalEdges('하나씩 세기',
          (s) => (lState.broken ? '계속' : (s.count < lState.target ? '계속' : '끝')),
          { 계속: '하나씩 세기', 끝: END })
        .compile({ recursionLimit: lState.limit });
      const r = g.invoke({ count: 0 });
      lOut.textContent = '';
      lOut.append(
        h('div.stat-row',
          statBox('돈 횟수', String(r.hops), r.stopped ? 'bad' : 'good'),
          statBox('마지막 count', String(r.state.count)),
          statBox('멈춘 이유', r.stopped ? '안전장치' : '조건 충족', r.stopped ? 'bad' : 'good'),
        ),
        r.stopped ? note('bad', h('b', r.stopped), ' ',
          '멈추는 조건이 없거나 잘못되면 영원히 돕니다. 그래서 도구가 ',
          b('기본 25번'), ' 이라는 안전장치를 두고 있습니다. 이 장치가 없으면 프로그램이 멈춰 버립니다.')
          : note('good', `조건(count = ${lState.target})을 만나 정상적으로 멈췄습니다.`),
        traceTable(r.trace.slice(0, 14), ['count']),
        r.trace.length > 14 ? h('p', { style: { color: 'var(--dim)' } }, `…(모두 ${r.trace.length}걸음)`) : null,
      );
    };

    wrap.append(
      screenHead('갈림길과 루프', '상황에 따라 다른 길로 가고, 조건이 될 때까지 되풀이합니다.', '⑥ 스스로 판단하기'),

      card('갈림길 — 조건 분기',
        h('p', '핵심은 ', b('어디로 갈지 정해 주는 함수'), ' 입니다. 이 함수는 계산을 하지 않고 ',
          b('길 이름표'), ' 만 돌려줍니다.'),
        code([
          'def 어디로_갈까(상태):',
          '    return "짝수" if 상태["number"] % 2 == 0 else "홀수"',
          '',
          'builder.add_conditional_edges(',
          '    "판단",              # 어느 노드 다음에 갈리는가',
          '    어디로_갈까,          # 이름표를 정하는 함수',
          '    {"짝수": "짝수 처리", "홀수": "홀수 처리"},   # 이름표 → 실제 노드',
          ')',
        ].join('\n'), { copy: false }),
        slider({ label: '넣을 숫자', min: 0, max: 20, value: 7, onInput: (v) => { bState.n = v; runBranch(); } }),
        bOut,
      ),

      card('되돌아오기 — 루프',
        h('p', '갈림길에서 ', b('자기 자신'), ' 으로 돌아오는 화살표를 만들면 그것이 반복입니다.'),
        code([
          '        ┌──────(아직 멀었다)──────┐',
          '        ▼                        │',
          '   [하나씩 세기] ──(판단)─────────┘',
          '        │',
          '     (다 됐다)',
          '        ▼',
          '       끝',
        ].join('\n'), { copy: false }),
        slider({ label: '몇까지 셀까', min: 1, max: 12, value: 5, onInput: (v) => { lState.target = v; runLoop(); } }),
        slider({ label: '안전장치 (최대 몇 걸음까지 허용)', min: 3, max: 40, value: 25, onInput: (v) => { lState.limit = v; runLoop(); } }),
        toggle('일부러 멈추는 조건을 없애기 (무한 루프 만들기)', false, (v) => { lState.broken = v; runLoop(); }),
        lOut,
      ),

      card('왜 이것을 배우나',
        h('p', '다음 화면에서 만들 ', b('에이전트'), ' 가 정확히 이 모양이기 때문입니다.'),
        code([
          '        ┌────────────────────────────┐',
          '        ▼                            │  ← 반복 (도구 결과를 보고 다시 생각)',
          '  시작 → [생각] ──(도구가 필요한가?)──→ [도구 실행]',
          '           │  아니오                     ↑ 갈림길',
          '           ▼',
          '          끝 (최종 답)',
        ].join('\n'), { copy: false }),
        h('p', '오늘 배운 ', b('갈림길 + 되돌아오기'), ' 두 가지가 그대로 재료가 됩니다. ',
          '에이전트는 새로운 개념이 아니라 ', b('이 둘의 조합'), ' 입니다.'),
      ),

      deepDive('무한 루프를 막는 세 가지 방법',
        h('ol',
          h('li', b('멈추는 조건을 반드시 넣는다 — '),
            '가장 기본입니다. 「몇 번 돌았는가」를 상태에 세어 두고 한계를 넘으면 끝내는 것이 안전합니다.'),
          h('li', b('안전장치(재귀 한도)에 기댄다 — '),
            '도구가 기본 25번쯤에서 강제로 멈춰 줍니다. 다만 이건 ', b('사고를 막는 장치이지 설계가 아닙니다'), '. ',
            '여기 걸렸다면 조건이 잘못된 것입니다.'),
          h('li', b('진행하고 있는지 확인한다 — '),
            '루프를 돌 때마다 상태가 실제로 달라지는지 봅니다. ',
            '같은 상태로 계속 돌고 있다면 조건을 아무리 검사해도 빠져나올 수 없습니다.'),
        ),
        h('p', b('에이전트에서 특히 조심할 것 — '),
          '모델이 도구를 부르고, 결과가 마음에 안 들어 또 부르고, 또 부르고… 하는 일이 실제로 생깁니다. ',
          '「도구는 최대 세 번까지」 같은 한계를 상태에 넣어 두는 것이 실무의 기본입니다.'),
      ),

      pyBox([
        '# 갈림길',
        'def route(state):',
        '    return "짝수" if state["number"] % 2 == 0 else "홀수"',
        '',
        'builder.add_conditional_edges(START, route,',
        '    {"짝수": "even_node", "홀수": "odd_node"})',
        '',
        '# 루프 — 자기 자신으로 돌아오는 화살표',
        'def 계속할까(state):',
        '    return "continue" if state["count"] < 5 else "end"',
        '',
        'builder.add_conditional_edges("increment", 계속할까,',
        '    {"continue": "increment", "end": END})',
        '',
        'graph = builder.compile()   # 기본 재귀 한도 25',
      ].join('\n')),

      quizBlock('t6/branch', [
        {
          type: 'choice',
          q: '조건 분기 함수(라우터)가 돌려주어야 하는 것은?',
          options: [
            '새로운 상태',
            '어디로 갈지 나타내는 이름표(문자열)',
            '노드 함수',
            '참 또는 거짓',
          ],
          answer: 1,
          why: '이름표를 돌려주고, 그 이름표를 실제 노드에 잇는 지도를 함께 줍니다.',
        },
        {
          type: 'choice',
          q: '루프에서 멈추는 조건을 잘못 짜면?',
          options: [
            '그래프가 만들어지지 않는다',
            '재귀 한도에 걸려 강제로 멈춘다',
            '자동으로 고쳐진다',
            '노드가 사라진다',
          ],
          answer: 1,
          why: '안전장치가 프로그램이 완전히 멈추는 것은 막아 주지만, 여기 걸렸다면 조건이 잘못된 것입니다.',
        },
        {
          type: 'choice',
          q: '에이전트의 흐름을 이루는 두 가지 요소는?',
          options: [
            '검색과 생성',
            '갈림길(도구가 필요한가)과 되돌아오기(결과 보고 다시 생각)',
            '학습과 추론',
            '토큰화와 임베딩',
          ],
          answer: 1,
          why: '에이전트는 새 개념이 아니라 조건 분기와 반복의 조합입니다.',
        },
      ]),
      nextHint('도구를 쓰는 에이전트로 →', () => ctx.go('agent', 'agent')),
    );
    runBranch();
    runLoop();
    return wrap;
  },
};

/* ══════════════════════════ 3. 도구를 쓰는 에이전트 ═══════════════════ */

/* 에이전트가 쓸 도구들 — 설명(무엇을 하는 도구인지)을 함께 적어 둔다 */
const ORDER_DB = { 'A-1204': '배송 중 (오늘 도착 예정)', 'B-3311': '배송 완료 (어제 도착)', 'C-0077': '준비 중' };
let RAGIDX = null;
const TOOLS = [
  {
    name: '계산기',
    desc: '두 숫자를 더하거나 곱합니다.',
    match: (t) => /(\d+)\s*(더하기|\+|곱하기|×|\*)\s*(\d+)/.test(t),
    run: (t) => {
      const m = t.match(/(\d+)\s*(더하기|\+|곱하기|×|\*)\s*(\d+)/);
      const [a, op, c] = [Number(m[1]), m[2], Number(m[3])];
      const v = /더하기|\+/.test(op) ? a + c : a * c;
      return `${a} ${op} ${c} = ${v}`;
    },
  },
  {
    name: '주문조회',
    desc: '주문번호로 배송 상태를 조회합니다.',
    // 없는 번호도 일단 도구가 받아서 「찾을 수 없다」고 정직하게 돌려주게 한다
    match: (t) => /[A-Z]-\d{4}/.test(t),
    run: (t) => {
      const id = t.match(/[A-Z]-\d{4}/)[0];
      return ORDER_DB[id] ? `${id} : ${ORDER_DB[id]}` : `${id} 라는 주문번호를 찾을 수 없습니다.`;
    },
  },
  {
    name: '안내문검색',
    desc: '도서관 안내문에서 관련 내용을 찾습니다. (탭 ⑤의 RAG 를 도구로 붙인 것)',
    match: (t) => /대출|반납|연체|열람실|자리|전자책|회원|도서관|빌리|빌릴|빌려|책|며칠|몇 권|예약|이용 시간|규정|정책/.test(t),
    run: (t) => {
      if (!RAGIDX) RAGIDX = buildIndex(splitByHeadings(LIBRARY_DOC));
      const r = retrieve(RAGIDX, t, { k: 2, mode: 'hybrid' });
      const a = answerFromContext(t, r.hits, { strict: true, short: true });
      return a.text;
    },
  },
];

const agentScreen = {
  id: 'agent',
  title: '도구를 쓰는 에이전트',
  render(ctx) {
    const wrap = h('div');
    const state = { q: 'A-1204 주문 어디까지 왔어?', useTools: true };
    const out = h('div');

    const run = () => {
      const g = stateGraph({ messages: addMessages })
        /* 「생각」 노드 — 진짜 모델 자리. 여기서는 규칙으로 판단한다. */
        .addNode('생각', (s) => {
          const last = s.messages[s.messages.length - 1];
          if (last.role === 'tool') {
            return { messages: [{ role: 'ai', text: `도구 결과를 확인했습니다. ${last.text}`, done: true }], done: true };
          }
          if (!state.useTools) {
            return { messages: [{ role: 'ai', text: '(도구 없이) 잘 모르겠습니다. 정확한 정보를 드리기 어렵습니다.', done: true }], done: true };
          }
          const tool = TOOLS.find((t) => t.match(last.text));
          if (tool) return { messages: [{ role: 'ai', text: `${tool.name} 도구를 써야겠습니다.`, tool: tool.name, arg: last.text }], tool: tool.name };
          return { messages: [{ role: 'ai', text: '도구 없이 바로 답할 수 있는 질문입니다. 무엇을 도와드릴까요?', done: true }], done: true };
        })
        /* 「도구」 노드 — 실제로 함수를 실행한다 */
        .addNode('도구 실행', (s) => {
          const call = [...s.messages].reverse().find((m) => m.tool);
          const tool = TOOLS.find((t) => t.name === call.tool);
          return { messages: [{ role: 'tool', text: tool.run(call.arg) }], tool: '' };
        })
        .addEdge(START, '생각')
        .addConditionalEdges('생각', (s) => (s.tool ? '도구 필요' : '끝'),
          { '도구 필요': '도구 실행', 끝: END })
        .addEdge('도구 실행', '생각')
        .compile({ recursionLimit: 8 });

      const r = g.invoke({ messages: [{ role: 'human', text: state.q }], tool: '', done: false });
      out.textContent = '';
      out.append(
        h('div.flow',
          h('span.flow-step', '시작'), h('span.flow-arrow', '→'),
          h('span.flow-step.done', '생각'), h('span.flow-arrow', '⇄'),
          h('span.flow-step' + (r.hops > 2 ? '.done' : ''), '도구 실행'), h('span.flow-arrow', '→'),
          h('span.flow-step', '끝'),
        ),
        h('div', { style: { border: '1px solid var(--line)', borderRadius: '10px', padding: '10px', background: '#0e1728' } },
          ...r.state.messages.map((m) => h(`div.msg.msg-${m.role === 'tool' ? 'system' : m.role}`,
            h('span.msg-role', m.role === 'human' ? '사용자' : m.role === 'tool' ? '도구' : '모델'),
            h('div.msg-text', m.text))),
        ),
        h('div.stat-row',
          statBox('돈 걸음', String(r.hops)),
          statBox('도구를 썼나', r.state.messages.some((m) => m.role === 'tool') ? '썼음' : '안 씀',
            r.state.messages.some((m) => m.role === 'tool') ? 'good' : ''),
        ),
      );
    };

    wrap.append(
      screenHead('도구를 쓰는 에이전트', '모델이 스스로 「이건 도구를 써야겠다」고 판단하게 만듭니다.', '⑥ 스스로 판단하기'),

      card('에이전트란',
        h('p', '모델은 계산이 약하고, 오늘 날짜를 모르고, 우리 회사 자료도 모릅니다. ',
          '그래서 ', b('필요할 때 도구를 부르게'), ' 합니다.'),
        code([
          '생각 → (도구가 필요한가?) → 도구 실행 → 결과 보고 다시 생각 → … → 답',
          '        ↑ 갈림길                          ↑ 되돌아오기',
          '        (앞 화면에서 배운 그것)',
        ].join('\n'), { copy: false }),
        note('warn', h('b', '오해하기 쉬운 곳 — 모델이 함수를 직접 실행하는 것이 아닙니다. '),
          '모델은 「이 도구를 이 값으로 써 주세요」라고 ', b('요청만'), ' 합니다. ',
          '실제 실행은 우리 프로그램이 합니다. 그래서 위험한 도구를 막는 것도 우리 몫입니다.'),
      ),

      card('이 에이전트가 가진 도구',
        table(['도구 이름', '무엇을 하나', '언제 쓰나'],
          TOOLS.map((t) => [h('b', t.name), t.desc,
            h('span', { style: { color: 'var(--dim)', fontSize: '.86rem' } },
              t.name === '계산기' ? '「3 더하기 5」 같은 계산이 보일 때'
                : t.name === '주문조회' ? '「A-1204」 같은 주문번호가 보일 때'
                  : '대출·반납·열람실 같은 안내문 관련 낱말이 보일 때')])),
        note('info', h('b', '도구 설명(docstring)이 아주 중요합니다. '),
          '진짜 에이전트에서는 모델이 이 설명만 읽고 「언제 이 도구를 쓸지」를 판단합니다. ',
          '설명이 모호하면 엉뚱한 도구를 부릅니다.'),
      ),

      card('돌려 보기',
        h('div.pills', ...[
          'A-1204 주문 어디까지 왔어?',
          '책은 며칠까지 빌릴 수 있나요?',
          '17 더하기 25 는?',
          '오늘 기분이 어때?',
          'Z-9999 주문 확인해 줘',
        ].map((t) => h('button.pill', {
          type: 'button',
          onclick: () => { state.q = t; wrap.querySelector('input.inp').value = t; run(); },
        }, t))),
        input({ value: state.q, onInput: (v) => { state.q = v; run(); } }),
        toggle('도구를 쥐여 주기', true, (v) => { state.useTools = v; run(); }),
        out,
        h('div.honest',
          h('b', '「생각」 노드는 진짜 모델이 아닙니다. '),
          '어떤 도구를 쓸지 규칙으로 정합니다(숫자가 보이면 계산기, 주문번호 모양이 보이면 주문조회…). ',
          '진짜 에이전트에서는 모델이 도구 설명을 읽고 스스로 고릅니다. ',
          b('그래프의 흐름 — 갈림길·되돌아오기·상태 누적 — 은 모두 진짜입니다.'),
          ' 「안내문검색」 도구는 탭 ⑤의 RAG 를 그대로 붙인 것이라 실제로 검색이 일어납니다.'),
      ),

      card('RAG 를 도구로 붙이면',
        h('p', '위에서 ', mono('책은 며칠까지 빌릴 수 있나요?'), ' 를 눌러 보세요. ',
          '에이전트가 「안내문검색」 도구를 부르고, 그 도구 안에서 ', b('탭 ⑤의 RAG 파이프라인'),
          ' 이 그대로 돌아갑니다.'),
        note('good', h('b', 'RAG(탭 ⑤) → 에이전트(탭 ⑥) 로 자연스럽게 이어집니다. '),
          'RAG 는 「언제나 검색한다」이고, 에이전트는 「필요할 때만 검색한다」입니다. ',
          '「오늘 기분이 어때?」 같은 질문에까지 문서를 검색하는 것은 낭비니까요.'),
      ),

      deepDive('도구를 설계할 때 조심할 것',
        terms([
          ['설명을 정확히', '모델은 설명만 보고 고릅니다. 「검색합니다」보다 「도서관 이용 안내문에서 대출·반납 규정을 찾습니다」가 훨씬 낫습니다.'],
          ['도구는 적게', '스무 개를 쥐여 주면 헷갈립니다. 역할별로 나누는 편이 낫습니다(다음 화면).'],
          ['실패를 알려 주기', '「찾을 수 없습니다」를 도구가 정직하게 돌려주어야 모델이 다시 물어볼 수 있습니다. 위에서 Z-9999 를 눌러 보세요.'],
          ['위험한 도구는 막기', '삭제·결제 같은 도구는 사람 확인을 거치게 합니다. 모델의 판단만 믿고 실행하면 안 됩니다.'],
          ['횟수 제한', '도구를 계속 부르며 맴도는 일이 실제로 생깁니다. 「최대 세 번」 같은 한계를 상태에 넣습니다.'],
        ]),
      ),

      pyBox([
        'from langchain_core.tools import tool',
        'from langgraph.prebuilt import ToolNode, tools_condition',
        'from langgraph.graph.message import add_messages',
        '',
        '@tool',
        'def 주문조회(주문번호: str) -> str:',
        '    """주문번호로 배송 상태를 조회합니다."""   # ← 모델이 읽는 설명',
        '    db = {"A-1204": "배송 중", "B-3311": "배송 완료"}',
        '    return db.get(주문번호, "해당 주문번호를 찾을 수 없습니다.")',
        '',
        'class 상태(TypedDict):',
        '    messages: Annotated[list, add_messages]   # 대화는 이어붙이기',
        '',
        'llm_with_tools = llm.bind_tools([주문조회])',
        '',
        'def 생각(state):',
        '    return {"messages": [llm_with_tools.invoke(state["messages"])]}',
        '',
        'builder.add_node("chatbot", 생각)',
        'builder.add_node("tools", ToolNode([주문조회]))',
        'builder.add_edge(START, "chatbot")',
        'builder.add_conditional_edges("chatbot", tools_condition)   # 갈림길',
        'builder.add_edge("tools", "chatbot")                        # 되돌아오기',
        'agent = builder.compile()',
      ].join('\n')),

      quizBlock('t6/agent', [
        {
          type: 'choice',
          q: '모델이 도구를 「쓴다」는 것은 정확히 무슨 뜻인가요?',
          options: [
            '모델이 함수를 직접 실행한다',
            '모델은 「이 도구를 이 값으로 써 달라」고 요청하고, 실행은 프로그램이 한다',
            '도구가 모델을 부른다',
            '도구가 모델 안에 들어 있다',
          ],
          answer: 1,
          why: '이 구분이 중요합니다. 실행은 우리 코드가 하므로, 위험한 도구를 막는 책임도 우리에게 있습니다.',
        },
        {
          type: 'choice',
          q: '도구의 설명(docstring)이 중요한 까닭은?',
          options: [
            '사람이 읽으려고',
            '모델이 그 설명을 보고 언제 쓸지 판단해서',
            '실행 속도를 높여서',
            '문법상 필수라서',
          ],
          answer: 1,
          why: '설명이 모호하면 엉뚱한 도구를 부릅니다. 도구 설명은 사실상 프롬프트의 일부입니다.',
        },
        {
          type: 'choice',
          q: 'RAG 와 「RAG 를 도구로 가진 에이전트」의 차이는?',
          options: [
            'RAG 는 언제나 검색하고, 에이전트는 필요할 때만 검색한다',
            'RAG 가 더 정확하다',
            '에이전트는 문서를 쓰지 않는다',
            '차이가 없다',
          ],
          answer: 0,
          why: '「오늘 기분 어때?」에까지 문서를 검색하는 것은 낭비입니다. 스스로 판단하게 하는 것이 에이전트입니다.',
        },
      ]),
      nextHint('멀티턴과 멀티에이전트로 →', () => ctx.go('agent', 'team')),
    );
    run();
    return wrap;
  },
};

/* ══════════════════════════ 4. 멀티턴과 멀티에이전트 ══════════════════ */
const teamScreen = {
  id: 'team',
  title: '멀티턴과 멀티에이전트',
  render(ctx) {
    const wrap = h('div');
    const saver = memorySaver();
    const state = { thread: '손님-A', memory: true };
    const out = h('div');
    const logBox = h('div');

    const buildTeam = () => stateGraph({ messages: addMessages })
      /* 관리자 — 담당자만 고른다. 규정 쪽 낱말을 먼저 보고, 없으면 주문 쪽을 본다. */
      .addNode('관리자', (s) => {
        const last = [...s.messages].reverse().find((m) => m.role === 'human');
        const t = last ? last.text : '';
        let next = '일반 담당';
        if (/대출|반납|반품|연체|열람실|자리|전자책|회원증|빌리|빌릴|빌려|예약|규정|정책|며칠|몇 권|기간|이용 시간|잃어버/.test(t)) next = '정책 담당';
        else if (/[A-Z]-\d{4}|주문|배송|어디까지|조회/.test(t)) next = '주문 담당';
        return { next };
      })
      .addNode('주문 담당', (s) => {
        const all = s.messages.map((m) => m.text).join(' ');
        const id = (all.match(/[A-C]-\d{4}/) || [])[0];
        const text = id
          ? (ORDER_DB[id] ? `주문 ${id} 은(는) ${ORDER_DB[id]} 입니다.` : `${id} 라는 주문번호를 찾을 수 없습니다.`)
          : '주문번호를 알려 주시면 조회해 드리겠습니다.';
        return { messages: [{ role: 'ai', text: `[주문 담당] ${text}` }] };
      })
      .addNode('정책 담당', (s) => {
        const last = [...s.messages].reverse().find((m) => m.role === 'human');
        if (!RAGIDX) RAGIDX = buildIndex(splitByHeadings(LIBRARY_DOC));
        const r = retrieve(RAGIDX, last.text, { k: 2, mode: 'hybrid' });
        const a = answerFromContext(last.text, r.hits, { strict: true, short: true });
        // 앞 대화에 주문번호가 있고 「그·그거」로 가리켰다면, 맥락을 이어받았음을 보여 준다
        const all = s.messages.map((m) => m.text).join(' ');
        const id = (all.match(/[A-Z]-\d{4}/) || [])[0];
        const carry = id && /그|그거|저거|이거/.test(last.text) ? `(앞서 말씀하신 주문 ${id} 관련) ` : '';
        return { messages: [{ role: 'ai', text: `[정책 담당] ${carry}${a.text}` }] };
      })
      .addNode('일반 담당', () => ({ messages: [{ role: 'ai', text: '[일반 담당] 안녕하세요! 무엇을 도와드릴까요?' }] }))
      .addEdge(START, '관리자')
      .addConditionalEdges('관리자', (s) => s.next,
        { '주문 담당': '주문 담당', '정책 담당': '정책 담당', '일반 담당': '일반 담당' })
      .addEdge('주문 담당', END)
      .addEdge('정책 담당', END)
      .addEdge('일반 담당', END)
      .compile({ checkpointer: state.memory ? saver : null });

    const send = (text) => {
      const g = buildTeam();
      const r = g.invoke({ messages: [{ role: 'human', text }], next: '' }, { threadId: state.thread });
      draw(r);
    };

    const draw = (r) => {
      const saved = saver.get(state.thread);
      logBox.textContent = '';
      const msgs = (state.memory ? saved?.messages : r?.state?.messages) || [];
      logBox.append(...msgs.map((m) => h(`div.msg.msg-${m.role}`,
        h('span.msg-role', m.role === 'human' ? '사용자' : '모델'),
        h('div.msg-text', m.text))));
      if (!msgs.length) logBox.appendChild(h('p', { style: { color: 'var(--dim)' } }, '아직 대화가 없습니다.'));

      out.textContent = '';
      if (r) {
        out.append(
          h('p', b('이번 질문이 지나간 길')),
          h('div.flow',
            ...r.trace.filter((t) => t.node !== '⟦시작⟧').map((t, i, arr) => [
              t.node === '⟨판단⟩'
                ? h('span.flow-step', { style: { borderColor: 'var(--accent2)' } }, `배정 → ${t.label}`)
                : h('span.flow-step' + (t.node === '⟦끝⟧' ? '' : '.done'), t.node),
              i < arr.length - 1 ? h('span.flow-arrow', '→') : null,
            ]).flat(),
          ),
          h('div.stat-row',
            statBox('대화방', state.thread),
            statBox('기억 장치', state.memory ? '켬' : '끔', state.memory ? 'good' : 'bad'),
            statBox('쌓인 메시지', String(msgs.length)),
            statBox('저장된 대화방', String(saver.threads().length)),
          ),
        );
      }
    };

    wrap.append(
      screenHead('멀티턴과 멀티에이전트', '이름은 비슷하지만 완전히 다른 개념입니다. 하나는 시간축, 하나는 역할축입니다.', '⑥ 스스로 판단하기'),

      card('먼저 확실히 구분하기',
        h('div.cols',
          h('div',
            h('p', b('멀티턴 = 한 담당자와 여러 번 대화'), ' ', h('span.badge', '시간축')),
            code([
              '🙋 → 🤖 → 🙋 → 🤖 → …',
              '',
              '핵심: 기억',
              '장치: 대화방 번호 + 저장 장치',
            ].join('\n'), { copy: false }),
          ),
          h('div',
            h('p', b('멀티에이전트 = 여러 전문가가 분담'), ' ', h('span.badge', '역할축')),
            code([
              '          ┌─ 🤖 주문 담당',
              '🙋 → 🧭 관리자 ─┼─ 🤖 정책 담당',
              '          └─ 🤖 일반 담당',
              '',
              '핵심: 협업',
              '장치: 관리자 + 조건 분기',
            ].join('\n'), { copy: false }),
          ),
        ),
        note('info', '이 둘은 ', b('함께 쓸 수 있습니다'), '. 아래 실습이 바로 「기억하는 상담팀」입니다.'),
      ),

      card('왜 전문가를 나누나',
        h('p', '한 에이전트에게 도구를 스무 개 쥐여 주면 어떻게 될까요? 무엇을 써야 할지 헷갈리고, ',
          '지시가 길어지고, 실수가 늘어납니다. 회사처럼 부서를 나누면 됩니다.'),
        table(['담당', '가진 도구', '지시'], [
          ['주문 담당', '주문 조회만', '짧고 분명하게'],
          ['정책 담당', '안내문 검색(RAG)만', '짧고 분명하게'],
          ['일반 담당', '없음', '인사와 잡담'],
          [h('b', '관리자'), '없음 (배정만)', '질문을 보고 담당자를 고른다'],
        ]),
        h('p', b('관리자의 배정은 앞 화면의 조건 분기 그대로입니다. '),
          '「짝수면 A, 홀수면 B」가 「주문이면 주문 담당, 정책이면 정책 담당」으로 바뀐 것뿐입니다.'),
      ),

      card('기억하는 상담팀 — 직접 돌려 보기',
        h('div.pills',
          pillGroup([{ label: '손님 A', value: '손님-A' }, { label: '손님 B', value: '손님-B' }],
            (v) => { state.thread = v; draw(null); }),
          toggle('기억 장치 켜기', true, (v) => { state.memory = v; draw(null); }),
          button('이 대화방 비우기', () => { saver.clear(); draw(null); }, 'ghost small'),
        ),
        h('div', { style: { display: 'flex', gap: '8px', margin: '10px 0' } },
          input({
            value: 'B-3311 주문 배송됐어요?',
            onEnter: (v) => { if (v.trim()) send(v.trim()); },
          }),
          button('보내기', () => {
            const i = wrap.querySelector('input.inp');
            if (i.value.trim()) send(i.value.trim());
          }),
        ),
        h('div.pills', ...[
          'B-3311 주문 배송됐어요?',
          '그럼 그 책은 며칠까지 빌릴 수 있나요?',
          '책은 몇 권까지 빌릴 수 있나요?',
          '안녕하세요',
        ].map((t) => h('button.pill', { type: 'button', onclick: () => send(t) }, t))),
        h('p', b('대화 기록')),
        logBox,
        out,
        note('good', h('b', '순서대로 눌러 보세요. '),
          '① ', mono('B-3311 주문 배송됐어요?'), ' → ', b('주문 담당'), ' 이 답합니다. ',
          '② ', mono('그럼 그 책은 며칠까지 빌릴 수 있나요?'), ' → 이번엔 ', b('정책 담당'), ' 이 답하는데, ',
          '앞에서 말한 주문번호를 ', b('그대로 이어받습니다'), '. ',
          '담당자가 바뀌어도 대화 기록은 팀 전체가 함께 봅니다. ',
          '기억 장치를 끄고 같은 순서로 해 보면 차이가 확 드러납니다.'),
      ),

      card('대화방 번호가 하는 일',
        h('p', '위에서 ', b('손님 A'), ' 로 대화한 뒤 ', b('손님 B'), ' 로 옮겨 보세요. B 는 A 의 대화를 모릅니다.'),
        code([
          '# 같은 그래프 하나로 여러 사람을 동시에 상대한다',
          'agent.invoke({...}, config={"configurable": {"thread_id": "손님-A"}})',
          'agent.invoke({...}, config={"configurable": {"thread_id": "손님-B"}})',
          '',
          '# 번호가 같으면 이어지고, 다르면 백지에서 시작한다',
          '# 실제 서비스에서는 로그인한 사용자 ID 를 넣는다',
        ].join('\n'), { copy: false }),
        note('info', h('b', '탭 ④의 「대화방 A / B」와 같은 개념입니다. '),
          '거기서는 대화 이력 저장소였고, 여기서는 그래프 상태 전체를 저장합니다. ',
          '그래서 중간에 멈췄다가 이어서 하는 것도 가능해집니다.'),
      ),

      deepDive('한 번에 안 끝나면 — 담당자끼리 넘기기',
        h('p', '지금 구조는 관리자가 한 명을 골라 보내면 끝입니다. ',
          '그런데 「주문을 확인한 뒤 그 주문의 반품 규정을 알려 줘」처럼 ',
          b('두 담당자가 이어서 일해야'), ' 하는 경우도 있습니다.'),
        code([
          '# 담당 노드를 END 가 아니라 관리자로 되돌린다',
          'team.add_edge("주문 담당", "supervisor")',
          'team.add_edge("정책 담당", "supervisor")',
          '',
          '# 관리자가 "이제 됐다" 를 돌려주면 끝낸다',
          'def route(state):',
          '    return state["next"]     # "주문"·"정책"·"FINISH" 중 하나',
          '',
          'team.add_conditional_edges("supervisor", route,',
          '    {"주문": "주문 담당", "정책": "정책 담당", "FINISH": END})',
        ].join('\n'), { copy: false }),
        note('warn', h('b', '이때 멈추는 조건을 반드시 넣어야 합니다. '),
          '관리자 → 담당자 → 관리자 → 담당자… 로 영원히 돌 수 있습니다. ',
          '앞 화면에서 본 루프의 교훈이 그대로 적용됩니다.'),
        h('p', b('다른 짜임새도 있습니다. '),
          '관리자 없이 담당자끼리 직접 넘기기, 여러 담당자가 동시에 일하고 결과를 합치기 등. ',
          '중요한 것은 ', b('어떤 짜임새든 결국 상태·노드·엣지·분기·반복의 조합'), ' 이라는 점입니다.'),
      ),

      pyBox([
        'from langgraph.checkpoint.memory import MemorySaver',
        'from langgraph.prebuilt import create_react_agent',
        '',
        '# ① 전문가 — 자기 도구만 가진 작은 에이전트',
        'order_agent  = create_react_agent(llm, [주문조회])',
        'policy_agent = create_react_agent(llm, [안내문검색])',
        '',
        '# ② 관리자 — 담당자를 고르기만 한다 (messages 는 건드리지 않는다)',
        'def supervisor(state):',
        '    system = ("너는 상담을 배정하는 관리자야. 담당자를 골라.\\n"',
        '              "- 주문/배송/조회 → order\\n"',
        '              "- 대출/반납/연체/정책 → policy\\n"',
        '              "- 그 외 인사/잡담 → general\\n"',
        '              "반드시 order, policy, general 중 한 단어로만 답해.")',
        '    decision = llm.invoke([{"role":"system","content":system}] + state["messages"])',
        '    return {"next": decision.content.strip().lower()}',
        '',
        '# ③ 배정 = 조건 분기',
        'team.add_conditional_edges("supervisor", lambda s: s["next"],',
        '    {"order": "order", "policy": "policy", "general": "general"})',
        '',
        '# ④ 기억 장치만 붙이면 「기억하는 상담팀」이 된다',
        'team_mem = team.compile(checkpointer=MemorySaver())',
        'team_mem.invoke({"messages": [...]},',
        '                {"configurable": {"thread_id": "손님-A"}})',
      ].join('\n')),

      think('전문가를 몇 명까지 나누는 것이 좋을까요? ',
        '너무 잘게 나누면 어떤 문제가 생길지, 회사의 부서 나누기와 비교해 생각해 보세요.'),

      quizBlock('t6/team', [
        {
          type: 'choice',
          q: '멀티턴과 멀티에이전트의 차이를 바르게 말한 것은?',
          options: [
            '멀티턴은 시간축(기억), 멀티에이전트는 역할축(분담)',
            '멀티턴은 역할축, 멀티에이전트는 시간축',
            '둘은 같은 말이다',
            '멀티에이전트는 대화가 한 번뿐이다',
          ],
          answer: 0,
          why: '멀티턴은 「한 담당자와 여러 번」, 멀티에이전트는 「여러 담당자가 나눠서」입니다. 함께 쓸 수도 있습니다.',
        },
        {
          type: 'choice',
          q: '이미 만든 상담팀 그래프에 기억을 붙이려면?',
          options: [
            '그래프를 처음부터 다시 만든다',
            '컴파일할 때 저장 장치(checkpointer)만 붙인다',
            '노드를 하나 더 만든다',
            '온도를 0으로 둔다',
          ],
          answer: 1,
          why: '그래프는 그대로 두고 저장 장치만 붙이면 됩니다. 이것이 상태를 분리해 둔 덕입니다.',
        },
        {
          type: 'choice',
          q: '관리자(Supervisor)의 배정은 앞에서 배운 무엇과 같은 것인가요?',
          options: ['반복(루프)', '조건 분기', '이어붙이기 규칙', '재귀 한도'],
          answer: 1,
          why: '「짝수면 A, 홀수면 B」가 「주문이면 주문 담당, 정책이면 정책 담당」으로 바뀐 것뿐입니다.',
        },
        {
          type: 'choice',
          q: '담당 노드를 END 대신 관리자로 되돌리면 무엇을 반드시 넣어야 하나요?',
          options: ['도구를 더 넣는다', '멈추는 조건(FINISH)', '온도 조절', '새 대화방 번호'],
          answer: 1,
          why: '관리자 → 담당자 → 관리자 … 로 영원히 돌 수 있습니다. 루프에는 언제나 멈추는 조건이 필요합니다.',
        },
      ]),
      nextHint('⑦ 파이썬 실습실로 →', () => ctx.go('python', 'run')),
    );
    draw(null);
    return wrap;
  },
};

export default {
  id: 'agent',
  num: 'Ⅵ',
  title: '스스로 판단하기',
  screens: [graphScreen, branchScreen, agentScreen, teamScreen],
};
