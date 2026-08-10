/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * graph.js — 「상태·노드·엣지」로 흐름을 만드는 아주 작은 그래프 엔진
 *
 * 직선으로만 흐르던 파이프라인(입력 → 검색 → 모델 → 출력)에
 * 갈림길과 되돌아오기를 넣으려면 순서도가 필요하다. 그 순서도를 코드로 만든다.
 *
 * 여기 있는 개념은 실제 도구(LangGraph)의 것과 이름·구조를 맞춰 두었다.
 *   State    모두가 함께 보는 메모장 (TypedDict 자리)
 *   Node     상태를 받아 「바뀔 부분만」 돌려주는 함수
 *   Edge     노드에서 노드로 가는 화살표
 *   Reducer  같은 칸에 새 값이 오면 덮어쓸지 이어붙일지 정하는 규칙
 *   재귀 한도 되돌아오기가 멈추지 않을 때를 대비한 안전장치
 *
 * 실행할 때마다 한 걸음씩의 기록(trace)을 남겨, 화면에서 상태가 어떻게
 * 바뀌는지 표로 따라갈 수 있게 했다.
 * ========================================================================== */

export const START = '⟦시작⟧';
export const END = '⟦끝⟧';

/** 이어붙이는 규칙 — 리스트끼리 더한다 (파이썬의 operator.add 자리) */
export const addReducer = (prev, next) => [...(prev || []), ...(Array.isArray(next) ? next : [next])];

/** 대화 메시지 전용 규칙 — 같은 id 의 메시지는 갈아 끼우고 나머지는 이어붙인다 */
export const addMessages = (prev, next) => {
  const out = [...(prev || [])];
  for (const m of Array.isArray(next) ? next : [next]) {
    const i = m.id !== undefined ? out.findIndex((x) => x.id === m.id) : -1;
    if (i >= 0) out[i] = m;
    else out.push(m);
  }
  return out;
};

export function stateGraph(schema = {}) {
  const nodes = new Map();
  const edges = new Map();        // from → to
  const branches = new Map();     // from → { router, map }

  const api = {
    addNode(name, fn) { nodes.set(name, fn); return api; },
    addEdge(from, to) {
      if (!edges.has(from)) edges.set(from, []);
      edges.get(from).push(to);
      return api;
    },
    addConditionalEdges(from, router, map) { branches.set(from, { router, map }); return api; },

    /** 화면에 그릴 수 있도록 구조를 꺼내 준다 */
    describe() {
      const list = [];
      for (const [from, tos] of edges) tos.forEach((to) => list.push({ from, to, kind: 'plain' }));
      for (const [from, b] of branches) {
        Object.entries(b.map).forEach(([label, to]) => list.push({ from, to, kind: 'cond', label }));
      }
      return { nodes: [...nodes.keys()], edges: list };
    },

    compile({ checkpointer = null, recursionLimit = 25 } = {}) {
      return {
        nodes: [...nodes.keys()],
        describe: api.describe,
        /**
         * @param {object} input 시작 상태
         * @param {object} config { threadId } — 같은 번호면 앞선 대화를 이어 받는다
         */
        invoke(input, config = {}) {
          const thread = config.threadId || 'default';
          const saved = checkpointer ? checkpointer.get(thread) : null;
          let state = merge(schema, saved ? { ...saved } : {}, input);
          const trace = [{ node: START, state: clone(state), note: saved ? `기억 장치에서 ${thread} 대화를 이어받음` : '' }];

          let cur = firstFrom(START);
          let hops = 0;
          let stopped = '';
          while (cur && cur !== END) {
            if (hops >= recursionLimit) { stopped = `재귀 한도 ${recursionLimit}번에 걸려 멈췄습니다.`; break; }
            hops += 1;
            const fn = nodes.get(cur);
            if (!fn) { stopped = `'${cur}' 라는 노드가 없습니다.`; break; }
            const patch = fn(state) || {};
            state = merge(schema, state, patch);
            trace.push({ node: cur, patch: clone(patch), state: clone(state) });
            cur = nextFrom(cur, state, trace);
          }
          if (checkpointer && !stopped) checkpointer.set(thread, clone(state));
          trace.push({ node: END, state: clone(state), note: stopped });
          return { state, trace, hops, stopped };
        },
      };

      function firstFrom(from) {
        const tos = edges.get(from);
        if (tos && tos.length) return tos[0];
        const b = branches.get(from);
        return b ? null : END;
      }
      function nextFrom(from, state, trace) {
        const b = branches.get(from);
        if (b) {
          const label = b.router(state);
          const to = b.map[label];
          trace.push({ node: '⟨판단⟩', router: from, label, to: to || END, state: clone(state) });
          return to || END;
        }
        const tos = edges.get(from);
        return tos && tos.length ? tos[0] : END;
      }
    },
  };

  // START 에서 조건 분기로 바로 갈라지는 경우도 지원한다
  const origCompile = api.compile;
  api.compile = (o) => {
    const g = origCompile(o);
    const b = branches.get(START);
    if (!b) return g;
    const inner = g.invoke;
    g.invoke = (input, config = {}) => inner(input, config);
    return g;
  };
  return api;
}

function clone(o) {
  return JSON.parse(JSON.stringify(o, (k, v) => (typeof v === 'function' ? '(함수)' : v)));
}

/** 상태 합치기 — 칸마다 규칙(reducer)이 있으면 그 규칙을, 없으면 덮어쓴다 */
function merge(schema, base, patch) {
  const out = { ...base };
  for (const [k, v] of Object.entries(patch || {})) {
    const reducer = schema[k];
    out[k] = typeof reducer === 'function' ? reducer(out[k], v) : v;
  }
  return out;
}

/** 대화를 대화방 번호별로 저장하는 장치 (MemorySaver 자리) */
export function memorySaver() {
  const store = new Map();
  return {
    get: (thread) => store.get(thread) || null,
    set: (thread, state) => store.set(thread, state),
    threads: () => [...store.keys()],
    clear: () => store.clear(),
    dump: () => Object.fromEntries(store),
  };
}

/* ═══════════════════════ 그래프 구조를 글자 그림으로 ═══════════════════ */
export function toMermaid(desc) {
  const lines = ['graph TD'];
  const safe = (s) => s.replace(/[⟦⟧⟨⟩]/g, '').replace(/\s+/g, '_') || 'n';
  lines.push(`  ${safe(START)}([시작])`);
  lines.push(`  ${safe(END)}([끝])`);
  desc.nodes.forEach((n) => lines.push(`  ${safe(n)}[${n}]`));
  desc.edges.forEach((e) => {
    const arrow = e.kind === 'cond' ? `-. ${e.label} .->` : '-->';
    lines.push(`  ${safe(e.from)} ${arrow} ${safe(e.to)}`);
  });
  return lines.join('\n');
}
