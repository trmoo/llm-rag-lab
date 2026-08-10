/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ⑦ 파이썬 실습실
 *   1. 코드 실행       설치 없이 브라우저에서 바로 돌린다
 *   2. 빈칸 채우기     실행 결과까지 자동 채점
 *   3. 종합 정리       전 범위 마무리 문제와 나의 기록
 * ========================================================================== */

import {
  h, card, note, deepDive, terms, table, screenHead, button, textarea, input,
  pillGroup, code, mono, b, nextHint, fx, pct, onResize,
} from '../lib/ui.js';
import { quizBlock, normAnswer } from '../lib/quiz.js';
import { runPython } from '../lib/pymini.js';
import { EXAMPLES, FILL_PROBLEMS } from '../data/snippets.js';
import { summary, getScratch, setScratch, clearAll } from '../lib/store.js';

const statBox = (k, v, kind = '') => h('div.stat' + (kind ? '.' + kind : ''), h('div.k', k), h('div.v', v));

/** 실행 결과를 화면 조각으로 */
function outputBox(res) {
  const box = h('div', {
    style: {
      background: '#0a1120', border: '1px solid var(--line)', borderRadius: '10px',
      padding: '12px 14px', fontFamily: 'var(--mono)', fontSize: '.88rem',
      whiteSpace: 'pre-wrap', minHeight: '44px', maxHeight: '340px', overflowY: 'auto',
    },
  });
  const text = Array.isArray(res.output) ? res.output.join('\n') : (res.output || '');
  if (text) box.appendChild(h('span', text));
  if (!res.ok) {
    box.appendChild(h('div', {
      style: { color: 'var(--bad)', marginTop: text ? '10px' : '0', borderTop: text ? '1px solid var(--line)' : '', paddingTop: text ? '8px' : '0' },
    }, res.error));
  } else if (!text) {
    box.appendChild(h('span', { style: { color: 'var(--dim)' } }, '(출력이 없습니다. print( ) 를 써 보세요)'));
  }
  return box;
}

/* ══════════════════════════ 1. 코드 실행 ══════════════════════════════ */
const runScreen = {
  id: 'run',
  title: '코드 실행',
  render(ctx) {
    const wrap = h('div');
    const state = { current: EXAMPLES[0] };
    const editor = textarea({ value: getScratch('py/editor', EXAMPLES[0].code), rows: 16 });
    const stdinBox = textarea({ value: '', rows: 2, placeholder: 'input( ) 이 읽어 갈 값을 한 줄에 하나씩' });
    const outBox = h('div');
    const infoBox = h('div');

    const run = () => {
      const inputs = stdinBox.value.split('\n').filter((x) => x !== '');
      const t0 = performance.now();
      const res = runPython(editor.value, { inputs });
      const ms = Math.round(performance.now() - t0);
      setScratch('py/editor', editor.value);
      outBox.textContent = '';
      outBox.append(
        h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '6px' } },
          h('b', '실행 결과'),
          h('span', { style: { color: res.ok ? 'var(--good)' : 'var(--bad)', fontSize: '.85rem' } },
            res.ok ? `정상 종료 · ${ms}ms` : '오류'),
        ),
        outputBox(res),
      );
      if (state.current?.expect) {
        const got = (Array.isArray(res.output) ? res.output : String(res.output || '').split('\n')).map((s) => s.trimEnd());
        const want = state.current.expect;
        const ok = want.every((line, i) => (got[i] || '').trim() === line.trim());
        outBox.appendChild(ok
          ? note('good', h('b', '기대한 결과와 같습니다. '), '이 예제는 앱의 시험에서도 매번 확인합니다.')
          : note('warn', h('b', '기대한 결과와 다릅니다. '), '코드를 고쳤다면 정상입니다. 예제를 다시 불러오려면 위에서 다시 누르세요.'));
      }
    };

    const load = (ex) => {
      state.current = ex;
      editor.value = ex.code;
      stdinBox.value = '';
      infoBox.textContent = '';
      infoBox.append(
        h('p', h('b', ex.title), h('span', { style: { color: 'var(--dim)', marginLeft: '10px', fontSize: '.9rem' } }, ex.lead)),
        ex.note ? note('warn', ex.note) : null,
      );
      run();
    };

    const groups = [...new Set(EXAMPLES.map((e) => e.group))];

    wrap.append(
      screenHead('코드 실행', '설치도 인터넷도 필요 없습니다. 여기서 바로 돌아갑니다.', '⑦ 파이썬 실습실'),

      card('예제 골라 오기',
        ...groups.map((g) => h('div', { style: { margin: '6px 0' } },
          h('span', { style: { color: 'var(--dim)', fontSize: '.86rem', marginRight: '8px' } }, g),
          h('div.pills', { style: { display: 'inline-flex' } },
            ...EXAMPLES.filter((e) => e.group === g).map((e) =>
              h('button.pill', { type: 'button', onclick: () => load(e) }, e.title))),
        )),
        infoBox,
      ),

      card('코드',
        editor,
        h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap', margin: '10px 0' } },
          button('▶ 실행 (Ctrl+Enter)', run),
          button('예제 다시 불러오기', () => load(state.current), 'ghost'),
          button('싹 지우기', () => { editor.value = ''; outBox.textContent = ''; }, 'ghost'),
        ),
        h('details', { style: { margin: '8px 0' } },
          h('summary', { style: { cursor: 'pointer', color: 'var(--dim)', fontSize: '.9rem' } },
            'input( ) 이 읽어 갈 값 미리 넣어 두기'),
          stdinBox,
        ),
        outBox,
      ),

      card('여기서 쓸 수 있는 것',
        h('div.honest',
          h('b', '이것은 교실용으로 직접 만든 아주 작은 파이썬입니다. '),
          '진짜 파이썬이 아니고, 브라우저에서 돌리려고 수업에 나오는 문법만 골라 만들었습니다. ',
          '그래서 ', b('실제 라이브러리(사이킷런·젠심·파이토치·랭체인 등)는 쓸 수 없습니다'), '. ',
          '대신 그 자리에서 ', b('같은 계산을 직접 만들어 봅니다'), ' — 원래 수업에서도 TF-IDF 는 손으로 한 번 만든 뒤 도구를 씁니다.'),
        table(['쓸 수 있는 것', '보기'], [
          ['변수·산술·비교·논리', mono('x = 3 + 4 * 2'), ],
          ['문자열과 f-문자열', mono('f"{이름} 님, {점수}점"')],
          ['리스트·딕셔너리·튜플', mono("리스트[0], 사전['열쇠'], (a, b)")],
          ['조건문·반복문', mono('if / elif / else, for, while')],
          ['리스트 컴프리헨션', mono('[w for w in 낱말들 if len(w) >= 2]')],
          ['함수와 클래스', mono('def, class, __init__, self')],
          ['내장 함수', mono('print len str int float round sum max min')],
          ['', mono('range list dict tuple set sorted enumerate zip any all reversed')],
          ['문자열 메서드', mono('split join strip lower upper replace count startswith')],
          ['리스트 메서드', mono('append pop insert remove index count sort reverse')],
          ['딕셔너리 메서드', mono('keys values items get pop most_common')],
          [h('b', 'import re'), mono('re.sub  re.findall  re.split  re.search  re.match')],
          [h('b', 'import math'), mono('math.log  math.sqrt  math.floor  math.pi')],
          [h('b', 'import random'), mono('random.randint  random.choice  random.shuffle')],
          [h('b', 'from collections import Counter'), mono('Counter(리스트).most_common(3)')],
          [h('b', 'import korean'), mono('korean.nouns  korean.morphs  korean.pos  korean.josa')],
        ], { compact: true }),
        note('warn', h('b', '지원하지 않는 것 — '),
          mono('lambda'), ', ', mono('with'), ', ', mono('try/except'), ', 제너레이터, 파일 읽기·쓰기, ',
          '그리고 ', mono('set( )'), ' 은 진짜 집합이 아니라 ', b('중복을 없앤 리스트'), ' 를 돌려줍니다 ',
          '(어휘 만들기에 쓰는 자리에서는 결과가 같습니다).'),
      ),

      card('오류가 났을 때',
        h('p', '빨간 글씨가 나와도 겁먹지 마세요. ', b('어디가 잘못됐는지 알려 주는 안내판'), ' 입니다. ',
          '마지막 줄부터 읽는 습관을 들이면 해결이 빨라집니다.'),
        table(['자주 보는 오류', '무슨 뜻인가', '어떻게 고치나'], [
          ['NameError', '모르는 이름을 썼다', '오타를 확인하거나, 그 변수를 먼저 만들었는지 본다'],
          ['SyntaxError', '문법이 어긋났다', '괄호·따옴표·콜론(:)이 빠지지 않았는지 본다'],
          ['IndentationError', '들여쓰기가 어긋났다', '같은 블록은 같은 칸 수로 맞춘다 (보통 4칸)'],
          ['TypeError', '값의 종류가 맞지 않다', mono('"3" + 1'), ' 처럼 문자열과 숫자를 섞지 않았는지 본다'],
          ['IndexError', '없는 자리를 꺼냈다', '리스트 길이를 넘지 않았는지 본다 (자리 번호는 0부터)'],
          ['KeyError', '없는 열쇠를 꺼냈다', mono('사전.get(열쇠, 기본값)'), ' 을 써 보면 안전하다'],
          ['ZeroDivisionError', '0으로 나눴다', '나누기 전에 분모가 0인지 확인한다'],
        ], { compact: true }),
      ),

      deepDive('이 파이썬은 어떻게 만들어졌나',
        h('p', '글자를 그대로 실행하는 것이 아니라 세 걸음을 거칩니다. 진짜 파이썬도 큰 틀은 같습니다.'),
        h('ol',
          h('li', b('토큰화 — '), '소스를 낱말 단위로 쪼갭니다. 들여쓰기도 하나의 토큰으로 만듭니다.'),
          h('li', b('구문 분석 — '), '토큰을 문법 나무(AST)로 조립합니다. ', mono('3 + 4 * 2'), ' 가 ',
            '「3 더하기 (4 곱하기 2)」라는 나무가 되는 곳입니다.'),
          h('li', b('실행 — '), '나무를 위에서부터 하나씩 밟아 갑니다.'),
        ),
        h('p', b('왜 직접 만들었나 — '),
          '진짜 파이썬을 브라우저에 넣으려면 수 메가바이트를 내려받아야 합니다. ',
          '교실 인터넷이 끊기면 수업이 멈추죠. 그래서 수업에 나오는 문법만 골라 작게 만들었습니다.'),
      ),

      nextHint('빈칸 채우기로 →', () => ctx.go('python', 'fill')),
    );

    editor.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); }
    });
    load(EXAMPLES[0]);
    if (getScratch('py/editor')) { /* 저장된 코드가 있어도 예제를 우선 보여 준다 */ }
    return wrap;
  },
};

/* ══════════════════════════ 2. 빈칸 채우기 ════════════════════════════ */
const fillScreen = {
  id: 'fill',
  title: '빈칸 채우기',
  render(ctx) {
    const wrap = h('div');
    const scoreLine = h('div.quiz-score');
    const done = new Set();
    const correct = new Set();

    const updateScore = () => {
      scoreLine.textContent = `푼 문제 ${done.size} / ${FILL_PROBLEMS.length} · 맞힌 문제 ${correct.size}개`;
      scoreLine.classList.toggle('full', done.size === FILL_PROBLEMS.length);
    };

    wrap.append(
      screenHead('빈칸 채우기', '⬚ 자리에 들어갈 것을 채우고, 실제로 돌려서 확인합니다.', '⑦ 파이썬 실습실'),
      note('info', '답을 채운 뒤 ', b('[채점하고 실행]'), ' 을 누르면 ',
        '① 답이 맞는지 ② 코드가 실제로 잘 도는지 두 가지를 한 번에 확인합니다.'),
    );

    FILL_PROBLEMS.forEach((p) => {
      const inputs = p.blanks.map((bl) => input({ size: 12, placeholder: `⬚${bl.n}` }));
      const outBox = h('div');
      const feedback = h('div');

      const fill = () => {
        let src = p.code;
        p.blanks.forEach((bl, i) => {
          src = src.replace(new RegExp(`⬚${bl.n}`, 'g'), inputs[i].value || '⬚');
        });
        return src;
      };

      const grade = () => {
        const results = p.blanks.map((bl, i) => ({
          bl,
          got: inputs[i].value.trim(),
          ok: bl.answers.some((a) => normAnswer(a) === normAnswer(inputs[i].value)),
        }));
        const allOk = results.every((r) => r.ok);
        const empty = results.some((r) => !r.got);

        feedback.textContent = '';
        if (empty) {
          feedback.appendChild(note('warn', '빈칸을 모두 채워 주세요.'));
          return;
        }
        done.add(p.id);
        if (allOk) correct.add(p.id); else correct.delete(p.id);
        updateScore();

        const src = fill();
        const res = runPython(src);
        outBox.textContent = '';
        outBox.append(h('p', b('채운 코드로 실행한 결과')), outputBox(res));

        feedback.append(...results.map((r) => h('div', {
          style: { color: r.ok ? 'var(--good)' : 'var(--bad)', fontSize: '.92rem' },
        }, `⬚${r.bl.n} → ${r.got}  ${r.ok ? '⭕' : `❌  (정답: ${r.bl.answers[0]})`}`)));
        feedback.appendChild(h('div', { style: { color: 'var(--dim)', fontSize: '.9rem', marginTop: '4px' } },
          ...results.map((r) => h('div', `⬚${r.bl.n} — ${r.bl.why}`))));
        if (allOk && res.ok) {
          feedback.appendChild(note('good', h('b', '맞았고, 실제로도 잘 돕니다. ')));
        } else if (allOk && !res.ok) {
          feedback.appendChild(note('warn', '답은 맞았는데 실행에서 오류가 났습니다. 다른 곳을 건드리지 않았는지 확인해 보세요.'));
        }
      };

      wrap.appendChild(card(`${p.group} · ${p.title}`,
        h('p', { style: { color: 'var(--dim)' } }, '💡 ', p.hint),
        code(p.code, { copy: false }),
        h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', margin: '10px 0' } },
          ...p.blanks.map((bl, i) => h('label', { style: { display: 'flex', gap: '6px', alignItems: 'center' } },
            h('span', { style: { color: 'var(--accent)', fontWeight: '700' } }, `⬚${bl.n}`), inputs[i])),
          button('채점하고 실행', grade, 'small'),
        ),
        feedback,
        outBox,
      ));
    });

    wrap.appendChild(scoreLine);
    wrap.appendChild(nextHint('종합 정리로 →', () => ctx.go('python', 'wrap')));
    updateScore();
    return wrap;
  },
};

/* ══════════════════════════ 3. 종합 정리 ══════════════════════════════ */
const wrapScreen = {
  id: 'wrap',
  title: '종합 정리',
  render(ctx) {
    const el = h('div');
    const s = summary();

    el.append(
      screenHead('종합 정리', '여섯 걸음을 한 장으로 되짚고, 마지막 문제를 풉니다.', '⑦ 파이썬 실습실'),

      card('한 장으로 되짚기',
        code([
          '지저분한 글',
          '   │  ① 정제 · 정규화 · 토큰화 · 불용어              [탭 ①]',
          '   ▼',
          '깨끗한 낱말 목록',
          '   │  ② 단어 주머니 → TF → IDF → TF-IDF             [탭 ②]',
          '   ▼',
          '숫자 벡터  ──(코사인 유사도)──▶ 닮은 글 찾기',
          '   │  ③ 임베딩 — 낱말의 「뜻」을 좌표로              [탭 ②·⑤]',
          '   ▼',
          '뜻이 담긴 벡터',
          '   │  ④ 순서와 문맥 — RNN → LSTM → Attention        [탭 ③]',
          '   ▼',
          '문맥을 반영한 표현',
          '   │  ⑤ 다음 낱말의 확률 → 하나 고르기               [탭 ④]',
          '   ▼',
          '글을 만들어 내는 모델',
          '   │  ⑥ 찾아서 · 넣고 · 답한다 (RAG)                [탭 ⑤]',
          '   ▼',
          '내 문서로 답하는 시스템',
          '   │  ⑦ 갈림길 · 되돌아오기 · 도구 (에이전트)         [탭 ⑥]',
          '   ▼',
          '스스로 판단하는 시스템',
        ].join('\n'), { copy: false }),
      ),

      card('꼭 기억할 열 가지',
        terms([
          ['전처리는 「다 지우기」가 아니다', '무엇을 남길지 정하는 일이다. 목적에 따라 ㅠㅠ 도 소중한 신호가 된다.'],
          ['TF 는 길이를, IDF 는 흔함을 바로잡는다', '둘을 곱해야 「이 글에서만 특별한 낱말」이 드러난다.'],
          ['코사인 유사도는 방향만 본다', '그래서 글 길이가 달라도 내용이 같으면 가깝다고 판단한다.'],
          ['정확도만 보면 속는다', '95%가 다 긍정인 자료에서는 전부 긍정이라 답해도 95%다.'],
          ['임베딩의 핵심은 「가까움」', '숫자 하나하나의 뜻은 아무도 모른다. 가까우면 비슷하다는 것만 성립하면 된다.'],
          ['Attention 은 「어디를 볼까」를 계산한다', '순서 정보는 따로 더해 주어야 한다(위치 인코딩).'],
          ['모델에는 기억이 없다', '기억하는 것처럼 보이는 것은 앞 대화를 매번 다시 보내기 때문이다.'],
          ['RAG 는 재학습이 아니다', '문서만 바꾸면 몇 초 만에 최신 내용이 반영된다.'],
          ['RAG 의 답은 검색 품질이 정한다', '들어간 것이 엉뚱하면 나오는 것도 엉뚱하다.'],
          ['재지 않으면 개선인지 알 수 없다', '질문–정답 세트를 먼저 만들고, 전후를 비교한 뒤에만 채택한다.'],
        ]),
      ),

      quizBlock('t7/final', [
        {
          type: 'choice',
          q: '흔한 낱말의 값을 깎아 주는 것은 무엇인가요?',
          options: ['TF', 'IDF', '코사인 유사도', '토큰화'],
          answer: 1,
          why: 'IDF 는 등장 문서 수가 많을수록 값이 작아집니다. 모든 문서에 나오면 0 이 되기도 합니다.',
        },
        {
          type: 'choice',
          q: '「조용하지 않아서 좋았어요」를 낱말 세기 방식이 자주 틀리는 까닭은?',
          options: [
            '문장이 짧아서',
            '낱말의 위치를 버려서 「않다」가 어디에 걸리는지 모르기 때문',
            '한국어라서',
            '불용어가 많아서',
          ],
          answer: 1,
          why: '순서와 문맥을 읽으려면 RNN·LSTM·Attention 같은 구조가 필요합니다.',
        },
        {
          type: 'choice',
          q: 'Self-Attention 에서 softmax 를 거친 한 줄의 합은?',
          options: ['0', '1', '토큰 개수', '정해져 있지 않다'],
          answer: 1,
          why: '「어디를 얼마나 볼까」의 비율이므로 합이 1 입니다.',
        },
        {
          type: 'choice',
          q: '온도(temperature)를 높이면?',
          options: [
            '늘 같은 답만 한다',
            '뜻밖의 낱말이 뽑힐 확률이 커진다',
            '모델이 다시 학습된다',
            '토큰이 줄어든다',
          ],
          answer: 1,
          why: '분포가 평평해져 2등·3등도 뽑힙니다. 사실을 전달해야 하는 RAG 에서는 0 에 가깝게 둡니다.',
        },
        {
          type: 'choice',
          q: 'chunk_overlap 을 두는 까닭은?',
          options: [
            '조각 수를 늘리려고',
            '답이 조각 경계에 걸려 잘리는 것을 막으려고',
            '검색을 빠르게 하려고',
            '저장 공간을 아끼려고',
          ],
          answer: 1,
          why: '겹침이 있으면 경계에 걸친 내용이 어느 한 조각에는 온전히 담깁니다.',
        },
        {
          type: 'multi',
          q: 'RAG 를 쓰면 좋은 점을 모두 고르세요.',
          options: [
            '모델이 모르는 우리 내부 문서로 답할 수 있다',
            '문서만 바꾸면 최신 내용이 곧바로 반영된다',
            '어느 문서에서 나온 답인지 확인할 수 있다',
            '모델이 더 똑똑해진다',
          ],
          answer: [0, 1, 2],
          why: '모델 자체는 그대로입니다. 달라지는 것은 「무엇을 보고 답하는가」입니다.',
        },
        {
          type: 'choice',
          q: '그래프에서 대화 기록을 담는 칸에 이어붙이기 규칙을 안 붙이면?',
          options: [
            '오류가 난다',
            '새 메시지가 앞 대화를 덮어써서 기록이 사라진다',
            '메시지가 두 번씩 쌓인다',
            '문제가 없다',
          ],
          answer: 1,
          why: '기본이 덮어쓰기입니다. 쌓여야 하는 칸에는 반드시 규칙을 붙입니다.',
        },
        {
          type: 'choice',
          q: '멀티턴과 멀티에이전트를 바르게 구분한 것은?',
          options: [
            '멀티턴 = 시간축(기억), 멀티에이전트 = 역할축(분담)',
            '멀티턴 = 역할축, 멀티에이전트 = 시간축',
            '같은 말이다',
            '멀티에이전트가 더 정확하다',
          ],
          answer: 0,
          why: '한 담당자와 여러 번 대화하는 것과, 여러 담당자가 나눠 일하는 것의 차이입니다.',
        },
        {
          type: 'short',
          q: () => ['모델이 사실이 아닌 내용을 자신 있게 만들어 내는 현상은? (세 글자)'],
          accept: ['환각', 'hallucination', '할루시네이션'],
          why: '환각(Hallucination)입니다. RAG 와 프롬프트 규칙이 이것을 줄이는 주요 방법입니다.',
        },
        {
          type: 'short',
          q: () => ['훈련 자료가 아닌 시험 자료로 ', mono('fit()'), ' 을 해서 성능이 부풀려지는 잘못을 무엇이라 하나요? (네 글자)'],
          accept: ['데이터누수', '데이터 누수', '누수', 'data leakage'],
          why: '데이터 누수(Data Leakage)입니다. 성능이 갑자기 너무 좋게 나오면 가장 먼저 의심할 것입니다.',
        },
      ], { title: '마무리 문제' }),

      card('나의 기록',
        h('div.stat-row',
          statBox('열어 본 화면', `${s.visited}개`),
          statBox('푼 퀴즈 묶음', `${s.quizSets}개`),
          statBox('맞힌 문제', s.quizTotal ? `${s.quizOk} / ${s.quizTotal}` : '아직 없음',
            s.quizTotal && s.quizOk / s.quizTotal >= 0.8 ? 'good' : 'accent'),
        ),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '이 기록은 ', b('이 브라우저에만'), ' 남습니다. 이름·학번을 받지 않고 서버로 보내지도 않습니다. ',
          '공용 컴퓨터를 쓴다면 아래 단추로 지워 주세요.'),
        h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } },
          button('결과 요약 복사', async () => {
            const t = `[LLM·RAG 실습실] 열어 본 화면 ${s.visited}개 · 푼 퀴즈 묶음 ${s.quizSets}개 · 맞힌 문제 ${s.quizOk}/${s.quizTotal}`;
            try { await navigator.clipboard.writeText(t); alert('복사했습니다.\n\n' + t); } catch { alert(t); }
          }),
          button('내 기록 지우기', () => {
            if (confirm('이 컴퓨터에 저장된 기록을 모두 지웁니다. 계속할까요?')) { clearAll(); location.reload(); }
          }, 'danger'),
        ),
      ),

      card('더 해 볼 것',
        terms([
          ['진짜 모델 붙여 보기', '각 화면의 「🐍 파이썬으로는」 칸에 실제 코드가 적혀 있습니다. 그대로 옮겨 쓰면 진짜 모델로 돌아갑니다.'],
          ['내 문서로 RAG 만들기', '학교 규정, 동아리 안내문, 수행평가 안내처럼 짧고 규칙이 분명한 문서가 연습에 좋습니다.'],
          ['질문–정답 세트 만들기', '무엇을 만들든 이것부터 만드세요. 재지 않으면 좋아졌는지 알 수 없습니다.'],
          ['어디까지가 흉내였는지 확인', '탭 ①의 「배움 지도」에 이 앱에서 무엇이 진짜 계산이고 무엇이 흉내인지 표로 정리해 두었습니다.'],
        ]),
      ),
    );
    return el;
  },
};

export default {
  id: 'python',
  num: 'Ⅶ',
  title: '파이썬 실습실',
  screens: [runScreen, fillScreen, wrapScreen],
};
