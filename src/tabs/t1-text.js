/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * 탭 ① 글을 다루기 — 지저분한 글을 컴퓨터가 읽을 수 있게 다듬는다.
 *   1. 배움 지도       이 앱에서 무엇을 어디까지 하는지 (정직하게 밝히는 곳)
 *   2. 전처리 공방     정제 규칙을 하나씩 껐다 켜며 결과가 어떻게 달라지는지
 *   3. 한국어 쪼개기   형태소 분석 — 조사가 붙는 언어의 어려움
 *   4. 정규표현식 놀이터
 * ========================================================================== */

import {
  h, card, note, cols, deepDive, terms, table, screenHead, button, textarea, input,
  toggle, pillGroup, code, pyBox, mono, b, nextHint, fx,
} from '../lib/ui.js';
import { quizBlock, think } from '../lib/quiz.js';
import { CLEAN_RULES, clean, squeezeRepeat, preprocess, counter, STOPWORDS } from '../lib/nlp.js';
import { pos, nouns, morphs, POS_KO, josa } from '../lib/korean.js';
import { MESSY_TEXTS, STOPWORD_CANDIDATES } from '../data/corpus.js';
import { barChart } from '../lib/chart.js';

/* ══════════════════════════ 1. 배움 지도 ══════════════════════════════ */
const mapScreen = {
  id: 'map',
  title: '배움 지도',
  render(ctx) {
    const wrap = h('div');
    wrap.append(
      screenHead('배움 지도',
        '「글을 숫자로 바꾸는 일」에서 출발해 「내 문서로 답하는 AI」까지, 여섯 걸음으로 갑니다.'),

      card('여섯 걸음',
        h('div.cols',
          // ① 은 지금 보고 있는 이 화면이 속한 탭이다. 「가기」가 배움 지도 자기 자신을
          //    가리키면 아무 일도 안 일어나므로, 실제 내용이 시작되는 화면으로 보낸다.
          stepCard(ctx, '①', '글을 다루기', 'text', '지저분한 글을 다듬고 낱말로 쪼갠다',
            ['정제·정규화', '토큰화', '한국어 형태소', '정규표현식'], 'clean'),
          stepCard(ctx, '②', '숫자로 바꾸기', 'number', '낱말을 숫자 벡터로 만들어 계산한다',
            ['단어 주머니', 'TF-IDF', '코사인 유사도', '감성 분류', '단어 임베딩']),
          stepCard(ctx, '③', '맥락을 읽기', 'context', '순서와 문맥을 이해하는 신경망',
            ['신경망 기초', 'RNN·LSTM', 'Self-Attention', '위치 정보']),
          stepCard(ctx, '④', '말을 만들기', 'llm', '다음 낱말을 고르는 모델과 대화하기',
            ['다음 낱말 예측', '토큰·컨텍스트 창', '프롬프트', '파이프라인', '기억하는 챗봇']),
          stepCard(ctx, '⑤', '내 문서로 답하기', 'rag', '찾아서 · 넣고 · 답한다',
            ['임베딩 검색', '문서 자르기', 'RAG 5단계', '검색 품질 올리기']),
          stepCard(ctx, '⑥', '스스로 판단하기', 'agent', '갈림길과 되돌아오기가 있는 흐름',
            ['상태·노드·엣지', '분기와 루프', '도구 쓰는 에이전트', '멀티턴·멀티에이전트']),
        ),
      ),

      card('이 앱을 어떻게 쓰나요',
        h('ol',
          h('li', '왼쪽 위 ', b('탭'), ' 은 큰 걸음, 그 아래 ', b('알약 단추'), ' 는 화면입니다. 순서대로 가면 이야기가 이어집니다.'),
          h('li', h('span.deep-tag', '더 깊이'), ' 라고 적힌 접힌 상자는 ', b('원하는 사람만'), ' 열어 보는 심화 서술입니다. 안 열어도 흐름은 이어집니다.'),
          h('li', '화면마다 끝에 ', b('이해도 확인'), ' 문제가 있습니다. 틀려도 바로 왜 그런지 알려 줍니다.'),
          h('li', '마지막 탭 ', b('파이썬 실습실'), ' 에서는 배운 것을 코드로 직접 돌려 볼 수 있습니다. 설치가 필요 없습니다.'),
        ),
        note('good',
          h('b', '개인정보를 한 글자도 받지 않습니다. '),
          '이름·학번 칸이 없고 서버도 없습니다. 퀴즈 점수는 이 브라우저에만 남고, 위쪽 ',
          mono('내 기록 지우기'), ' 로 한 번에 지울 수 있습니다.'),
      ),

      card('무엇이 진짜로 계산되고, 무엇이 흉내인가',
        h('p', '학습 앱에서 가장 중요한 것은 ', b('무엇을 믿어도 되는지'), ' 를 아는 것입니다. 그래서 미리 밝혀 둡니다.'),
        table(['화면', '이 앱에서 하는 일', '진짜인가'], [
          ['TF-IDF·코사인 유사도', '수식 그대로 계산합니다', h('span', { style: { color: 'var(--good)' } }, '✅ 진짜 계산')],
          ['감성 분류기', '나이브 베이즈·로지스틱 회귀를 브라우저에서 실제로 학습시킵니다', h('span', { style: { color: 'var(--good)' } }, '✅ 진짜 학습')],
          ['단어 임베딩(Word2Vec)', 'Skip-gram 을 실제로 학습시킵니다 (1초쯤 걸립니다)', h('span', { style: { color: 'var(--good)' } }, '✅ 진짜 학습')],
          ['Self-Attention', '행렬 곱과 softmax 를 그대로 계산합니다', h('span', { style: { color: 'var(--good)' } }, '✅ 진짜 계산')],
          ['다음 낱말 예측기', '작은 n-gram 모델을 실제로 세어 만듭니다', h('span', { style: { color: 'var(--warn)' } }, '⚠️ 진짜지만 아주 작음')],
          ['한국어 형태소 분석', '사전과 규칙으로 만든 교실용 분석기입니다', h('span', { style: { color: 'var(--warn)' } }, '⚠️ 실제 도구와 다를 수 있음')],
          ['문장 임베딩', '뜻이 가까운 낱말을 미리 묶어 만든 「뜻 지도」를 씁니다', h('span', { style: { color: 'var(--warn)' } }, '⚠️ 원리는 같고 규모가 다름')],
          ['RAG 의 답 만들기', '찾은 글에서 근거 문장을 뽑아 조립합니다', h('span', { style: { color: 'var(--bad)' } }, '❌ 진짜 LLM 아님')],
        ]),
        note('warn',
          h('b', '왜 이렇게 만들었나 — '),
          '교실 인터넷이 끊겨도 열려야 하고, API 키와 요금 없이 모두가 같은 화면을 봐야 하기 때문입니다. ',
          '흉내인 부분은 그 화면에서도 노란 상자로 다시 알려 줍니다. ',
          '진짜 모델을 붙이는 방법은 각 화면의 ', mono('🐍 파이썬으로는'), ' 칸에 코드로 적어 두었습니다.'),
      ),

      quizBlock('t1/map', [
        {
          type: 'choice',
          q: '이 앱에서 「진짜로 학습이 일어나는」 화면은 어느 것일까요?',
          options: ['RAG 의 답 만들기', '단어 임베딩(Word2Vec)', '배움 지도', '문장 임베딩'],
          answer: 1,
          why: 'Word2Vec 화면은 Skip-gram 을 브라우저에서 실제로 학습시킵니다. 손실이 줄어드는 것을 직접 볼 수 있습니다.',
        },
        {
          type: 'choice',
          q: '이 앱이 진짜 LLM 을 넣지 않은 가장 큰 이유는 무엇일까요?',
          options: [
            '진짜 LLM 은 정확도가 낮아서',
            '교실 인터넷·API 키·요금 없이 누구나 열 수 있어야 해서',
            '파이썬으로 만들 수 없어서',
            '브라우저에서는 계산이 아예 불가능해서',
          ],
          answer: 1,
          why: '수업에서는 모두가 같은 화면을 같은 조건에서 볼 수 있어야 합니다. 그래서 계산은 브라우저 안에서 끝나도록 만들었습니다.',
        },
      ]),
      nextHint('전처리 공방으로 →', () => ctx.go('text', 'clean')),
    );
    return wrap;
  },
};

/** @param {string} [screenId] 비우면 그 탭의 첫 화면으로 간다 */
function stepCard(ctx, num, title, tabId, lead, items, screenId) {
  return h('div.card', { style: { margin: 0 } },
    h('div.card-body',
      h('div', { style: { display: 'flex', gap: '10px', alignItems: 'baseline' } },
        h('span', { style: { fontSize: '1.4rem', color: 'var(--accent)' } }, num),
        h('b', { style: { fontSize: '1.05rem' } }, title),
      ),
      h('p', { style: { color: 'var(--dim)', margin: '4px 0 8px', fontSize: '.9rem' } }, lead),
      h('div.pills', ...items.map((x) => h('span.pill', { style: { cursor: 'default' } }, x))),
      button('가기 →', () => {
        const tab = ctx.TABS.find((t) => t.id === tabId);
        ctx.go(tabId, screenId || tab.screens[0].id);
      }, 'ghost small'),
    ),
  );
}

/* ══════════════════════════ 2. 전처리 공방 ════════════════════════════ */
const cleanScreen = {
  id: 'clean',
  title: '전처리 공방',
  render(ctx) {
    const wrap = h('div');
    // 스위치는 모두 꺼진 채로 시작한다.
    // 다듬기 전의 날것을 먼저 보고, 규칙을 하나씩 켜며 무엇이 달라지는지 확인하게 하려는 것이다.
    const state = {
      text: MESSY_TEXTS[0],
      on: Object.fromEntries(CLEAN_RULES.map((r) => [r.key, false])),
      squeeze: false,
    };

    const stepsBox = h('div');
    const resultBox = h('div');

    const draw = () => {
      const { steps } = clean(state.text, state.on);
      let cur = state.text;
      const rows = [];
      rows.push(['원본', h('span', { style: { color: 'var(--dim)' } }, cur)]);
      for (const s of steps) {
        cur = s.after;
        rows.push([
          h('span', s.label, s.changed ? '' : h('span', { style: { color: 'var(--dim)' } }, ' (바뀐 것 없음)')),
          h('span', { style: { color: s.changed ? 'var(--fg)' : 'var(--dim)' } }, cur || '(비었음)'),
        ]);
      }
      if (state.squeeze) {
        const before = cur;
        cur = squeezeRepeat(cur);
        rows.push(['반복 글자 줄이기', h('span', { style: { color: before !== cur ? 'var(--fg)' : 'var(--dim)' } }, cur)]);
      }
      stepsBox.textContent = '';
      stepsBox.appendChild(table(['단계', '그 단계까지 마친 글'], rows));

      const toks = nouns(cur);
      const kept = toks.filter((t) => !STOPWORDS.includes(t) && t.length >= 2);
      resultBox.textContent = '';
      resultBox.append(
        h('p', b('명사만 뽑기 → '), h('span.tokens',
          ...toks.map((t) => {
            const drop = STOPWORDS.includes(t) || t.length < 2;
            return h('span.tok' + (drop ? '.tok-drop' : '.tok-Noun'), t);
          }))),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
          '흐릿하게 줄이 그어진 것은 ', b('불용어'), ' 이거나 한 글자라서 버린 낱말입니다.'),
        h('p', b('최종 결과 → '), mono('[' + kept.map((t) => `'${t}'`).join(', ') + ']')),
      );
    };

    const ta = textarea({
      value: state.text, rows: 3,
      onInput: (v) => { state.text = v; draw(); },
    });

    /* 스위치들을 미리 만들어 두어야 [모두 켜기 / 모두 끄기] 로 함께 움직일 수 있다 */
    const ruleToggles = CLEAN_RULES.map((r) =>
      toggle(r.label, false, (v) => { state.on[r.key] = v; draw(); }));
    const squeezeToggle = toggle('반복 글자 줄이기 (ㅋㅋㅋㅋ → ㅋㅋ)', false,
      (v) => { state.squeeze = v; draw(); });

    const setAll = (v) => {
      CLEAN_RULES.forEach((r, i) => { state.on[r.key] = v; ruleToggles[i].setValue(v); });
      state.squeeze = v;
      squeezeToggle.setValue(v);
      draw();
    };

    wrap.append(
      screenHead('전처리 공방', '요리 전에 재료를 손질하듯, 분석 전에 글을 다듬습니다. 규칙을 하나씩 켜 보세요.',
        '① 글을 다루기'),

      card('왜 다듬어야 할까',
        h('p', '사람은 아래 두 글이 같은 뜻이라는 것을 바로 압니다. 컴퓨터에게는 ', b('완전히 다른 글자 나열'), ' 일 뿐입니다.'),
        h('div.cols',
          h('div', mono('맛있어요!!!!'), h('br'), mono('맛있어요'), h('br'), mono('맛있어욬ㅋㅋ')),
          h('div', { style: { color: 'var(--dim)' } }, '→ 다듬지 않으면 셋을 서로 다른 낱말로 셉니다. 「맛있다」가 세 번 나온 것이 아니라 각각 한 번씩 나온 것이 되어 버립니다.'),
        ),
      ),

      card('직접 해 보기',
        h('div.pills',
          ...MESSY_TEXTS.map((t, i) => h('button.pill', {
            type: 'button',
            onclick: () => { state.text = t; ta.value = t; draw(); },
          }, `예문 ${i + 1}`)),
        ),
        ta,
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem', margin: '12px 0 4px' } },
          '스위치를 ', b('하나씩 켜'), ' 보세요. 켤 때마다 아래 표에 그 단계까지 마친 글이 쌓입니다.'),
        h('div', { style: { margin: '4px 0 12px' } }, ...ruleToggles, squeezeToggle),
        h('div.pills',
          button('모두 켜기', () => setAll(true), 'ghost small'),
          button('모두 끄기', () => setAll(false), 'ghost small'),
        ),
        stepsBox,
        h('hr', { style: { border: 0, borderTop: '1px solid var(--line)', margin: '16px 0' } }),
        resultBox,
      ),

      card('규칙 하나하나가 하는 일',
        table(['규칙', '무엇을 지우나', '왜'],
          CLEAN_RULES.map((r) => [r.label, mono(String(r.re).replace(/\/g[u]?$/, '/')), r.why])),
        note('info',
          '순서도 중요합니다. ', b('특수문자를 먼저 지우면'), ' 주소(URL)의 ', mono('://'), ' 가 사라져 ',
          mono('https'), ' 만 낱말로 남습니다. 그래서 ', b('주소를 먼저 지우고'), ' 특수문자를 지웁니다.'),
      ),

      deepDive('이 규칙들의 정규식을 한 조각씩 뜯어보기',
        h('p', '위 표의 패턴이 암호처럼 보이지만, 조각 몇 개만 알면 그대로 읽힙니다. ',
          '조각 자체는 ', h('button.link', {
            type: 'button', onclick: () => ctx.go('text', 'regex'),
          }, '정규표현식 놀이터'), ' 에서 직접 만져 볼 수 있습니다.'),

        h('p', b('① 주소 지우기 '), mono('https?:\\S+|www\\.\\S+')),
        table(['조각', '뜻'], [
          [mono('https'), '글자 그대로 h·t·t·p·s'],
          [mono('?'), '바로 앞의 s 가 ', h('b', '있거나 없거나'), ' — 그래서 http 와 https 를 한꺼번에 잡는다'],
          [mono(':'), '글자 그대로 콜론'],
          [mono('\\S'), '공백이 ', h('b', '아닌'), ' 글자 하나 (대문자 S 다. 소문자 \\s 는 공백을 뜻한다)'],
          [mono('+'), '앞의 것이 한 번 이상 — 주소가 끝날 때까지 쭉'],
          [mono('|'), '또는 — 왼쪽이 안 맞으면 오른쪽을 본다'],
          [mono('www\\.'), h('span', '점 앞의 역슬래시에 주의. 그냥 ', mono('.'), ' 는 「아무 글자」라서 www 뒤 아무 글자나 걸린다')],
        ], { compact: true }),

        h('p', b('② HTML 표시 지우기 '), mono('<[^>]+>')),
        h('p', { style: { color: 'var(--dim)' } },
          mono('[^>]'), ' 는 「', b('> 가 아닌'), ' 글자」입니다. 대괄호 ', b('안'), ' 의 맨 앞 ', mono('^'),
          ' 이 「아닌 것」이라는 뜻이에요. (대괄호 밖의 ', mono('^'), ' 은 「줄의 시작」이라 뜻이 전혀 다릅니다.)'),
        code([
          '<.+>      →  <b>강추</b> 를 통째로 잡는다   ← 「가능한 한 길게」 잡으려 하기 때문',
          '<[^>]+>   →  <b> 와 </b> 를 따로 잡는다    ← > 를 만나면 멈추므로',
        ].join('\n'), { copy: false }),

        h('p', b('③ 특수문자 지우기 '), mono('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]')),
        h('p', '「', b('한글·자모·영문·숫자·공백이 아닌 것'), '」을 모두 지웁니다. ',
          '지울 것을 하나하나 적는 대신 ', b('남길 것만 적고 나머지를 지우는'), ' 방식이라 훨씬 안전합니다. ',
          '세상의 모든 특수문자를 나열할 수는 없으니까요.'),

        h('p', b('④ 그런데 왜 자음·모음 규칙을 따로 두었나')),
        note('warn',
          h('b', '한글 「가-힣」에는 ㅋ 과 ㅠ 가 들어 있지 않습니다. '),
          '「가-힣」은 ', b('받침까지 갖춰 완성된 글자'), ' 11,172자만 가리킵니다. ',
          'ㅋㅋㅋ·ㅠㅠ 처럼 홀로 쓰인 자음·모음은 ', mono('ㄱ-ㅎ'), ' 과 ', mono('ㅏ-ㅣ'),
          ' 라는 ', b('다른 구역'), ' 에 있습니다.'),
        h('p', '이 앱은 특수문자 규칙에 ', mono('ㄱ-ㅎㅏ-ㅣ'), ' 를 넣어 ',
          b('자모를 일부러 남겨 두고'), ', 지울지 말지를 따로 껐다 켤 수 있게 했습니다. ',
          '감성 분석에서는 ', mono('ㅠㅠ'), ' 가 「부정」을 알려 주는 소중한 신호일 수 있기 때문입니다.'),
        note('info',
          h('b', '아래 파이썬 예제와 결과가 다른 까닭이 여기 있습니다. '),
          '예제의 ', mono('[^가-힣a-zA-Z0-9 ]'), ' 에는 자모가 빠져 있어 ',
          mono('ㅋㅋㅋ'), ' 이 특수문자 규칙 ', b('한 번에'), ' 지워집니다. ',
          '위 화면에서는 스위치 두 개를 모두 켜야 같은 결과가 됩니다. ',
          b('같은 「전처리」라도 패턴 한 글자 차이로 남는 것이 달라집니다.')),

        h('p', b('⑤ 이모지는 왜 범위로 적나 '), mono('[\\u{1F300}-\\u{1FAFF}…]')),
        h('p', { style: { color: 'var(--dim)' } },
          '이모지는 글자가 아니라 ', b('유니코드 번호 구역'), ' 으로 모여 있습니다. ',
          '「😀부터 🫠까지」를 번호 범위로 적는 것이 목록을 나열하는 것보다 확실합니다. ',
          '이때 패턴 끝에 ', mono('u'), ' 표시를 붙여야 번호를 글자 하나로 제대로 읽습니다.'),

        h('p', b('⑥ 반복 글자 줄이기 '), mono('(.)\\1{2,}'), ' → ', mono('$1$1')),
        table(['조각', '뜻'], [
          [mono('(.)'), '아무 글자 하나를 잡고 ', h('b', '기억해 둔다'), ' (괄호가 기억하라는 표시)'],
          [mono('\\1'), h('span', '아까 ', h('b', '기억해 둔 바로 그 글자'), ' — 「아무 글자」가 아니다')],
          [mono('{2,}'), '앞의 것이 두 번 이상 더 — 그래서 ', h('b', '모두 세 번 이상')],
          [mono('$1$1'), '바꿀 내용. 기억해 둔 글자를 두 번 쓴다 (파이썬에서는 ' + '\\1\\1' + ')'],
        ], { compact: true }),
        code([
          'ㅋㅋㅋㅋㅋ  →  (.) 가 ㅋ 을 기억,  \\1{2,} 가 나머지 ㅋㅋㅋㅋ 에 걸림  →  ㅋㅋ',
          'ㅋㅋ        →  세 번이 안 되어 걸리지 않음                        →  ㅋㅋ (그대로)',
          '진짜아아아  →  ㅇ 이 아니라 「아」를 기억한다                      →  진짜아아',
        ].join('\n'), { copy: false }),

        h('p', b('⑦ 왜 빈 문자열이 아니라 「공백」으로 바꾸나')),
        code([
          '노트!볼펜',
          '   → 빈 문자열로 바꾸면   "노트볼펜"     ← 없던 낱말이 하나 생겼다 ❌',
          '   → 공백으로 바꾸면     "노트 볼펜"    ← 두 낱말로 제대로 갈린다 ✅',
        ].join('\n'), { copy: false }),
        h('p', '대신 공백이 여기저기 늘어나므로, ', b('맨 마지막에 연속 공백 정리'), ' 가 꼭 필요해집니다. ',
          '규칙의 순서가 정해져 있는 이유입니다.'),

        h('p', b('⑧ 순서를 바꾸면 왜 망가지나')),
        code([
          '[올바른 순서] 주소 먼저',
          '  "https://a.test/1 강추"',
          '     → 주소 지우기    →  " 강추"',
          '     → 특수문자 지우기 →  " 강추"          ✅',
          '',
          '[뒤바뀐 순서] 특수문자 먼저',
          '  "https://a.test/1 강추"',
          '     → 특수문자 지우기 →  "https a test 1 강추"   ← 주소 모양이 흩어졌다',
          '     → 주소 지우기    →  "https a test 1 강추"   ← http:// 가 없으니 못 잡는다 ❌',
        ].join('\n'), { copy: false }),
        h('p', '위 화면에서 ', b('특수문자 지우기만 먼저 켜'), ' 보면 이 일이 실제로 일어나는 것을 볼 수 있습니다.'),
      ),

      deepDive('정제와 정규화는 어떻게 다른가',
        h('p', b('정제(Cleaning)'), ' 는 ', b('없애는'), ' 일입니다. 특수문자·주소·태그처럼 분석에 방해되는 것을 지웁니다.'),
        h('p', b('정규화(Normalization)'), ' 는 ', b('통일하는'), ' 일입니다. 같은 뜻인데 꼴이 다른 것을 한 꼴로 맞춥니다.'),
        terms([
          ['대소문자 통일', 'Python / PYTHON / python → python'],
          ['반복 글자 줄이기', 'ㅋㅋㅋㅋㅋ → ㅋㅋ (세 번 이상을 두 번으로)'],
          ['어간 추출(Stemming)', '영어에서 철자 규칙만으로 꼬리를 자른다. running → run, 하지만 organization → organ 처럼 지나치게 깎이기도 한다.'],
          ['표제어 추출(Lemmatization)', '사전을 찾아 원형으로 되돌린다. was → be, mice → mouse. 느리지만 정확하다.'],
          ['한국어는?', '조사·어미가 붙으므로 형태소 분석이 이 역할을 대신한다 (다음 화면).'],
        ]),
        h('p', '어디까지 정규화할지는 ', b('목적에 따라 다릅니다'), '. 감성 분석에서는 ',
          mono('ㅠㅠ'), ' 가 중요한 신호일 수 있으므로 일부러 남기기도 합니다. ',
          '「무조건 다 지운다」가 아니라 「무엇을 남길지 정한다」가 전처리의 본질입니다.'),
      ),

      pyBox([
        'import re',
        '',
        'def 정제하기(글):',
        '    글 = re.sub(r\'http\\S+\', \' \', 글)          # 주소 지우기',
        '    글 = re.sub(r\'<[^>]+>\', \' \', 글)           # HTML 표시 지우기',
        '    글 = re.sub(r\'[^가-힣a-zA-Z0-9 ]\', \' \', 글) # 한글·영문·숫자·공백만 남기기',
        '    글 = re.sub(r\'(.)\\1{2,}\', r\'\\1\\1\', 글)     # 반복 글자 줄이기',
        '    글 = re.sub(r\'\\s+\', \' \', 글).strip()       # 연속 공백 정리',
        '    return 글',
        '',
        'print(정제하기("ㅋㅋㅋ 노트 좋아요!!!! https://a.test/1"))',
      ].join('\n'), '마지막 탭 「파이썬 실습실」에서 그대로 실행해 볼 수 있습니다'),

      quizBlock('t1/clean', [
        {
          type: 'choice',
          q: () => [mono("re.sub(r'[^가-힣 ]', '', '노트 좋아요!!!')"), ' 의 결과는?'],
          options: ["'노트 좋아요!!!'", "'노트 좋아요'", "'노트좋아요'", "'!!!'"],
          answer: 1,
          why: '[^가-힣 ] 은 「한글과 공백이 아닌 것」입니다. 느낌표만 지워지고 공백은 남습니다.',
        },
        {
          type: 'choice',
          q: '주소(URL) 를 지우는 규칙을 특수문자 규칙보다 먼저 두는 까닭은?',
          options: [
            '주소가 더 길어서',
            '특수문자를 먼저 지우면 주소의 모양이 깨져 통째로 지울 수 없어서',
            '주소에는 한글이 없어서',
            '순서는 사실 상관없어서',
          ],
          answer: 1,
          why: 'https://a.test 에서 특수문자를 먼저 지우면 「https a test」 처럼 흩어져, 주소 모양(http로 시작)이 사라집니다.',
        },
        {
          type: 'multi',
          q: '전처리에서 「무조건 지우면 안 되는」 경우를 모두 고르세요.',
          options: [
            '감성 분석에서 ㅠㅠ, ㅋㅋ 같은 표현',
            '주문번호 A-1234 의 붙임표',
            '문장 끝의 마침표 (문장 단위로 나눌 계획일 때)',
            '분석에 쓰지 않는 HTML 태그',
          ],
          answer: [0, 1, 2],
          why: '지울지 말지는 목적이 정합니다. 감정 신호·식별자·문장 경계는 지우면 정보를 잃습니다. HTML 태그는 대개 안전하게 지울 수 있습니다.',
        },
      ]),
      nextHint('한국어 쪼개기로 →', () => ctx.go('text', 'morph')),
    );

    draw();
    return wrap;
  },
};

/* ══════════════════════════ 3. 한국어 쪼개기 ══════════════════════════ */
const morphScreen = {
  id: 'morph',
  title: '한국어 쪼개기',
  render(ctx) {
    const wrap = h('div');
    const state = { text: '도서관에서 빌린 책을 오늘 반납했어요', mode: 'pos', stem: false };
    const out = h('div');

    const draw = () => {
      out.textContent = '';
      const simple = state.text.split(/\s+/).filter(Boolean);
      out.append(
        h('p', b('① 그냥 띄어쓰기로 자르면 '), h('span.tokens', ...simple.map((t) => h('span.tok', t)))),
        h('p', { style: { color: 'var(--dim)', fontSize: '.9rem', marginTop: '-6px' } },
          '「도서관에서」와 「도서관은」이 서로 다른 낱말이 됩니다. 같은 도서관인데요.'),
      );

      const list = pos(state.text, { stem: state.stem });
      if (state.mode === 'pos') {
        out.append(h('p', b('② 형태소 + 품사 '), h('span.tokens',
          ...list.map((x) => h(`span.tok.tok-${x.p}`, x.t, h('span.pos', POS_KO[x.p] || x.p))))));
      } else if (state.mode === 'morphs') {
        out.append(h('p', b('② 형태소만 '), mono('[' + morphs(state.text, { stem: state.stem }).map((t) => `'${t}'`).join(', ') + ']')));
      } else {
        const ns = nouns(state.text);
        out.append(
          h('p', b('② 명사만 '), mono('[' + ns.map((t) => `'${t}'`).join(', ') + ']')),
          h('p', { style: { color: 'var(--dim)', fontSize: '.9rem' } },
            '텍스트 분석에서 가장 많이 쓰는 방법입니다. 문장의 뜻을 짊어지는 것이 대개 명사이기 때문입니다.'),
        );
      }
    };

    const ta = input({ value: state.text, onInput: (v) => { state.text = v; draw(); } });

    wrap.append(
      screenHead('한국어 쪼개기', '조사와 어미가 달라붙는 언어를, 뜻을 가진 가장 작은 조각으로 나눕니다.', '① 글을 다루기'),

      card('영어는 쉽고 한국어는 어렵다',
        h('div.cols',
          h('div',
            h('p', b('영어')),
            mono('I go / You go / He goes'),
            h('p', { style: { color: 'var(--dim)' } }, '낱말 꼴이 거의 그대로입니다. 공백으로 자르면 끝.'),
          ),
          h('div',
            h('p', b('한국어')),
            mono('나는 / 내가 / 나에게 / 나조차'),
            h('p', { style: { color: 'var(--dim)' } }, '같은 「나」인데 뒤에 붙는 것이 계속 바뀝니다. 공백으로 자르면 넷이 서로 다른 낱말이 됩니다.'),
          ),
        ),
        note('info', '이런 언어를 ', b('교착어'), ' 라고 합니다. 뜻을 가진 조각(', b('형태소'), ')이 줄줄이 달라붙습니다.'),
      ),

      card('직접 쪼개 보기',
        h('div.pills',
          ...['도서관에서 빌린 책을 오늘 반납했어요',
            '노트 종이가 두꺼워서 필기가 편했어요',
            '연체하면 대출이 정지된다고 안내받았습니다',
            '나는 나를 나에게 나조차 모른다'].map((t) => h('button.pill', {
            type: 'button',
            onclick: () => { state.text = t; ta.value = t; draw(); },
          }, t.slice(0, 12) + '…')),
        ),
        ta,
        h('div', { style: { marginTop: '10px' } },
          pillGroup([
            { label: '형태소 + 품사', value: 'pos' },
            { label: '형태소만', value: 'morphs' },
            { label: '명사만', value: 'nouns' },
          ], (v) => { state.mode = v; draw(); }),
          toggle('원형으로 되돌리기 (반납했어요 → 반납하다)', false, (v) => { state.stem = v; draw(); }),
        ),
        out,
        h('div.honest',
          h('b', '정직하게 밝힙니다 — '),
          '이 분석기는 브라우저에서 돌리려고 사전과 규칙으로 직접 만든 ', b('교실용'), ' 입니다. ',
          '사전에 없는 낱말은 통째로 명사로 봅니다. 실제 도구(KoNLPy 의 Okt·Kkma 등)와 결과가 다를 수 있습니다. ',
          '여기서 볼 것은 「정확한 분석」이 아니라 ', b('왜 쪼개야 하는가'), ' 입니다.'),
      ),

      card('품사 표시 읽는 법',
        table(['표시', '뜻', '예'], [
          ['Noun', '명사', '도서관, 책, 오늘'],
          ['Josa', '조사', '에서, 을, 는, 이'],
          ['Verb', '동사', '반납했어요, 빌렸다'],
          ['Adjective', '형용사', '두꺼워서, 좋았다'],
          ['Adverb', '부사', '정말, 아주, 너무'],
          ['Number / Alpha', '숫자 / 영문', '21, python'],
        ], { compact: true }),
      ),

      card('불용어 — 빼면 오히려 잘 보이는 낱말들',
        h('p', '조사·접속사처럼 어디에나 나오는 낱말은 문서를 구별하는 데 도움이 되지 않습니다. 이런 것을 ',
          b('불용어(Stopword)'), ' 라고 하고, 대개 지웁니다.'),
        h('p', mono('[' + STOPWORD_CANDIDATES.map((w) => `'${w}'`).join(', ') + ']')),
        note('warn',
          h('b', '한국어에는 「공식 불용어 목록」이 없습니다. '),
          '영어에는 라이브러리가 주는 목록이 있지만, 한국어는 ', b('분석 목적에 맞게 직접 만들어야'), ' 합니다. ',
          '예를 들어 「안」은 보통 불용어이지만, ', mono('조용하지 않아서'), ' 처럼 부정을 만드는 자리에서는 지우면 안 됩니다.'),
      ),

      deepDive('형태소 분석기는 안에서 무슨 일을 하나',
        h('p', '이 앱의 분석기는 어절 하나마다 아래 순서로 시도합니다. 실제 도구도 큰 틀은 비슷합니다(다만 통계 모형을 함께 씁니다).'),
        h('ol',
          h('li', '뒤에서 ', b('조사'), ' 를 떼어 본다 — 「에서도」는 「에서」 + 「도」 두 겹까지'),
          h('li', '남은 앞부분이 ', b('명사 사전'), ' 에 있으면 확정 — 조금만 뗀 갈래를 먼저 본다'),
          h('li', '「명사 + 하다」 꼴인지 본다 — ', mono('연체하면 → 연체 + 하면')),
          h('li', b('용언(동사·형용사) 어간'), ' 으로 시작하는지 본다'),
          h('li', b('복합명사'), ' 인지 본다 — ', mono('도서관이용 → 도서관 + 이용')),
          h('li', '다 실패하면 통째로 명사로 추정한다'),
        ),
        note('info',
          '②에서 「조금만 뗀 갈래를 먼저」가 중요합니다. ', mono('종이가'), ' 를 욕심내어 두 번 떼면 ',
          mono('종 + 이 + 가'), ' 가 되어 버립니다. ', mono('종이 + 가'), ' 가 맞습니다.'),
        h('p', b('진짜 도구는 어떻게 다른가 — '),
          '실제 분석기는 사전만 쓰지 않고, 「이 자리에 어떤 품사가 올 확률이 높은가」를 통계로 함께 봅니다. ',
          '그래서 사전에 없는 새 낱말도 문맥으로 어느 정도 처리합니다. 대신 자바 같은 무거운 환경이 필요합니다.'),
      ),

      pyBox([
        'import korean   # 이 앱의 교실용 분석기 (실제 수업에서는 konlpy 의 Okt 를 씁니다)',
        '',
        '문장 = "도서관에서 빌린 책을 오늘 반납했어요"',
        '',
        'print(korean.morphs(문장))              # 형태소만',
        'print(korean.pos(문장))                 # 형태소 + 품사',
        'print(korean.nouns(문장))               # 명사만 ← 가장 많이 씀',
        'print(korean.morphs(문장, stem=True))   # 원형으로 되돌리기',
      ].join('\n'), '실제 수업 코드는 from konlpy.tag import Okt / okt.nouns(문장) 입니다'),

      think('「조용하지 않아서 좋았어요」에서 「안」이나 「않다」를 불용어로 지우면 뜻이 어떻게 바뀔까요? ',
        '불용어 목록을 만들 때 무엇을 조심해야 할지 이야기해 보세요.'),

      quizBlock('t1/morph', [
        {
          type: 'choice',
          q: () => ['「학교에서」를 형태소로 바르게 나눈 것은?'],
          options: ['학 + 교에서', '학교 + 에서', '학교에 + 서', '학교에서 (나눌 수 없음)'],
          answer: 1,
          why: '「학교」는 명사, 「에서」는 조사입니다. 뜻을 가진 가장 작은 조각으로 나눕니다.',
        },
        {
          type: 'choice',
          q: '텍스트 분석에서 명사만 뽑아 쓰는 일이 많은 까닭은?',
          options: [
            '명사가 가장 짧아서',
            '명사가 문장의 주제를 가장 잘 나타내서',
            '조사는 컴퓨터가 읽지 못해서',
            '동사는 개수가 적어서',
          ],
          answer: 1,
          why: '「도서관 / 책 / 반납」만 봐도 무슨 글인지 짐작됩니다. 조사와 어미는 문법 역할을 할 뿐 주제를 알려 주지 않습니다.',
        },
        {
          type: 'short',
          q: () => ['뜻을 가진 가장 작은 언어 단위를 무엇이라고 하나요? (세 글자)'],
          accept: ['형태소'],
          why: '형태소(morpheme)입니다. 「맛있었어요」는 맛있 + 었 + 어요 세 형태소로 나뉩니다.',
        },
      ]),
      nextHint('정규표현식 놀이터로 →', () => ctx.go('text', 'regex')),
    );
    draw();
    return wrap;
  },
};

/* ══════════════════════════ 4. 정규표현식 놀이터 ══════════════════════ */
const PATTERNS = [
  { p: '[가-힣]+', label: '한글 덩어리', why: '한글 글자가 하나 이상 이어진 부분' },
  { p: '[^가-힣 ]', label: '한글·공백이 아닌 것', why: '이것을 지우면 한글만 남는다' },
  { p: '[0-9]+', label: '숫자', why: '숫자가 하나 이상 이어진 부분' },
  { p: 'https?:\\S+', label: '주소(URL)', why: 'http 또는 https 로 시작해 공백 아닌 글자가 이어지는 덩어리' },
  { p: '<[^>]+>', label: 'HTML 표시', why: '< 로 시작해 > 로 끝나는 덩어리' },
  { p: '\\s+', label: '공백 덩어리', why: '공백·탭·줄바꿈이 하나 이상' },
  { p: '(.)\\1{2,}', label: '세 번 이상 반복된 글자', why: '(.) 로 한 글자를 잡고, \\1 로 「같은 글자」를 다시 가리킨다' },
  { p: '[가-힣]{2,3}(?=하)', label: '「하」 앞의 두세 글자', why: '뒤를 내다보는 패턴 — 연체하면 에서 「연체」만 잡는다' },
];

const regexScreen = {
  id: 'regex',
  title: '정규표현식 놀이터',
  render(ctx) {
    const wrap = h('div');
    const state = {
      text: 'ㅋㅋㅋ 노트 좋아요!!!! https://shop.test/item/12 가격은 3500원 <b>강추</b>',
      pattern: '[^가-힣 ]',
      replace: '',
    };
    const view = h('div');
    const outBox = h('div');

    const draw = () => {
      view.textContent = '';
      outBox.textContent = '';
      let re;
      try {
        re = new RegExp(state.pattern, 'g');
      } catch (e) {
        view.appendChild(note('bad', h('b', '패턴을 이해할 수 없습니다: '), String(e.message)));
        return;
      }
      // 찾은 부분에 색칠
      const marked = h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.92rem', lineHeight: '2.1', background: '#0e1728', padding: '12px', borderRadius: '10px', border: '1px solid var(--line)' } });
      let last = 0;
      let m;
      let count = 0;
      re.lastIndex = 0;
      while ((m = re.exec(state.text)) !== null) {
        if (m.index > last) marked.appendChild(document.createTextNode(state.text.slice(last, m.index)));
        marked.appendChild(h('span', {
          style: { background: 'var(--accent)', color: '#08111f', borderRadius: '4px', padding: '1px 3px' },
        }, m[0] || '␀'));
        last = m.index + (m[0].length || 1);
        count += 1;
        if (m[0] === '') re.lastIndex += 1;
        if (count > 400) break;
      }
      if (last < state.text.length) marked.appendChild(document.createTextNode(state.text.slice(last)));
      view.append(h('p', b(`찾은 곳 ${count}군데`)), marked);

      const replaced = state.text.replace(new RegExp(state.pattern, 'g'), state.replace);
      outBox.append(
        h('p', b('바꾼 결과 '), h('span', { style: { color: 'var(--dim)', fontSize: '.86rem' } },
          mono(`re.sub(r'${state.pattern}', '${state.replace}', 글)`))),
        h('div', { style: { fontFamily: 'var(--mono)', fontSize: '.92rem', background: '#0a1120', padding: '12px', borderRadius: '10px', border: '1px solid var(--line)' } },
          replaced || '(모두 지워졌습니다)'),
      );
    };

    const patInput = input({ value: state.pattern, onInput: (v) => { state.pattern = v; draw(); } });
    const repInput = input({ value: state.replace, placeholder: '(비우면 지웁니다)', onInput: (v) => { state.replace = v; draw(); } });
    const textInput = textarea({ value: state.text, rows: 2, onInput: (v) => { state.text = v; draw(); } });

    wrap.append(
      screenHead('정규표현식 놀이터', '「이런 모양의 글자를 찾아라」를 짧게 적는 방법. 전처리의 주된 연장입니다.', '① 글을 다루기'),

      card('먼저 만져 보기',
        h('p', b('대상 글')), textInput,
        h('div.cols',
          h('div', h('p', b('찾을 패턴')), patInput),
          h('div', h('p', b('무엇으로 바꿀까')), repInput),
        ),
        h('div.pills', ...PATTERNS.map((x) => h('button.pill', {
          type: 'button',
          onclick: () => { state.pattern = x.p; patInput.value = x.p; draw(); },
        }, x.label))),
        view,
        outBox,
      ),

      card('자주 쓰는 조각',
        table(['조각', '뜻', '보기'], [
          [mono('[가-힣]'), '한글 한 글자', '가, 힣, 책'],
          [mono('[a-zA-Z]'), '영문 한 글자', 'a, Z'],
          [mono('[0-9]'), '숫자 한 글자', '0, 7'],
          [mono('[^…]'), '「…이 아닌 것」', mono('[^가-힣]') + ' = 한글이 아닌 것'],
          [mono('+'), '앞의 것이 한 번 이상', mono('[0-9]+') + ' = 350, 7'],
          [mono('*'), '앞의 것이 없거나 여러 번', ''],
          [mono('?'), '앞의 것이 있거나 없거나', mono('https?') + ' = http, https'],
          [mono('.'), '아무 글자 하나', ''],
          [mono('\\s'), '공백·탭·줄바꿈', ''],
          [mono('\\S'), '공백이 아닌 것', ''],
          [mono('{2,}'), '두 번 이상 반복', ''],
          [mono('(…)'), '묶어서 기억해 두기', mono('\\1') + ' 로 다시 부른다'],
        ], { compact: true }),
      ),

      card('고른 패턴은 무슨 뜻인가',
        h('div', ...PATTERNS.map((x) => h('p', { style: { margin: '6px 0' } },
          mono(x.p), ' — ', h('span', { style: { color: 'var(--dim)' } }, x.why)))),
      ),

      deepDive('정규표현식을 외워야 하나요',
        h('p', '아닙니다. 실무에서도 자주 쓰는 예닐곱 개만 손에 익히고, 나머지는 그때그때 찾아 씁니다. ',
          '중요한 것은 ', b('무엇을 지우고 무엇을 남길지 스스로 정하는 것'), ' 입니다. 패턴을 적는 일은 그다음입니다.'),
        h('p', b('욕심내면 위험합니다. '), '한 줄에 모든 것을 처리하려고 복잡한 패턴을 쓰면, ',
          '나중에 자기가 쓴 것도 읽지 못합니다. 여러 줄로 나눠 한 번에 하나씩 처리하는 편이 낫습니다.'),
        h('p', b('되돌아보기(lookahead) — '), mono('[가-힣]{2,3}(?=하)'), ' 의 ', mono('(?=하)'), ' 는 ',
          '「뒤에 하가 오는 자리」라는 조건만 보고 ', b('하 자체는 잡지 않습니다'), '. ',
          mono('연체하면'), ' 에서 ', mono('연체'), ' 만 잡히는 이유입니다.'),
        h('p', b('탐욕(greedy) — '), mono('<.+>'), ' 는 ', mono('<b>강추</b>'), ' 를 ', b('통째로'), ' 잡습니다. ',
          '「가능한 한 길게」 잡으려 하기 때문입니다. 그래서 태그 하나씩 잡으려면 ',
          mono('<[^>]+>'), ' 처럼 「> 가 아닌 것들」로 막아야 합니다. 직접 두 패턴을 넣어 비교해 보세요.'),
      ),

      quizBlock('t1/regex', [
        {
          type: 'choice',
          q: () => [mono('[^0-9]'), ' 는 무엇을 가리키나요?'],
          options: ['숫자', '숫자가 아닌 글자', '0부터 9까지의 낱말', '아무 글자'],
          answer: 1,
          why: '대괄호 안 맨 앞의 ^ 는 「아닌 것」이라는 뜻입니다. 대괄호 밖의 ^(줄 시작)와는 다른 역할입니다.',
        },
        {
          type: 'choice',
          q: () => [mono('(.)\\1{2,}'), ' 가 잡아내는 것은?'],
          options: [
            '아무 글자 세 개',
            '같은 글자가 세 번 이상 이어진 부분',
            '괄호로 묶인 부분',
            '점(.)이 두 개 이상',
          ],
          answer: 1,
          why: '(.) 이 한 글자를 기억하고, \\1 이 「그와 같은 글자」를 가리킵니다. {2,} 로 두 번 더 반복 = 모두 세 번 이상입니다.',
        },
        {
          type: 'choice',
          q: () => ['글에서 태그 ', mono('<b>'), ' 와 ', mono('</b>'), ' 를 각각 지우려면?'],
          options: [mono('<.+>'), mono('<[^>]+>'), mono('<*>'), mono('[<>]+')],
          answer: 1,
          why: '<.+> 는 욕심내어 <b>강추</b> 를 통째로 잡아 「강추」까지 지웁니다. <[^>]+> 는 > 를 만나면 멈춥니다.',
        },
      ]),
      nextHint('② 숫자로 바꾸기로 →', () => ctx.go('number', 'bow')),
    );
    draw();
    return wrap;
  },
};

export default {
  id: 'text',
  num: 'Ⅰ',
  title: '글을 다루기',
  screens: [mapScreen, cleanScreen, morphScreen, regexScreen],
};
