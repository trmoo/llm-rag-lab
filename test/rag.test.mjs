/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * rag.test.mjs — 안내문에 적힌 값과 앱이 내놓는 답이 어긋나지 않는지 지킨다.
 *
 * 이 시험이 깨지면 「학생이 화면에서 보는 답」이 「문서에 적힌 사실」과
 * 달라졌다는 뜻이므로, 반드시 원인을 찾아 고쳐야 한다.
 * ========================================================================== */

import { describe, test, eq, near, ok, includes } from './harness.mjs';
import {
  splitByChars, splitByHeadings, buildIndex, retrieve, buildPrompt,
  answerFromContext, answerWithoutContext, sentences, expandQuery, PROMPT_PARTS,
} from '../src/lib/rag.js';
import { embed, wordVector, pca2, topAxes, DIM, AXIS_NAMES } from '../src/lib/embed.js';
import { cosine } from '../src/lib/nlp.js';
import {
  LIBRARY_DOC, STORE_DOC, DOC_FACTS, NO_ANSWER_QUESTIONS, NAIVE_GUESSES, SAMPLE_QUESTIONS,
} from '../src/data/docs.js';

const sections = splitByHeadings(LIBRARY_DOC);
const index = buildIndex(sections);

describe('임베딩', () => {
  test('차원 수와 축 이름 개수가 맞는다', () => {
    eq(embed('책').length, DIM);
    eq(AXIS_NAMES.length, DIM);
  });
  test('길이가 1로 맞춰져 있다', () => near(Math.hypot(...embed('도서관에서 책을 빌렸어요')), 1, 1e-9));
  test('뜻이 가까우면 벡터도 가깝다', () => {
    ok(cosine(wordVector('강아지'), wordVector('개')) > 0.6, '강아지와 개는 가까워야 한다');
    ok(cosine(wordVector('강아지'), wordVector('코드')) < 0.4, '강아지와 코드는 멀어야 한다');
  });
  test('같은 낱말은 늘 같은 벡터 (모르는 낱말도)', () => {
    eq(wordVector('쿼드러플렛'), wordVector('쿼드러플렛'));
  });
  test('낱말이 안 겹쳐도 뜻이 같으면 가깝다', () => {
    const a = embed('강아지가 공원에서 뛰어놀아요');
    const b = embed('개가 야외에서 달리고 있어요');
    const c = embed('파이썬 코드에서 오류가 났어요');
    ok(cosine(a, b) > cosine(a, c), '뜻이 같은 쪽이 더 가까워야 한다');
  });
  test('2차원으로 줄여도 개수가 유지된다', () => {
    const vs = ['책', '도서관', '커피', '노트'].map(embed);
    const { points } = pca2(vs);
    eq(points.length, 4);
    ok(points.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])), '좌표가 숫자여야 한다');
  });
  test('가장 세게 켜진 갈래를 뽑을 수 있다', () => {
    const t = topAxes(embed('도서관에서 책을 빌렸어요'), 3).map((x) => x.name);
    ok(t.includes('책·도서관'), `책·도서관 축이 켜져야 합니다: ${t}`);
  });
});

describe('문서 자르기', () => {
  test('제목대로 자르면 섹션 수만큼 나온다', () => {
    eq(sections.length, 8);
    eq(sections[1].metadata.섹션, '대출');
  });
  test('꼬리표(섹션 제목)가 함께 저장된다', () => {
    ok(sections.every((s) => s.metadata.섹션), '모든 조각에 섹션 이름이 있어야 한다');
  });
  test('글자 수로 자르면 조각이 크기를 크게 넘지 않는다', () => {
    const ch = splitByChars(LIBRARY_DOC, { chunkSize: 300, overlap: 0 });
    ok(ch.every((c) => c.length <= 340), '조각이 너무 큽니다');
    ok(ch.length > 1, '여러 조각으로 나뉘어야 한다');
  });
  test('겹치기를 켜면 앞 조각의 끝이 다음 조각 앞에 붙는다', () => {
    const a = splitByChars(LIBRARY_DOC, { chunkSize: 200, overlap: 0 });
    const b = splitByChars(LIBRARY_DOC, { chunkSize: 200, overlap: 40 });
    ok(b[1].startsWith(a[0].slice(-40)), '겹치는 부분이 앞뒤를 이어야 한다');
  });
  test('문장 나누기는 「…하루마다 」의 다 에서 끊기지 않는다', () => {
    const s = sentences('늦은 하루마다 이틀씩 대출이 정지됩니다. 다음 문장입니다.');
    eq(s.length, 2);
    includes(s[0], '이틀씩');
  });
});

describe('검색', () => {
  test('질문마다 관련 있는 섹션을 1등으로 올린다', () => {
    const cases = [
      ['책은 며칠까지 빌릴 수 있나요?', '대출'],
      ['열람실 자리는 얼마나 쓸 수 있나요?', '열람실과 자리 예약'],
      ['전자책도 연체가 되나요?', '전자책'],
    ];
    for (const [q, want] of cases) {
      const r = retrieve(index, q, { k: 1, mode: 'hybrid' });
      eq(r.hits[0].doc.metadata.섹션, want, `질문: ${q}`);
    }
  });
  test('꼬리표로 범위를 좁힐 수 있다', () => {
    const r = retrieve(index, '기간', { k: 3, mode: 'hybrid', filter: { 섹션: '대출' } });
    ok(r.hits.every((h) => h.doc.metadata.섹션 === '대출'), '대출 섹션만 나와야 한다');
  });
  test('기준 점수를 아주 높이면 아무것도 안 나온다', () => {
    const r = retrieve(index, '주차장', { k: 3, mode: 'hybrid', threshold: 0.9 });
    eq(r.hits.length, 0);
  });
  test('MMR 을 켜면 서로 다른 섹션을 고른다', () => {
    const r = retrieve(index, '대출과 반납', { k: 3, mode: 'hybrid', mmr: true, lambda: 0.4, fetchK: 6 });
    const names = r.hits.map((h) => h.doc.metadata.섹션);
    eq(new Set(names).size, names.length, '같은 섹션이 겹치면 안 된다');
  });
  test('낱말 검색만으로는 못 찾는 것을 뜻 검색이 찾는다', () => {
    const kw = retrieve(index, '책을 빌리는 방법', { k: 1, mode: 'keyword' });
    const vec = retrieve(index, '책을 빌리는 방법', { k: 1, mode: 'vector' });
    ok(vec.hits.length > 0 && vec.hits[0].vec > 0.3, '뜻 검색은 관련 조각을 찾아야 한다');
    ok(kw.hits.length > 0, '낱말 검색도 무언가는 돌려준다');
  });
  test('질문 넓히기가 「며칠 → 기간·일」로 늘린다', () => {
    const r = expandQuery(['며칠', '책']);
    ok(r.tokens.includes('기간'), '기간이 더해져야 한다');
    ok(r.added.length >= 1, '무엇이 넓혀졌는지 기록해야 한다');
  });
});

describe('답 만들기', () => {
  test(`안내문에 적힌 값 ${DOC_FACTS.length}가지를 모두 맞힌다`, () => {
    const wrong = [];
    for (const f of DOC_FACTS) {
      const r = retrieve(index, f.question, { k: 3, mode: 'hybrid' });
      const a = answerFromContext(f.question, r.hits, { strict: true });
      if (!f.accept.some((x) => a.text.includes(x))) wrong.push(`${f.question} → ${a.text.slice(0, 50)}`);
    }
    eq(wrong, [], '틀린 답이 있습니다');
  });
  test('안내문에 없는 질문에는 「없다」고 답한다', () => {
    const wrong = [];
    for (const q of NO_ANSWER_QUESTIONS) {
      const r = retrieve(index, q, { k: 3, mode: 'hybrid' });
      const a = answerFromContext(q, r.hits, { strict: true });
      if (a.supported) wrong.push(`${q} → ${a.text.slice(0, 50)}`);
    }
    eq(wrong, [], '지어낸 답이 있습니다');
  });
  test('규칙을 끄면 지어낸다 — 규칙 한 줄의 힘', () => {
    const q = '주차장은 무료인가요?';
    const r = retrieve(index, q, { k: 3, mode: 'hybrid' });
    const strict = answerFromContext(q, r.hits, { strict: true });
    const loose = answerFromContext(q, r.hits, {
      strict: false, guess: answerWithoutContext(q, NAIVE_GUESSES).text,
    });
    ok(!strict.supported && !strict.hallucinated, '규칙이 있으면 모른다고 해야 한다');
    ok(loose.hallucinated, '규칙이 없으면 지어내야 한다');
  });
  test('출처를 함께 돌려준다', () => {
    const q = '책은 며칠까지 빌릴 수 있나요?';
    const r = retrieve(index, q, { k: 2, mode: 'hybrid' });
    const a = answerFromContext(q, r.hits, { strict: true, cite: true });
    includes(a.text, '📎 근거');
    includes(a.text, '대출');
  });
  test('근거는 한 조각에서만 모은다', () => {
    const q = '열람실 자리는 얼마나 쓸 수 있나요?';
    const r = retrieve(index, q, { k: 3, mode: 'hybrid' });
    const a = answerFromContext(q, r.hits, { strict: true });
    eq(a.used.length, 1, '여러 조각을 섞으면 답이 뒤죽박죽이 된다');
  });
});

describe('문서 없이 답하기 (환각 재현)', () => {
  test('그럴듯하지만 안내문과 다른 답을 내놓는다', () => {
    const a = answerWithoutContext('책은 며칠까지 빌릴 수 있나요?', NAIVE_GUESSES);
    ok(a.wrong, '틀린 답이라고 표시되어야 한다');
    includes(a.text, '14일');
  });
  test('실제 안내문은 21일이므로 서로 다르다', () => {
    includes(LIBRARY_DOC, '대출 기간은 21일');
  });
  test('모든 예상 답이 「틀림」으로 표시되어 있다', () => {
    ok(NAIVE_GUESSES.every((g) => g.wrong), '환각 예시는 모두 틀린 것이어야 한다');
  });
});

describe('프롬프트 조립', () => {
  test('규칙을 켜면 프롬프트에 들어간다', () => {
    const r = retrieve(index, '대출', { k: 1, mode: 'hybrid' });
    const on = buildPrompt('대출 기간은?', r.hits, { only: true, refuse: true });
    includes(on, '컨텍스트');
    includes(on, '추측하지 말고');
    const off = buildPrompt('대출 기간은?', r.hits, {});
    ok(!off.includes('추측하지 말고'), '끄면 안 들어가야 한다');
  });
  test('컨텍스트와 질문 자리가 모두 있다', () => {
    const r = retrieve(index, '대출', { k: 2, mode: 'hybrid' });
    const p = buildPrompt('며칠인가요?', r.hits, { role: true });
    includes(p, '[컨텍스트]');
    includes(p, '[질문]');
    includes(p, '며칠인가요?');
  });
  test('프롬프트 조각이 모두 이름과 문장을 갖는다', () => {
    for (const [k, v] of Object.entries(PROMPT_PARTS)) {
      ok(v.label && v.text, `${k} 조각에 label 또는 text 가 없습니다`);
    }
  });
});

describe('자료 무결성', () => {
  test('예시 질문의 꼬리표가 사실과 맞는다', () => {
    for (const q of SAMPLE_QUESTIONS) {
      const r = retrieve(index, q.q, { k: 3, mode: 'hybrid' });
      const a = answerFromContext(q.q, r.hits, { strict: true });
      if (q.tag === '문서에 없음') ok(!a.supported, `${q.q} 는 답이 없어야 합니다`);
      else ok(a.supported, `${q.q} 는 답이 있어야 합니다`);
    }
  });
  test('정답 세트에 answers 가 비어 있지 않다', () => {
    ok(DOC_FACTS.every((f) => f.accept && f.accept.length), 'accept 가 필요합니다');
  });
  test('정답 세트의 accept 문구가 안내문에 실제로 있다', () => {
    const missing = DOC_FACTS.filter((f) => !f.accept.some((a) => LIBRARY_DOC.includes(a)));
    eq(missing.map((f) => f.key), [], '안내문에 없는 값을 정답으로 두면 안 된다');
  });
  test('두 번째 안내문도 제목 구조로 잘린다', () => {
    const s2 = splitByHeadings(STORE_DOC);
    ok(s2.length >= 3, '섹션이 세 개 이상');
  });
});
