/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * python.test.mjs — 실습실에 실린 코드가 정말 돌아가는지 확인한다.
 *
 * 화면에 「기대 출력」을 적어 두고 실제로는 다르게 도는 것이 가장 나쁘다.
 * 그래서 예제와 빈칸 문제를 매번 실행해 본다.
 * ========================================================================== */

import { describe, test, eq, ok, includes } from './harness.mjs';
import { runPython } from '../src/lib/pymini.js';
import { EXAMPLES, FILL_PROBLEMS } from '../src/data/snippets.js';

const out = (src, opts) => {
  const r = runPython(src, opts);
  return { ...r, lines: (Array.isArray(r.output) ? r.output : String(r.output || '').split('\n')).map((s) => s.trimEnd()) };
};

describe('미니 파이썬 — 기본 문법', () => {
  test('print 와 f-문자열', () => {
    const r = out('이름 = "지민"\nprint(f"{이름} 님 안녕하세요")');
    eq(r.lines[0], '지민 님 안녕하세요');
  });
  test('리스트 컴프리헨션 (조건 포함)', () => {
    const r = out('낱말 = ["가", "나다", "라마바"]\nprint([w for w in 낱말 if len(w) >= 2])');
    eq(r.lines[0], "['나다', '라마바']");
  });
  test('sorted / set / enumerate / zip', () => {
    const r = out([
      'print(sorted(set(["나", "가", "나"])))',
      'print(list(enumerate(["a", "b"], 1)))',
      'print(list(zip([1, 2], ["가", "나"])))',
      'print(sorted([3, 1, 2], reverse=True))',
    ].join('\n'));
    eq(r.lines[0], "['가', '나']");
    eq(r.lines[1], "[(1, 'a'), (2, 'b')]");
    eq(r.lines[2], "[(1, '가'), (2, '나')]");
    eq(r.lines[3], '[3, 2, 1]');
  });
  test('any / all / reversed', () => {
    const r = out('print(any([False, True]), all([True, False]), list(reversed([1,2,3])))');
    eq(r.lines[0], 'True False [3, 2, 1]');
  });
  test('딕셔너리를 그대로 찍어도 열쇠가 보인다', () => {
    const r = out('print({"가": 1, "나": 2})');
    eq(r.lines[0], "{'가': 1, '나': 2}");
  });
  test('날 문자열 r\'…\' 이 역슬래시를 살린다', () => {
    const r = out("import re\nprint(re.sub(r'\\d+', 'N', 'a1b22c'))");
    eq(r.lines[0], 'aNbNc');
  });
});

describe('미니 파이썬 — 수업에 쓰는 모듈', () => {
  test('re.sub / findall / split', () => {
    const r = out([
      'import re',
      "print(re.sub(r'[^가-힣 ]', '', '노트 좋아요!!! 123'))",
      "print(re.findall(r'[0-9]+', 'a1b22c333'))",
      "print(re.split(r'\\s+', '가 나  다'))",
    ].join('\n'));
    eq(r.lines[0], '노트 좋아요'); // 시험 도구가 줄 끝 공백을 다듬는다
    eq(r.lines[1], "['1', '22', '333']");
    eq(r.lines[2], "['가', '나', '다']");
  });
  test('collections.Counter 와 most_common', () => {
    const r = out([
      'from collections import Counter',
      'c = Counter(["가", "가", "나"])',
      'print(c["가"], c["나"])',
      'print(c.most_common(1))',
    ].join('\n'));
    eq(r.lines[0], '2 1');
    eq(r.lines[1], "[('가', 2)]");
  });
  test('math.log 로 IDF 를 계산할 수 있다', () => {
    const r = out([
      'import math',
      '문서들 = [["노트","필기"], ["볼펜","필기"], ["볼펜","배송"], ["필기","배송"], ["볼펜","좋다"]]',
      '등장 = sum([1 for 문서 in 문서들 if "노트" in 문서])',
      'print(round(math.log(len(문서들) / (등장 + 1)), 4))',
    ].join('\n'));
    eq(r.lines[0], '0.9163');
  });
  test('korean 모듈이 붙어 있다', () => {
    const r = out('import korean\nprint(korean.nouns("도서관에서 책을 빌렸어요"))');
    ok(r.ok, r.error);
    includes(r.lines[0], '도서관');
  });
  test('진짜 라이브러리는 친절하게 막는다', () => {
    const r = out('import sklearn');
    ok(!r.ok, '막혀야 한다');
    includes(r.error, '직접 만들어');
  });
});

describe('미니 파이썬 — 오류 안내', () => {
  test('없는 이름을 쓰면 NameError', () => {
    const r = out('print(없는변수)');
    ok(!r.ok);
    includes(r.error, 'NameError');
  });
  test('0으로 나누면 알려 준다', () => {
    const r = out('print(1 / 0)');
    ok(!r.ok);
    includes(r.error, 'ZeroDivision');
  });
  test('무한 반복은 스스로 멈춘다', () => {
    const r = out('while True:\n    x = 1', { maxSteps: 20000 });
    ok(!r.ok, '멈춰야 한다');
    includes(r.error, '오래');
  });
});

describe('실습실 예제', () => {
  for (const ex of EXAMPLES) {
    test(`「${ex.title}」 가 오류 없이 돈다`, () => {
      const r = out(ex.code);
      ok(r.ok, `실행 오류: ${r.error}`);
    });
    if (ex.expect) {
      test(`「${ex.title}」 의 출력이 적어 둔 것과 같다`, () => {
        const r = out(ex.code);
        ok(r.ok, `실행 오류: ${r.error}`);
        ex.expect.forEach((want, i) => {
          eq(r.lines[i], want, `${i + 1}번째 줄이 다릅니다.`);
        });
      });
    }
  }
});

describe('빈칸 채우기 문제', () => {
  for (const p of FILL_PROBLEMS) {
    test(`「${p.title}」 — 정답을 넣으면 오류 없이 돈다`, () => {
      let src = p.code;
      for (const bl of p.blanks) src = src.replace(new RegExp(`⬚${bl.n}`, 'g'), bl.answers[0]);
      ok(!src.includes('⬚'), '채우지 못한 빈칸이 남았습니다');
      const r = out(src);
      ok(r.ok, `실행 오류: ${r.error}`);
    });
    test(`「${p.title}」 — 주석에 적어 둔 기대 결과가 실제와 같다`, () => {
      let src = p.code;
      for (const bl of p.blanks) src = src.replace(new RegExp(`⬚${bl.n}`, 'g'), bl.answers[0]);
      const r = out(src);
      // 코드 줄 끝에 달아 둔 「… # 기대값」 만 본다 (설명용 전체 주석 줄은 건너뛴다)
      const wants = [...p.code.matchAll(/\S[ \t]+#\s*([\[0-9'"][^\n]*)$/gm)].map((m) => m[1].trim());
      if (!wants.length) return;
      const joined = r.lines.join('\n');
      for (const w of wants) includes(joined, w, '주석에 적은 기대 결과가 실제와 다릅니다.');
    });
    test(`「${p.title}」 — 빈칸마다 답과 해설이 있다`, () => {
      ok(p.blanks.length > 0, '빈칸이 하나 이상 있어야 한다');
      ok(p.blanks.every((b) => b.answers.length && b.why), '답과 해설이 모두 필요합니다');
      for (const bl of p.blanks) ok(p.code.includes(`⬚${bl.n}`), `코드에 ⬚${bl.n} 이 없습니다`);
    });
  }
});
