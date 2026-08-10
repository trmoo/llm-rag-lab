/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * nlp.test.mjs — 화면에 적어 둔 수치와 계산이 어긋나지 않는지 지킨다.
 *
 * ⚠️ 여기 있는 값을 고칠 때는 반드시 화면 문구도 함께 고칠 것.
 *    (학생이 보는 숫자와 계산 결과가 다르면 안 된다)
 * ========================================================================== */

import { describe, test, eq, near, ok, includes } from './harness.mjs';
import {
  clean, squeezeRepeat, preprocess, counter, vocabulary, bowMatrix,
  tf, idf, docFreq, tfidfMatrix, topTerms, cosine, euclid, similarityMatrix,
  bm25Search, mmrSelect, charNgrams, trainTestSplit, makeVectorizer,
  naiveBayes, logisticRegression, metrics, confusion, thresholdTable, rng,
} from '../src/lib/nlp.js';
import { pos, nouns, morphs, josa } from '../src/lib/korean.js';
import { TOY_DOCS, POSITIVE_REVIEWS, NEGATIVE_REVIEWS, W2V_SENTENCES } from '../src/data/corpus.js';

const docs = TOY_DOCS.map((d) => d.words);

describe('전처리', () => {
  test('주소·특수문자를 지우고 반복 글자를 줄인다', () => {
    // ㅋㅋㅋ 은 「가-힣」 범위 밖이라 특수문자 규칙에서 통째로 지워진다
    const r = clean('ㅋㅋㅋ 노트 진짜아아아 좋아요!!!! https://a.test/1 <b>강추</b>');
    eq(squeezeRepeat(r.text), '노트 진짜아아 좋아요 강추');
  });
  test('규칙을 끄면 그 규칙은 적용되지 않는다', () => {
    const r = clean('노트 좋아요!!!', { special: false });
    includes(r.text, '!!!');
  });
  test('반복 글자는 세 번 이상일 때만 두 번으로 줄인다', () => {
    eq(squeezeRepeat('ㅋㅋ'), 'ㅋㅋ');
    eq(squeezeRepeat('ㅋㅋㅋㅋㅋ'), 'ㅋㅋ');
  });
  test('단계별 기록이 규칙 수만큼 남는다', () => {
    const r = clean('테스트 https://a.test');
    ok(r.steps.length >= 5, '정제 단계 기록이 남아야 한다');
  });
});

describe('한국어 분석기', () => {
  test('조사를 바르게 떼어 낸다', () => {
    eq(pos('도서관에서').map((x) => x.t), ['도서관', '에서']);
    eq(pos('학교를').map((x) => x.t), ['학교', '를']);
  });
  test('「종이가」를 「종 + 이 + 가」로 잘못 쪼개지 않는다', () => {
    eq(pos('종이가').map((x) => x.t), ['종이', '가']);
  });
  test('「명사 + 하다」 꼴을 분리한다 — 검색이 여기서 자주 깨진다', () => {
    const t = pos('연체하면').map((x) => x.t);
    eq(t[0], '연체');
    eq(nouns('연체하면 대출이 정지됩니다').includes('연체'), true);
  });
  test('명사만 뽑는다', () => {
    const n = nouns('도서관에서 빌린 책을 오늘 반납했어요');
    ok(n.includes('도서관'), '도서관이 있어야 한다');
    ok(n.includes('책'), '책이 있어야 한다');
    ok(!n.includes('에서'), '조사는 명사가 아니다');
  });
  test('원형으로 되돌리기', () => {
    eq(morphs('맛있었어요', { stem: true }), ['맛있다']);
    eq(morphs('반납했어요', { stem: true })[1], '반납하다');
  });
  test('조사를 받침에 맞게 붙인다', () => {
    eq(josa('토큰', '이'), '토큰이');
    eq(josa('벡터', '이'), '벡터가');
    eq(josa('노트', '은'), '노트는');
    eq(josa('필기', '은'), '필기는');
    eq(josa('책', '을'), '책을');
    eq(josa('도서', '을'), '도서를');
  });
});

describe('단어 주머니', () => {
  test('어휘는 일곱 개 (가나다순)', () => {
    eq(vocabulary(docs), ['노트', '배송', '볼펜', '빠르다', '좋다', '편하다', '필기']);
  });
  test('문서1 벡터', () => {
    const { matrix } = bowMatrix(docs);
    eq(matrix[0], [2, 0, 0, 0, 1, 0, 1]);
  });
});

describe('TF-IDF (화면에 적힌 값과 같아야 한다)', () => {
  test('TF(노트, 문서1) = 0.5', () => near(tf('노트', docs[0]), 0.5));
  test('df(배송) = 4, df(노트) = 1', () => {
    eq(docFreq('배송', docs), 4);
    eq(docFreq('노트', docs), 1);
  });
  test('IDF(노트) = ln(5/2) = 0.9163', () => near(idf('노트', docs, 'plain'), 0.9163, 1e-4));
  test('IDF(배송) = 0 — 거의 모든 문서에 나오면 쓸모가 없어진다', () => {
    near(idf('배송', docs, 'plain'), 0, 1e-9);
  });
  test('TF-IDF(노트, 문서1) = 0.4581', () => {
    const { matrix, vocab } = tfidfMatrix(docs, { mode: 'plain', normalize: false });
    near(matrix[0][vocab.indexOf('노트')], 0.4581, 1e-4);
  });
  test('문서1의 핵심어는 「노트」', () => {
    const { matrix, vocab } = tfidfMatrix(docs, { mode: 'plain', normalize: false });
    eq(topTerms(matrix[0], vocab, 1)[0].word, '노트');
  });
  test('실제 도구식 IDF 는 0 이 되지 않는다 (+1 덕분)', () => {
    ok(idf('배송', docs, 'sklearn') > 0.9, '흔한 낱말도 최소 1 쯤은 남는다');
  });
  test('길이를 1로 맞추면 벡터 크기가 1', () => {
    const { matrix } = tfidfMatrix(docs, { mode: 'sklearn', normalize: true });
    near(Math.hypot(...matrix[0]), 1, 1e-9);
  });
});

describe('닮은 정도', () => {
  test('방향이 같으면 1', () => near(cosine([1, 1], [3, 3]), 1, 1e-9));
  test('직각이면 0', () => near(cosine([1, 1], [1, -1]), 0, 1e-9));
  test('길이가 달라도 코사인은 같고 거리는 다르다', () => {
    near(cosine([1, 1, 0], [3, 3, 0]), 1, 1e-9);
    ok(euclid([1, 1, 0], [3, 3, 0]) > 2.8, '거리는 멀게 나온다');
  });
  test('자기 자신과의 유사도는 1', () => {
    const { matrix } = tfidfMatrix(docs, { mode: 'sklearn', normalize: true });
    const s = similarityMatrix(matrix);
    for (let i = 0; i < s.length; i++) near(s[i][i], 1, 1e-9);
  });
});

describe('검색', () => {
  test('BM25 는 낱말이 겹치는 문서를 위로 올린다', () => {
    const r = bm25Search(['노트', '필기'], docs, { topK: 2 });
    eq(r[0].i, 0);
  });
  test('겹치는 낱말이 없으면 0점', () => {
    const r = bm25Search(['우주선'], docs, { topK: 1 });
    near(r[0].score, 0, 1e-9);
  });
  test('MMR 은 서로 다른 것을 고른다', () => {
    // 0·1 은 거의 같고, 2 는 다르다
    const sim = [0.9, 0.88, 0.6];
    const pair = [[1, 0.98, 0.1], [0.98, 1, 0.1], [0.1, 0.1, 1]];
    const plain = sim.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, i]) => i);
    eq(plain, [0, 1], '보통 검색은 비슷한 둘을 고른다');
    const { picked } = mmrSelect(sim, pair, { k: 2, lambda: 0.5 });
    eq(picked, [0, 2], 'MMR 은 겹치지 않는 것을 고른다');
  });
  test('lambda 를 1로 두면 보통 검색과 같아진다', () => {
    const sim = [0.9, 0.88, 0.6];
    const pair = [[1, 0.98, 0.1], [0.98, 1, 0.1], [0.1, 0.1, 1]];
    const { picked } = mmrSelect(sim, pair, { k: 2, lambda: 1 });
    eq(picked, [0, 1]);
  });
});

describe('글자 n-gram', () => {
  test('2~4글자 묶음을 만든다', () => {
    eq(charNgrams('바삭하고', 2, 2), ['바삭', '삭하', '하고']);
  });
  test('꼴이 달라도 공통 조각이 남는다 — 한국어에 통하는 이유', () => {
    const a = new Set(charNgrams('맛있다', 2, 3));
    const b = new Set(charNgrams('맛있어요', 2, 3));
    ok([...a].some((x) => b.has(x)), '「맛있」 같은 공통 조각이 있어야 한다');
  });
});

describe('분류기', () => {
  const X = [...POSITIVE_REVIEWS, ...NEGATIVE_REVIEWS];
  const y = [...POSITIVE_REVIEWS.map(() => 1), ...NEGATIVE_REVIEWS.map(() => 0)];

  test('자료가 긍정·부정 같은 수로 고르게 되어 있다', () => {
    eq(POSITIVE_REVIEWS.length, NEGATIVE_REVIEWS.length);
    ok(POSITIVE_REVIEWS.length >= 40, '한쪽에 40개 이상');
  });
  test('나누기는 씨앗이 같으면 늘 같은 결과', () => {
    const a = trainTestSplit(X, y, { seed: 42 });
    const b = trainTestSplit(X, y, { seed: 42 });
    eq(a.testIdx, b.testIdx);
  });
  test('훈련과 시험이 겹치지 않는다', () => {
    const s = trainTestSplit(X, y, { seed: 7 });
    ok(!s.trainIdx.some((i) => s.testIdx.includes(i)), '겹치면 안 된다');
    eq(s.trainIdx.length + s.testIdx.length, X.length);
  });
  test('여러 번 나눠 평균을 내면 65% 이상은 나온다', () => {
    let acc = 0;
    const seeds = [1, 3, 7, 11, 42, 99, 2026];
    for (const seed of seeds) {
      const s = trainTestSplit(X, y, { testSize: 0.25, seed });
      const vec = makeVectorizer({ min: 2, max: 4, minDf: 3 }).fit(s.Xtrain);
      const nb = naiveBayes(vec.counts(s.Xtrain), s.ytrain);
      acc += metrics(s.ytest, nb.predict(vec.counts(s.Xtest))).accuracy;
    }
    ok(acc / seeds.length >= 0.65, `평균 정확도가 낮습니다: ${(acc / seeds.length).toFixed(3)}`);
  });
  test('로지스틱 회귀가 감정 낱말을 신호로 배운다', () => {
    const vec = makeVectorizer({ min: 2, max: 4, minDf: 3 }).fit(X);
    const lr = logisticRegression(vec.transform(X), y, { epochs: 400, lr: 3, l2: 0.004 });
    const ranked = lr.weights.map((w, i) => ({ w, g: vec.vocab[i] })).sort((a, b) => b.w - a.w);
    const top = ranked.slice(0, 12).map((r) => r.g).join(' ');
    const bottom = ranked.slice(-12).map((r) => r.g).join(' ');
    ok(/만족|친절|훌륭|좋았|편했|추천/.test(top), `긍정 신호가 위에 없습니다: ${top}`);
    ok(/별로|실망|불편|답답|아쉽/.test(bottom), `부정 신호가 아래에 없습니다: ${bottom}`);
  });
  test('학습한 자료를 다시 넣으면 잘 맞힌다', () => {
    const vec = makeVectorizer({ min: 2, max: 4, minDf: 2 }).fit(X);
    const lr = logisticRegression(vec.transform(X), y, { epochs: 400, lr: 3, l2: 0.004 });
    const m = metrics(y, lr.predict(vec.transform(X)));
    ok(m.accuracy > 0.9, `훈련 정확도가 낮습니다: ${m.accuracy}`);
  });
  test('혼동 행렬과 지표 계산', () => {
    const c = confusion([1, 1, 0, 0], [1, 0, 0, 1]);
    eq(c, { tp: 1, tn: 1, fp: 1, fn: 1 });
    const m = metrics([1, 1, 0, 0], [1, 0, 0, 1]);
    near(m.accuracy, 0.5);
    near(m.precision, 0.5);
    near(m.recall, 0.5);
    near(m.f1, 0.5);
  });
  test('정확도의 함정 — 전부 긍정이라 답해도 95%', () => {
    const yy = [...Array(95).fill(1), ...Array(5).fill(0)];
    const pred = yy.map(() => 1);
    near(metrics(yy, pred).accuracy, 0.95, 1e-9);
    near(metrics(yy, pred).recall, 1, 1e-9);
  });
  test('기준선을 올리면 정밀도가 오르고 재현율이 내린다', () => {
    const yy = [1, 1, 1, 0, 0, 0];
    const p = [0.9, 0.6, 0.45, 0.4, 0.2, 0.1];
    const rows = thresholdTable(yy, p, 9);
    const low = rows[0];
    const high = rows[rows.length - 1];
    ok(high.recall <= low.recall, '기준을 올리면 재현율은 내려가야 한다');
  });
});

describe('난수', () => {
  test('같은 씨앗이면 같은 값', () => {
    const a = rng(42);
    const b = rng(42);
    eq([a(), a(), a()], [b(), b(), b()]);
  });
  test('0 이상 1 미만', () => {
    const r = rng(1);
    for (let i = 0; i < 200; i++) {
      const v = r();
      ok(v >= 0 && v < 1, `범위를 벗어남: ${v}`);
    }
  });
});

describe('예제 자료', () => {
  test('Word2Vec 문장은 모두 두 낱말 이상', () => {
    ok(W2V_SENTENCES.every((s) => s.length >= 2), '문장이 너무 짧으면 학습이 안 된다');
  });
  test('전처리를 거쳐도 낱말이 남는다', () => {
    for (const r of POSITIVE_REVIEWS.slice(0, 10)) {
      ok(preprocess(r).length >= 1, `전처리 후 비었습니다: ${r}`);
    }
  });
  test('빈도 세기', () => {
    const c = counter(['가', '가', '나']);
    eq(c.get('가'), 2);
    eq(c.get('나'), 1);
  });
});
