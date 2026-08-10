/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * embed.js — 교실용 임베딩 (뜻이 가까우면 숫자도 가깝다)
 *
 * ⚠️ 먼저 정직하게 밝힐 것
 *   이 앱은 인터넷이 끊긴 교실에서도 열려야 한다. 그래서 진짜 임베딩 모델
 *   (수억 개 문장으로 학습한 768차원짜리)을 넣을 수 없다.
 *   대신 "뜻이 비슷한 낱말을 미리 스무 갈래로 묶어 둔 뜻 지도"를 만들어,
 *   낱말마다 각 갈래에 얼마나 걸쳐 있는지를 숫자로 적었다.
 *
 *   원리는 진짜와 같다 — 뜻이 가까우면 벡터가 가깝고, 코사인 유사도가 높다.
 *   다른 점은 그 숫자를 「학습으로 얻었는가」 「사람이 적었는가」뿐이다.
 *   화면에도 이 사실을 적어 두었다.
 *
 * 갈래(축)를 스무 개 두고, 여기에 극성·구체성·행위성·시간성 네 개를 더해
 * 모두 24차원으로 만든다.
 * ========================================================================== */

/* 각 축은 [축이름, 그 축에 속하는 낱말들] 이다. */
const AXES = [
  ['동물', '강아지 개 고양이 반려견 반려동물 새 물고기 토끼 햄스터 사료 꼬리 짖다 야옹'],
  ['음식', '김밥 라면 피자 치킨 샌드위치 빵 밥 음료 커피 라떼 차 케이크 쿠키 디저트 간식 맛 맛있다 맛없다 먹다 배고프다 식사 메뉴'],
  ['책·도서관', '도서관 책 도서 대출 반납 연장 연체 정지 열람실 자료실 서가 사서 신간 전자책 독서 빌리다 빌려 정기간행물 잡지 회원증 회원 예약 좌석 사물함'],
  ['학교·공부', '학교 학생 수업 시험 과제 공부 성적 선생님 교사 교실 숙제 발표 모둠 조별 학년 방과후 동아리'],
  ['컴퓨터·코드', '컴퓨터 노트북 코드 파이썬 프로그램 실행 오류 에러 함수 변수 리스트 개발 설치 라이브러리 주석'],
  ['인공지능', '인공지능 모델 학습 훈련 신경망 임베딩 벡터 토큰 트랜스포머 어텐션 언어모델 챗봇 생성 예측 파라미터 가중치'],
  ['검색·문서', '검색 질문 답변 문서 자료 정보 찾다 출처 근거 색인 청크 조각 검색기 관련 유사'],
  ['이동·운동', '달리다 뛰다 걷다 산책 운동 공원 야외 운동장 자전거 놀다 뛰어놀다 움직이다'],
  ['시간', '오늘 어제 내일 아침 점심 저녁 시간 기간 하루 이틀 주말 평일 언제 기한 마감 며칠 일 날 요일 토요일 일요일 오전 오후 시 분 동안 이내'],
  ['좋은 느낌', '좋다 훌륭하다 만족 최고 추천 친절 편하다 즐겁다 기쁘다 깨끗하다 쾌적 넉넉하다 뿌듯'],
  ['나쁜 느낌', '나쁘다 실망 불만 별로 아쉽다 불편하다 시끄럽다 더럽다 불친절 답답하다 부족하다 짜증'],
  ['돈·구매', '가격 비용 결제 환불 교환 주문 배송 택배 할인 무료 유료 영수증 구매 판매 요금 수수료 연체료 적립 취소'],
  ['장소·시설', '카페 매장 문구점 건물 자리 공간 시설 화장실 주차 주차장 콘센트 창가 층 안내데스크 무선 인터넷'],
  ['규정·안내', '규정 정책 안내 이용 기준 조건 제한 신청 방법 절차 원칙 유의 주의 필수'],
  ['수·계산', '숫자 계산 평균 합계 비율 값 점수 확률 통계 수식 더하기 곱하기 나누기 크기 몇 권 번 최대 최소 얼마'],
  ['글·언어', '글 문장 낱말 단어 텍스트 표현 의미 문맥 맥락 번역 요약 문단 어휘 국어'],
  ['사람·관계', '사람 친구 가족 이름 회원 담당자 사용자 고객 직원 사장님 우리 나 너'],
  ['정도·크기', '크다 작다 많다 적다 넓다 좁다 길다 짧다 빠르다 느리다 높다 낮다 무겁다 가볍다'],
  ['문제·해결', '문제 오류 고장 해결 원인 확인 수리 대처 실패 성공 개선 보완'],
  ['날씨·자연', '날씨 비 눈 바람 햇빛 하늘 나무 꽃 자연 계절 봄 여름 가을 겨울'],
];

/** 낱말마다 값이 다른 네 가지 성질 (축 20개 뒤에 붙는다) */
const POLARITY_POS = new Set('좋다 훌륭하다 만족 최고 추천 친절 편하다 즐겁다 기쁘다 깨끗하다 쾌적 넉넉하다 뿌듯 성공 개선 무료 빠르다 넓다'.split(' '));
const POLARITY_NEG = new Set('나쁘다 실망 불만 별로 아쉽다 불편하다 시끄럽다 더럽다 불친절 답답하다 부족하다 짜증 오류 고장 실패 느리다 좁다 비싸다'.split(' '));
const ACTIONY = new Set('달리다 뛰다 걷다 놀다 뛰어놀다 먹다 빌리다 찾다 신청 주문 결제 반납 교환 환불 실행 학습 훈련 검색 계산 요약 번역'.split(' '));
const TIMEY = new Set('오늘 어제 내일 아침 점심 저녁 시간 기간 하루 이틀 주말 평일 기한 마감 계절 봄 여름 가을 겨울'.split(' '));

export const DIM = AXES.length + 4;
export const AXIS_NAMES = [...AXES.map((a) => a[0]), '느낌(+/−)', '행위성', '시간성', '구체성'];

/* 낱말 → 축 번호 목록 */
const WORD_AXES = new Map();
AXES.forEach(([, words], ai) => {
  words.split(/\s+/).forEach((w) => {
    if (!WORD_AXES.has(w)) WORD_AXES.set(w, []);
    WORD_AXES.get(w).push(ai);
  });
});

export const KNOWN_WORDS = [...WORD_AXES.keys()];

/** 문자열을 늘 같은 숫자로 바꾸는 해시 (모르는 낱말도 매번 같은 벡터가 되게) */
function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function noiseVec(word, scale = 0.18) {
  const v = new Array(DIM).fill(0);
  let s = hash(word) || 1;
  for (let i = 0; i < DIM; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    v[i] = ((s / 4294967296) * 2 - 1) * scale;
  }
  return v;
}

const cache = new Map();

/** 낱말 하나의 벡터 (길이 1로 맞춰 돌려준다) */
export function wordVector(wordRaw) {
  const word = String(wordRaw).trim();
  if (!word) return new Array(DIM).fill(0);
  if (cache.has(word)) return cache.get(word);

  const v = noiseVec(word);
  const axes = WORD_AXES.get(word);
  if (axes) {
    axes.forEach((ai) => { v[ai] += 1; });
  } else {
    // 모르는 낱말이라도 아는 낱말을 품고 있으면 그 뜻을 얼마간 물려받는다
    // (예: 「도서관에서」 같은 덩어리, 「반려동물용품」 같은 합성어)
    for (const [kw, ax] of WORD_AXES) {
      if (kw.length >= 2 && word.includes(kw)) ax.forEach((ai) => { v[ai] += 0.6; });
    }
  }
  if (POLARITY_POS.has(word)) v[AXES.length] += 1;
  if (POLARITY_NEG.has(word)) v[AXES.length] -= 1;
  if (ACTIONY.has(word)) v[AXES.length + 1] += 0.8;
  if (TIMEY.has(word)) v[AXES.length + 2] += 0.8;
  v[AXES.length + 3] += axes ? 0.5 : 0.1; // 사전에 있는 구체적인 낱말인가

  const n = Math.hypot(...v) || 1;
  const out = v.map((x) => x / n);
  cache.set(word, out);
  return out;
}

/** 문장 하나의 벡터 — 낱말 벡터의 평균을 길이 1로 맞춘다 */
export function embed(text) {
  const words = String(text)
    .replace(/[^가-힣a-zA-Z0-9 ]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
  if (!words.length) return new Array(DIM).fill(0);
  const sum = new Array(DIM).fill(0);
  for (const w of words) {
    const v = wordVector(w);
    for (let i = 0; i < DIM; i++) sum[i] += v[i];
  }
  const n = Math.hypot(...sum) || 1;
  return sum.map((x) => x / n);
}

export function embedAll(texts) {
  return texts.map(embed);
}

/** 어느 축이 세게 켜졌는지 (화면에서 「이 문장은 무슨 갈래인가」를 보여 줄 때) */
export function topAxes(vec, k = 4) {
  return vec
    .map((v, i) => ({ name: AXIS_NAMES[i], v, i }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, k);
}

/* ─────────────────────── 24차원을 2차원 지도로 (PCA) ────────────────────── */
/**
 * 주성분 두 개를 거듭제곱법으로 찾는다.
 * 사람은 24차원을 볼 수 없으니, 가장 많이 흩어지는 두 방향만 남겨 평면에 그린다.
 */
export function pca2(vectors) {
  const n = vectors.length;
  const d = vectors[0]?.length || 0;
  if (!n || !d) return { points: [], axes: [] };
  const mean = new Array(d).fill(0);
  vectors.forEach((v) => { for (let i = 0; i < d; i++) mean[i] += v[i] / n; });
  const X = vectors.map((v) => v.map((x, i) => x - mean[i]));

  const mulCov = (w) => {
    // (XᵀX)w 를 X 를 두 번 훑어 계산한다 (공분산 행렬을 직접 만들지 않는다)
    const t = new Array(n).fill(0);
    for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < d; i++) s += X[r][i] * w[i]; t[r] = s; }
    const out = new Array(d).fill(0);
    for (let r = 0; r < n; r++) { const tr = t[r]; for (let i = 0; i < d; i++) out[i] += X[r][i] * tr; }
    return out;
  };
  const power = (deflate) => {
    let w = new Array(d).fill(0).map((_, i) => Math.sin(i * 12.9898) * 43758.5453 % 1);
    let nw = Math.hypot(...w) || 1;
    w = w.map((x) => x / nw);
    for (let it = 0; it < 80; it++) {
      let y = mulCov(w);
      if (deflate) {
        const p = y.reduce((s, x, i) => s + x * deflate[i], 0);
        y = y.map((x, i) => x - p * deflate[i]);
      }
      nw = Math.hypot(...y);
      if (nw < 1e-12) break;
      w = y.map((x) => x / nw);
    }
    return w;
  };
  const a1 = power(null);
  const a2 = power(a1);
  const proj = (v) => [
    v.reduce((s, x, i) => s + (x - mean[i]) * a1[i], 0),
    v.reduce((s, x, i) => s + (x - mean[i]) * a2[i], 0),
  ];
  const points = vectors.map(proj);
  // 보기 좋게 −1 ~ 1 범위로 늘린다
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const sc = (arr) => {
    const lo = Math.min(...arr);
    const hi = Math.max(...arr);
    const r = hi - lo || 1;
    return (v) => ((v - lo) / r) * 1.7 - 0.85;
  };
  const fx = sc(xs);
  const fy = sc(ys);
  return { points: points.map((p) => [fx(p[0]), fy(p[1])]), axes: [a1, a2], project: (v) => { const p = proj(v); return [fx(p[0]), fy(p[1])]; } };
}
