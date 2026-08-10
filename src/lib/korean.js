/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * korean.js — 교실용 간이 한국어 형태소 분석기 + 조사 붙이기
 *
 * 왜 직접 만들었나?
 *   진짜 형태소 분석기(자바 기반)는 브라우저에서 돌지 않는다.
 *   그렇다고 "한국어는 조사가 붙어서 어렵다"를 말로만 하고 넘어가면
 *   학생이 그 어려움을 직접 볼 수가 없다. 그래서 교실에서 보여 줄 만큼만
 *   하는 작은 분석기를 사전 + 규칙으로 만들었다.
 *
 * ⚠️ 정직하게 밝혀 둘 것 — 이 분석기는 사전에 있는 낱말만 제대로 쪼갠다.
 *    실제 KoNLPy(Okt·Kkma 등)와 결과가 다를 수 있고, 화면에도 그렇게 적어 두었다.
 *    학습 목표는 "정확한 분석기 만들기"가 아니라 "왜 쪼개야 하는가"를 보는 것이다.
 *
 * 쪼개는 방법 (어절 하나마다)
 *   ① 뒤쪽에서 조사 덩어리를 떼어 본다 (에서도 → 에서 + 도)
 *   ② 남은 앞부분이 명사 사전에 있으면 명사로 확정
 *   ③ 아니면 용언(동사·형용사) 어간으로 시작하는지 본다 → 어간 + 어미
 *   ④ 그래도 안 되면 두 낱말이 붙은 복합명사인지 본다 (도서관+이용)
 *   ⑤ 다 실패하면 통째로 명사로 추정한다 (모르는 낱말)
 * ========================================================================== */

/* ─────────────────────────── ① 조사 (뒤에 붙는 것) ───────────────────────── */

// 긴 것부터 찾아야 '에서' 가 '에' + '서' 로 잘못 쪼개지지 않는다.
export const JOSA = [
  '에게서', '으로서', '으로써', '이라고', '에서는', '에서도', '에게는', '으로는', '까지는',
  '라고', '부터', '까지', '에서', '에게', '한테', '께서', '으로', '처럼', '보다', '마다',
  '조차', '밖에', '이나', '라도', '이랑', '하고', '이야', '이며', '이란', '이든',
  '은', '는', '이', '가', '을', '를', '의', '에', '도', '만', '과', '와', '로', '랑', '야', '나', '든',
];

/**
 * 어절 끝에서 조사를 떼어 낸 여러 갈래를 만든다 (최대 두 겹).
 * 앞쪽일수록 조금 떼어 낸 것이라, 명사 사전에 걸리는 첫 갈래를 고르면
 * 「종이 + 가」를 「종 + 이 + 가」로 잘못 쪼개는 일이 없다.
 */
function josaCandidates(word) {
  const out = [{ stem: word, josa: [] }];
  const acc = [];
  let rest = word;
  for (let depth = 0; depth < 2; depth++) {
    let hit = null;
    for (const j of JOSA) {
      if (rest.length > j.length && rest.endsWith(j)) { hit = j; break; }
    }
    if (!hit) break;
    acc.unshift(hit);
    rest = rest.slice(0, -hit.length);
    out.push({ stem: rest, josa: [...acc] });
  }
  return out;
}

/* ────────────────────────── ② 명사 사전 ─────────────────────────────────── */
/* 이 앱의 예제 글에 나오는 낱말 + 수업에서 자주 쓰는 일반 낱말을 모았다.
   사전에 없는 낱말도 ⑤ 규칙으로 명사로 추정되니 앱이 멈추지는 않는다. */
export const NOUNS = new Set(`
도서관 자료실 열람실 정기간행물실 회원 회원증 대출 반납 연장 예약 연체 분실 훼손 이용 안내 규정 정책 기간 기한 권수 도서 책 전자책 신간 잡지 신문 좌석 사물함 복사기 프린터 무선 인터넷 공기 조명
학교 학생 선생님 교사 수업 시간표 과제 시험 성적 발표 조별 모둠 동아리 부원 활동 계획서 신청서 서류 담당 상담 진로 방과후 급식 식단 매점 운동장 강당 도서 실습실 컴퓨터실
문구점 문구 노트 필기 필기구 볼펜 형광펜 지우개 자 가위 풀 스티커 다이어리 파일 클립 포장 배송 교환 환불 주문 결제 영수증 재고 가격 할인 적립 쿠폰 회원가입 배달 택배 상자 종이 잉크 모서리 품질 색 뒷면
카페 음료 커피 라떼 차 디저트 케이크 쿠키 자리 좌석 콘센트 소음 사장님 직원 친절 청결 매장 주문 대기 진동벨
텍스트 데이터 분석 처리 전처리 정제 정규화 토큰 토큰화 형태소 품사 명사 동사 조사 어미 불용어 어간 표제어 정규표현식 패턴 문자 문자열 공백 특수문자 이모지 주소
단어 낱말 문장 문단 문서 어휘 사전 빈도 횟수 비율 가중치 점수 벡터 차원 행렬 표 열 행 값 숫자 계산 공식 로그 평균 합계 제곱근 내적 크기 방향 각도 유사도 거리 코사인
모형 모델 학습 훈련 검증 시험 예측 분류 분류기 정확도 정밀도 재현율 혼동 행렬 특성 레이블 정답 오답 과적합 누수 성능 결과 실험 비교 개선
임베딩 의미 맥락 문맥 관계 이웃 지도 공간 좌표 군집 묶음 주제 개념 축
신경망 층 은닉 입력 출력 뉴런 활성화 함수 손실 오차 경사 하강 역전파 기울기 학습률 에포크 배치 순전파 소실 게이트 기억 상태 셀
어텐션 질의 열쇠 값 헤드 위치 인코딩 블록 잔차 정규화 병렬 순차 마스크 생성 확률 분포 샘플링 온도 후보 다음
언어 모델 사전학습 파인튜닝 프롬프트 지시 역할 예시 형식 단계 사고 환각 지식 최신 비용 토큰수 한도 창 컨텍스트 세션 대화 이력 요약 기억 메시지 시스템 사용자 응답 무상태
검색 질문 답변 근거 출처 문단 조각 청크 겹침 크기 저장소 색인 인덱싱 증강 파이프라인 검색기 결합 하이브리드 다양성 재정렬 임계값 필터 꼬리표 메타데이터 범위
그래프 상태 노드 엣지 화살표 흐름 분기 갈림길 반복 루프 조건 라우터 감독자 담당 전문가 협업 도구 호출 실행 결과 대화방 번호 저장 장치
파이썬 코드 함수 변수 리스트 딕셔너리 반복문 조건문 주석 오류 에러 실행 출력 입력 라이브러리 설치 불러오기 객체 클래스 메서드
사람 이름 나이 학년 반 친구 가족 시간 오늘 어제 내일 오전 오후 아침 점심 저녁 하루 이틀 요일 주말 평일 이번 다음 지난
가격 비용 무료 유료 제한 조건 방법 이유 문제 해결 장점 단점 차이 특징 종류 예 예시 정리 요약 핵심 기본 심화 연습 실습 과제 목표 흐름 순서 단계
돈 값 요금 수수료 연체료 주차 주차장 비밀번호 층 서류 신분 자물쇠 물건 상태 담당자 누리집 예약기 반납기 통화 음료 뚜껑 공휴일 월요일 토요일 일요일 오전 오후 시 분 초 일 주 달 년 권 개 번 명
`.trim().split(/\s+/));

/* ────────────────────────── ③ 용언 사전 (동사·형용사) ────────────────────── */
/* [원형, 품사, 어간 목록]. 어간이 여러 개인 것은 불규칙 활용 때문이다. */
const VERBS = [
  ['하다', 'Verb', ['하', '해', '했', '한']],
  ['되다', 'Verb', ['되', '돼', '됐', '된']],
  ['있다', 'Adjective', ['있', '있었']],
  ['없다', 'Adjective', ['없', '없었']],
  ['같다', 'Adjective', ['같']],
  ['보다', 'Verb', ['보', '봤', '봐']],
  ['주다', 'Verb', ['주', '줬', '줘']],
  ['받다', 'Verb', ['받', '받았']],
  ['쓰다', 'Verb', ['쓰', '썼', '써']],
  ['읽다', 'Verb', ['읽', '읽었']],
  ['찾다', 'Verb', ['찾', '찾았']],
  ['넣다', 'Verb', ['넣', '넣었']],
  ['빼다', 'Verb', ['빼', '뺐']],
  ['나누다', 'Verb', ['나누', '나눠', '나눴']],
  ['자르다', 'Verb', ['자르', '잘라', '잘랐']],
  ['합치다', 'Verb', ['합치', '합쳐', '합쳤']],
  ['만들다', 'Verb', ['만들', '만든', '만듦']],
  ['바꾸다', 'Verb', ['바꾸', '바꿔', '바꿨', '바꾼']],
  ['알다', 'Verb', ['알', '아는', '앎']],
  ['모르다', 'Verb', ['모르', '몰라', '몰랐']],
  ['배우다', 'Verb', ['배우', '배워', '배웠']],
  ['가르치다', 'Verb', ['가르치', '가르쳐', '가르쳤']],
  ['생각하다', 'Verb', ['생각하', '생각해', '생각했']],
  ['이해하다', 'Verb', ['이해하', '이해해', '이해했']],
  ['설명하다', 'Verb', ['설명하', '설명해', '설명했']],
  ['사용하다', 'Verb', ['사용하', '사용해', '사용했']],
  ['비교하다', 'Verb', ['비교하', '비교해', '비교했']],
  ['확인하다', 'Verb', ['확인하', '확인해', '확인했']],
  ['빌리다', 'Verb', ['빌리', '빌려', '빌렸']],
  ['반납하다', 'Verb', ['반납하', '반납해', '반납했']],
  ['신청하다', 'Verb', ['신청하', '신청해', '신청했']],
  ['주문하다', 'Verb', ['주문하', '주문해', '주문했']],
  ['배송하다', 'Verb', ['배송하', '배송해', '배송했']],
  ['교환하다', 'Verb', ['교환하', '교환해', '교환했']],
  ['환불하다', 'Verb', ['환불하', '환불해', '환불했']],
  ['좋다', 'Adjective', ['좋', '좋았']],
  ['나쁘다', 'Adjective', ['나쁘', '나빠', '나빴']],
  ['맛있다', 'Adjective', ['맛있', '맛있었']],
  ['맛없다', 'Adjective', ['맛없', '맛없었']],
  ['친절하다', 'Adjective', ['친절하', '친절해', '친절했', '친절한']],
  ['불친절하다', 'Adjective', ['불친절하', '불친절해', '불친절했', '불친절한']],
  ['깨끗하다', 'Adjective', ['깨끗하', '깨끗해', '깨끗했', '깨끗한']],
  ['조용하다', 'Adjective', ['조용하', '조용해', '조용했', '조용한']],
  ['시끄럽다', 'Adjective', ['시끄럽', '시끄러워', '시끄러웠', '시끄러운']],
  ['넓다', 'Adjective', ['넓', '넓었']],
  ['좁다', 'Adjective', ['좁', '좁았']],
  ['빠르다', 'Adjective', ['빠르', '빨라', '빨랐', '빠른']],
  ['느리다', 'Adjective', ['느리', '느려', '느렸', '느린']],
  ['쉽다', 'Adjective', ['쉽', '쉬워', '쉬웠', '쉬운']],
  ['어렵다', 'Adjective', ['어렵', '어려워', '어려웠', '어려운']],
  ['많다', 'Adjective', ['많', '많았']],
  ['적다', 'Adjective', ['적', '적었']],
  ['크다', 'Adjective', ['크', '커', '컸', '큰']],
  ['작다', 'Adjective', ['작', '작았']],
  ['비싸다', 'Adjective', ['비싸', '비쌌', '비싼']],
  ['저렴하다', 'Adjective', ['저렴하', '저렴해', '저렴했', '저렴한']],
  ['정확하다', 'Adjective', ['정확하', '정확해', '정확했', '정확한']],
  ['부정확하다', 'Adjective', ['부정확하', '부정확해', '부정확한']],
  ['편하다', 'Adjective', ['편하', '편해', '편했', '편한']],
  ['불편하다', 'Adjective', ['불편하', '불편해', '불편했', '불편한']],
  ['아쉽다', 'Adjective', ['아쉽', '아쉬워', '아쉬웠', '아쉬운']],
  ['만족하다', 'Verb', ['만족하', '만족해', '만족했', '만족한']],
  ['실망하다', 'Verb', ['실망하', '실망해', '실망했', '실망한']],
  ['추천하다', 'Verb', ['추천하', '추천해', '추천했', '추천한']],
];

const VERB_BY_STEM = (() => {
  const m = [];
  for (const [base, pos, stems] of VERBS) {
    for (const s of stems) m.push({ stem: s, base, pos });
  }
  // 긴 어간부터 먼저 맞춰 본다 (생각하 < 생각했 순서 문제 방지)
  m.sort((a, b) => b.stem.length - a.stem.length);
  return m;
})();

/* ────────────────────────── ④ 부사·관형사 ──────────────────────────────── */
/* 「명사 + 하다」의 꼬리들. 긴 것부터 확인해야 「했습니다」가 「했」으로 잘리지 않는다. */
const HADA_ENDINGS = [
  '했습니다', '하겠습니다', '하십시오', '했어요', '합니다', '하려면', '하시면', '하는데',
  '하면서', '하지만', '해서는', '하려고', '하도록', '해야', '하면', '하고', '해서', '했다',
  '한다', '하는', '하기', '하지', '하여', '해요', '했던', '할', '함', '해',
];

const ADVERBS = new Set(`
정말 진짜 매우 아주 너무 조금 좀 잘 더 훨씬 가장 제일 항상 자주 가끔 거의 전혀 별로 다시 먼저 바로 특히 역시
그리고 그러나 그런데 그래서 하지만 또한 또 즉 따라서 그러므로 예를 물론 다만 오히려 결국 마침내 이제 아직 벌써
`.trim().split(/\s+/));

const DETERMINERS = new Set(['이', '그', '저', '어떤', '무슨', '이런', '그런', '저런', '모든', '각', '여러', '새']);

/* ────────────────────────── 어절 하나 쪼개기 ───────────────────────────── */

const isHangul = (s) => /^[가-힣]+$/.test(s);
const isLatin = (s) => /^[A-Za-z][A-Za-z0-9_]*$/.test(s);
const isNumber = (s) => /^[0-9]+(\.[0-9]+)?$/.test(s);

function splitCompound(word) {
  // 두 낱말이 붙은 복합명사 (도서관이용 → 도서관 + 이용)
  for (let i = 2; i <= word.length - 1; i++) {
    const a = word.slice(0, i);
    const c = word.slice(i);
    if (NOUNS.has(a) && NOUNS.has(c)) return [a, c];
  }
  return null;
}

function analyzeEojeol(word, stemMode) {
  const out = [];
  if (!word) return out;
  if (isNumber(word)) { out.push({ t: word, p: 'Number' }); return out; }
  if (isLatin(word)) { out.push({ t: word, p: 'Alpha' }); return out; }
  if (!isHangul(word)) { out.push({ t: word, p: 'Punctuation' }); return out; }

  if (ADVERBS.has(word)) { out.push({ t: word, p: 'Adverb' }); return out; }
  if (DETERMINERS.has(word) && word.length >= 2) { out.push({ t: word, p: 'Determiner' }); return out; }

  const cands = josaCandidates(word);

  // ① 조사를 떼어 낸 갈래 중 명사 사전에 걸리는 첫 번째 (가장 적게 뗀 것)
  for (const c of cands) {
    if (c.stem && NOUNS.has(c.stem)) {
      out.push({ t: c.stem, p: 'Noun' });
      c.josa.forEach((j) => out.push({ t: j, p: 'Josa' }));
      return out;
    }
  }

  // ②-a 「명사 + 하다」 꼴 (연체하면 → 연체 + 하면, 주문했어요 → 주문 + 했어요)
  //     이 처리를 넣지 않으면 「연체하면」이 통째로 한 낱말이 되어
  //     문서 속 「연체」와 이어지지 않는다. 검색이 실패하는 흔한 원인이다.
  for (const suf of HADA_ENDINGS) {
    if (word.length > suf.length && word.endsWith(suf)) {
      const front = word.slice(0, -suf.length);
      if (NOUNS.has(front)) {
        out.push({ t: front, p: 'Noun' });
        out.push({ t: stemMode ? front + '하다' : suf, p: 'Verb' });
        return out;
      }
    }
  }

  // ② 용언 (조사를 떼지 않은 원래 어절로 본다 — 어미가 조사처럼 보일 수 있으므로)
  for (const v of VERB_BY_STEM) {
    if (word.startsWith(v.stem)) {
      const tail = word.slice(v.stem.length);
      // 어미로 볼 수 있는 꼬리인지 (한글이고 4글자 이하)
      if (tail.length <= 4 && (tail === '' || isHangul(tail))) {
        out.push({ t: stemMode ? v.base : word, p: v.pos });
        return out;
      }
    }
  }

  // ③ 복합명사 (도서관이용 → 도서관 + 이용)
  for (const c of cands) {
    const comp = c.stem && splitCompound(c.stem);
    if (comp) {
      comp.forEach((x) => out.push({ t: x, p: 'Noun' }));
      c.josa.forEach((j) => out.push({ t: j, p: 'Josa' }));
      return out;
    }
  }

  // ④ 모르는 낱말 — 조사를 한 겹만 떼고 나머지는 명사로 추정
  const one = cands[1];
  if (one && one.stem) {
    out.push({ t: one.stem, p: 'Noun' });
    one.josa.forEach((j) => out.push({ t: j, p: 'Josa' }));
    return out;
  }
  out.push({ t: word, p: 'Noun' });
  return out;
}

/* ────────────────────────── 바깥에 내보내는 함수 ───────────────────────── */

/** 문장을 어절 + 문장부호로 나눈다 */
function eojeols(text) {
  const out = [];
  const re = /[가-힣]+|[A-Za-z][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[^\s]/g;
  let m;
  while ((m = re.exec(text))) out.push(m[0]);
  return out;
}

/** 형태소 + 품사. KoNLPy 의 okt.pos() 자리에 해당한다. */
export function pos(text, { stem = false } = {}) {
  return eojeols(text).flatMap((w) => analyzeEojeol(w, stem));
}

/** 형태소만. okt.morphs() 자리 */
export function morphs(text, opt) {
  return pos(text, opt).map((x) => x.t);
}

/** 명사만. okt.nouns() 자리 — 텍스트 분석에서 가장 많이 쓴다. */
export function nouns(text) {
  return pos(text).filter((x) => x.p === 'Noun').map((x) => x.t);
}

/** 품사 이름을 한국어로 */
export const POS_KO = {
  Noun: '명사', Josa: '조사', Verb: '동사', Adjective: '형용사', Adverb: '부사',
  Determiner: '관형사', Number: '숫자', Alpha: '영문', Punctuation: '기호',
};

/* ──────────────────────── 조사 자동으로 붙이기 ─────────────────────────── */
/* 낱말이 데이터에서 오기 때문에 문장을 미리 정할 수 없다.
   「토큰이」/「토큰가」 가 아니라 받침에 맞게 골라 준다. */

function hasFinalConsonant(word) {
  const s = String(word).trim();
  if (!s) return false;
  const last = s[s.length - 1];
  const cp = last.charCodeAt(0);
  if (cp >= 0xac00 && cp <= 0xd7a3) return (cp - 0xac00) % 28 !== 0;
  // 숫자·영문은 읽는 소리로 판단한다 (8266 → 「육」 → 받침 있음)
  const READ = { 0: true, 1: true, 2: false, 3: true, 4: false, 5: false, 6: true, 7: true, 8: true, 9: false };
  if (/[0-9]/.test(last)) return READ[Number(last)];
  const L = { l: true, m: true, n: true, r: true, ng: true };
  if (/[A-Za-z]/.test(last)) return !!L[last.toLowerCase()];
  return false;
}

/** josa('토큰','이') → '토큰이' / josa('벡터','이') → '벡터가' */
export function josa(word, kind) {
  const has = hasFinalConsonant(word);
  const PAIR = {
    '이': ['이', '가'], '가': ['이', '가'],
    '은': ['은', '는'], '는': ['은', '는'],
    '을': ['을', '를'], '를': ['을', '를'],
    '과': ['과', '와'], '와': ['과', '와'],
    '으로': ['으로', '로'], '로': ['으로', '로'],
    '이라': ['이라', '라'], '아': ['아', '야'],
  };
  const p = PAIR[kind];
  if (!p) return word + kind;
  // '로' 는 ㄹ 받침이면 '로' 를 쓴다 (물로, 서울로)
  if ((kind === '으로' || kind === '로') && has && /[ㄹ]|[가-힣]$/.test(word)) {
    const cp = word.charCodeAt(word.length - 1);
    if (cp >= 0xac00 && cp <= 0xd7a3 && (cp - 0xac00) % 28 === 8) return word + '로';
  }
  return word + (has ? p[0] : p[1]);
}
