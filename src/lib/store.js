/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * store.js — 그 브라우저에만 남는 학습 기록
 *
 * 개인정보를 한 글자도 받지 않는다. 이름·학번 칸이 없고 서버도 없다.
 * 퀴즈 점수와 본 화면 목록만 localStorage 에 남으며,
 * 공용 컴퓨터를 생각해 [내 기록 지우기] 를 한 번에 누를 수 있게 해 두었다.
 * ========================================================================== */

const KEY = 'llm-rag-lab/v1';

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function write(obj) {
  try {
    localStorage.setItem(KEY, JSON.stringify(obj));
  } catch {
    /* 사생활 보호 모드 등에서 저장이 막혀도 앱은 그대로 돌아가야 한다 */
  }
}

/** 퀴즈 결과 기록 (같은 열쇠는 더 높은 점수만 남긴다) */
export function recordQuiz(id, ok, total) {
  const s = read();
  s.quiz = s.quiz || {};
  const prev = s.quiz[id];
  if (!prev || ok > prev.ok) s.quiz[id] = { ok, total };
  write(s);
}

/** 화면을 열었다고 표시 */
export function markVisited(screenId) {
  const s = read();
  s.visited = s.visited || {};
  s.visited[screenId] = 1;
  write(s);
}

export function isVisited(screenId) {
  return !!read().visited?.[screenId];
}

/** 전체 요약 { visited, quizDone, quizOk, quizTotal } */
export function summary() {
  const s = read();
  const quiz = s.quiz || {};
  let ok = 0;
  let total = 0;
  for (const v of Object.values(quiz)) {
    ok += v.ok;
    total += v.total;
  }
  return {
    visited: Object.keys(s.visited || {}).length,
    quizSets: Object.keys(quiz).length,
    quizOk: ok,
    quizTotal: total,
  };
}

/** 화면별 자유 저장소 (실습창 코드 등) */
export function getScratch(id, fallback = '') {
  return read().scratch?.[id] ?? fallback;
}

export function setScratch(id, value) {
  const s = read();
  s.scratch = s.scratch || {};
  s.scratch[id] = value;
  write(s);
}

export function clearAll() {
  try {
    localStorage.removeItem(KEY);
  } catch { /* 무시 */ }
}
