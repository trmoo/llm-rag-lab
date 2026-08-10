/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * quiz.js — 화면 중간중간 넣는 이해도 확인 문제
 *
 * 문항 종류 세 가지
 *   choice   보기 중 하나 고르기      { q, options:[…], answer: 인덱스, why }
 *   multi    보기 중 여럿 고르기      { q, options:[…], answer:[인덱스…], why }
 *   short    짧게 써 넣기            { q, accept:[정답문자열…], why }
 *
 * 채점 원칙
 *   · 아무것도 안 고르고 [확인] 을 누르면 「고르지 않았다」고 알려 준다.
 *     (choicePicker 를 쓰는 이유 — 첫 보기가 기본 선택이 되면 안 된다)
 *   · 틀려도 정답을 바로 알려 주고, 왜 그런지( why )를 함께 보여 준다.
 *   · 점수는 그 브라우저에만 남는다. 이름·학번은 받지 않는다.
 * ========================================================================== */

import { h, button, note, choicePicker, input } from './ui.js';
import { recordQuiz } from './store.js';

/** 짧은 답 채점용 — 공백·문장부호·대소문자를 무시하고 비교한다. */
export function normAnswer(s) {
  return String(s)
    .trim()
    .toLowerCase()
    .replace(/[\s,.·'"`()[\]{}!?~-]/g, '')
    .replace(/입니다$|이다$|예요$|에요$/, '');
}

/**
 * 문항 묶음 하나를 화면 조각으로 만든다.
 * @param {string} id   진도 저장용 열쇠 (화면마다 다르게)
 * @param {object[]} items 문항 배열
 */
export function quizBlock(id, items, opts = {}) {
  const title = opts.title || '이해도 확인';
  const wrap = h('section.quiz');
  const head = h('div.quiz-head',
    h('span.quiz-badge', '❓ ' + title),
    h('span.quiz-count', `${items.length}문항`),
  );
  wrap.appendChild(head);

  const state = items.map(() => ({ done: false, ok: false }));
  const scoreLine = h('div.quiz-score');

  const updateScore = () => {
    const done = state.filter((s) => s.done).length;
    const ok = state.filter((s) => s.ok).length;
    scoreLine.textContent = done === 0 ? '' : `푼 문제 ${done}/${items.length} · 맞힌 문제 ${ok}개`;
    if (done === items.length) {
      recordQuiz(id, ok, items.length);
      scoreLine.classList.add('full');
      scoreLine.textContent += ok === items.length ? '  🎉 모두 맞혔습니다!' : '  다시 풀어 봐도 좋아요.';
    }
  };

  items.forEach((item, i) => {
    wrap.appendChild(oneItem(item, i, (ok) => {
      state[i] = { done: true, ok };
      updateScore();
    }));
  });
  wrap.appendChild(scoreLine);
  return wrap;
}

function oneItem(item, idx, done) {
  const box = h('div.quiz-item');
  box.appendChild(h('div.quiz-q', h('span.quiz-no', `Q${idx + 1}.`), ...(typeof item.q === 'function' ? item.q() : [item.q])));

  const feedback = h('div.quiz-feedback');
  feedback.hidden = true;

  let getValue;
  let resetter = () => {};

  if (item.type === 'short') {
    const inp = input({ placeholder: '답을 써 보세요', size: 24 });
    box.appendChild(h('div.quiz-body', inp));
    getValue = () => inp.value;
    resetter = () => { inp.value = ''; };
  } else if (item.type === 'multi') {
    const boxes = item.options.map((opt, i) => {
      const cb = h('input', { type: 'checkbox' });
      return { cb, el: h('label.quiz-check', cb, h('span', opt)), i };
    });
    box.appendChild(h('div.quiz-body.quiz-multi', ...boxes.map((x) => x.el)));
    getValue = () => boxes.filter((x) => x.cb.checked).map((x) => x.i);
    resetter = () => boxes.forEach((x) => { x.cb.checked = false; });
  } else {
    const picker = choicePicker(item.options.map((opt, i) => ({ label: opt, value: i })));
    picker.classList.add('quiz-choice');
    box.appendChild(h('div.quiz-body', picker));
    getValue = () => picker.getValue();
    resetter = () => picker.reset();
  }

  let answered = false;
  const btn = button('확인', () => {
    if (answered) { // 다시 풀기
      answered = false;
      feedback.hidden = true;
      btn.textContent = '확인';
      box.classList.remove('ok', 'no');
      resetter();
      return;
    }
    const v = getValue();
    const judged = judge(item, v);
    if (judged === null) {
      feedback.hidden = false;
      feedback.className = 'quiz-feedback warn';
      feedback.textContent = item.type === 'short' ? '답을 먼저 써 주세요.' : '보기를 먼저 골라 주세요.';
      return;
    }
    answered = true;
    btn.textContent = '다시 풀기';
    box.classList.add(judged ? 'ok' : 'no');
    feedback.hidden = false;
    feedback.className = 'quiz-feedback ' + (judged ? 'ok' : 'no');
    feedback.textContent = '';
    feedback.appendChild(h('div.quiz-verdict', judged ? '⭕ 맞았습니다' : '❌ 아쉽습니다'));
    if (!judged) feedback.appendChild(h('div.quiz-answer', '정답: ' + answerText(item)));
    if (item.why) feedback.appendChild(h('div.quiz-why', item.why));
    done(judged);
  }, 'small');

  box.appendChild(h('div.quiz-actions', btn));
  box.appendChild(feedback);
  return box;
}

function judge(item, v) {
  if (item.type === 'short') {
    if (!String(v || '').trim()) return null;
    const got = normAnswer(v);
    return item.accept.some((a) => normAnswer(a) === got);
  }
  if (item.type === 'multi') {
    if (!v || v.length === 0) return null;
    const want = [...item.answer].sort().join(',');
    return [...v].sort().join(',') === want;
  }
  if (v === null || v === undefined) return null;
  return v === item.answer;
}

function answerText(item) {
  if (item.type === 'short') return item.accept[0];
  if (item.type === 'multi') return item.answer.map((i) => item.options[i]).join(' / ');
  return item.options[item.answer];
}

/** 한 문항짜리 짧은 확인용 (화면 중간에 끼워 넣을 때) */
export function miniQuiz(id, item) {
  return quizBlock(id, [item], { title: '잠깐, 확인!' });
}

/** 정답이 없는 「생각해 보기」 — 채점하지 않는다. */
export function think(...children) {
  return note('ask', h('b', '생각해 보기 '), ...children);
}
