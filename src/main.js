/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * main.js — 앱의 뼈대
 *   탭 여섯 개, 그 아래 화면 단추, 주소 해시(#rag/pipeline)로 화면 바로 가기.
 *   화면을 그리기 전에 반드시 beginScreen() 을 불러 앞 화면이 남긴
 *   리스너·타이머를 걷어 낸다. (탭을 오갈 때 쌓이는 것을 막는다)
 * ========================================================================== */

import './style.css';
import { h, fill, beginScreen, button } from './lib/ui.js';
import { markVisited, isVisited, summary, clearAll } from './lib/store.js';

import TAB1 from './tabs/t1-text.js';
import TAB2 from './tabs/t2-number.js';
import TAB3 from './tabs/t3-context.js';
import TAB4 from './tabs/t4-llm.js';
import TAB5 from './tabs/t5-rag.js';
import TAB6 from './tabs/t6-agent.js';
import TAB7 from './tabs/t7-python.js';

const TABS = [TAB1, TAB2, TAB3, TAB4, TAB5, TAB6, TAB7];

/* ─────────────────────────── 화면 찾아가기 ──────────────────────────── */
function findScreen(tabId, screenId) {
  const tab = TABS.find((t) => t.id === tabId) || TABS[0];
  const screen = tab.screens.find((s) => s.id === screenId) || tab.screens[0];
  return { tab, screen };
}

function go(tabId, screenId) {
  location.hash = `#${tabId}/${screenId}`;
}

/* ─────────────────────────────── 그리기 ─────────────────────────────── */
const app = document.getElementById('app');
const tabBar = h('nav.tabs');
const screenBar = h('div.screenbar-in');
const content = h('main');

function buildShell() {
  const top = h('header.topbar',
    h('div.topbar-in',
      h('div.brand',
        h('h1', 'LLM·RAG 실습실'),
        h('span.sub', '글을 숫자로 → 뜻을 배우는 기계 → 내 문서로 답하는 AI'),
      ),
      button('내 기록 지우기', () => {
        if (confirm('이 컴퓨터에 저장된 퀴즈 점수와 진도를 모두 지웁니다. 계속할까요?')) {
          clearAll();
          render();
        }
      }, 'ghost small'),
    ),
    tabBar,
  );
  const foot = h('footer.foot',
    h('div.foot-in',
      h('span', h('b', '© 2026 티쳐무'), ' · 모든 권리 보유 — 무단 배포 및 상업적 이용을 금합니다.'),
      h('span', '학교 수업 목적으로만 이용해 주세요.'),
      h('span', '개인정보를 수집하지 않습니다. 점수는 이 브라우저에만 남습니다.'),
    ),
  );
  app.append(top, h('div.screenbar', screenBar), content, foot);
}

function render() {
  const [tabId, screenId] = location.hash.replace(/^#/, '').split('/');
  const { tab, screen } = findScreen(tabId, screenId);

  // 탭 단추
  fill(tabBar, ...TABS.map((t, i) => {
    const btn = h('button.tab' + (t === tab ? '.on' : ''), {
      type: 'button',
      onclick: () => go(t.id, t.screens[0].id),
    }, h('span.tnum', t.num), t.title);
    return btn;
  }));

  // 화면 단추
  fill(screenBar, ...tab.screens.map((s) => h('button.sbtn' + (s === screen ? '.on' : ''), {
    type: 'button',
    onclick: () => go(tab.id, s.id),
  }, s.title, isVisited(`${tab.id}/${s.id}`) ? h('span.dot', '●') : null)));

  // 본문 — 앞 화면이 남긴 것을 먼저 걷어 낸다
  beginScreen();
  content.textContent = '';
  markVisited(`${tab.id}/${s2id(screen)}`);
  const ctx = { go, summary, TABS };
  try {
    content.appendChild(screen.render(ctx));
  } catch (e) {
    content.appendChild(h('div.note.note-bad.render-error',
      h('span.note-icon', '❌'),
      h('div.note-body',
        h('b', '이 화면을 그리는 중 오류가 났습니다.'),
        h('div', String(e && e.message ? e.message : e)),
      ),
    ));
    console.error(e);
  }
  window.scrollTo({ top: 0 });
  document.title = `${screen.title} · LLM·RAG 실습실`;
}

const s2id = (s) => s.id;

window.addEventListener('hashchange', render);
buildShell();
if (!location.hash) location.hash = `#${TABS[0].id}/${TABS[0].screens[0].id}`;
render();
