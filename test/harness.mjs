/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * harness.mjs — 아주 작은 시험 도구
 *   외부 시험 라이브러리를 쓰지 않는다. node 만 있으면 돌아간다.
 * ========================================================================== */

let total = 0;
let failed = 0;
const failures = [];
let group = '';

export function describe(name, fn) {
  group = name;
  fn();
  group = '';
}

export function test(name, fn) {
  total += 1;
  try {
    fn();
  } catch (e) {
    failed += 1;
    failures.push({ group, name, message: e.message });
  }
}

export function eq(got, want, msg) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g !== w) throw new Error(`${msg || ''} 기대 ${w} 이지만 ${g} 이(가) 나왔습니다`);
}

export function near(got, want, tol = 1e-4, msg) {
  if (!Number.isFinite(got) || Math.abs(got - want) > tol) {
    throw new Error(`${msg || ''} 기대 ${want} (±${tol}) 이지만 ${got} 이(가) 나왔습니다`);
  }
}

export function ok(cond, msg) {
  if (!cond) throw new Error(msg || '참이어야 하는데 거짓입니다');
}

export function includes(hay, needle, msg) {
  if (!String(hay).includes(needle)) {
    throw new Error(`${msg || ''} "${needle}" 이(가) 들어 있어야 하는데 없습니다: ${String(hay).slice(0, 160)}`);
  }
}

export function report() {
  const pass = total - failed;
  console.log('');
  console.log('─'.repeat(64));
  if (failed === 0) {
    console.log(`✅ 시험 ${total}가지 모두 통과`);
  } else {
    console.log(`❌ ${failed}가지 실패 / 모두 ${total}가지 (통과 ${pass})`);
    console.log('');
    for (const f of failures) {
      console.log(`  · [${f.group}] ${f.name}`);
      console.log(`      ${f.message}`);
    }
  }
  console.log('─'.repeat(64));
  return failed === 0;
}

export const counts = () => ({ total, failed });
