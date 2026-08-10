/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * run.mjs — 시험을 모아서 돌린다.
 *   시험 파일을 늘릴 때는 여기 목록에 한 줄 더하면 된다.
 *   (여러 사람이 동시에 만들어도 같은 파일을 건드리지 않도록 나눠 두었다)
 * ========================================================================== */

import { report, counts } from './harness.mjs';

const FILES = [
  './nlp.test.mjs',
  './rag.test.mjs',
  './python.test.mjs',
  './app.test.mjs',
];

console.log('LLM·RAG 실습실 — 시험을 시작합니다\n');
for (const f of FILES) {
  const before = counts().total;
  await import(f);
  const after = counts();
  console.log(`  ${f.replace('./', '').padEnd(20)} ${after.total - before}가지`);
}

process.exit(report() ? 0 : 1);
