/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * syntax.mjs — 소스 파일 전수 문법 검사
 *
 * 왜 따로 두었나?
 *   npm test 는 화면(탭) 파일을 불러오지 않는다. DOM 이 없어서 돌릴 수 없기 때문이다.
 *   그래서 탭 파일에 괄호 하나를 빠뜨려도 시험은 통과하고 빌드에서야 터진다.
 *   이 검사가 그 틈을 막는다. 배포 워크플로에서도 빌드 전에 돌린다.
 * ========================================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const root = path.dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const files = [];

function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.js')) files.push(p);
  }
}
walk(path.join(root, 'src'));
files.push(path.join(root, 'vite.config.js'));

let bad = 0;
for (const f of files) {
  const src = fs.readFileSync(f, 'utf8');
  try {
    esbuild.transformSync(src, { loader: 'js', format: 'esm' });
  } catch (e) {
    bad += 1;
    console.log(`❌ ${path.relative(root, f)}`);
    for (const err of e.errors || []) {
      console.log(`   ${err.location?.line}행: ${err.text}`);
    }
  }
}

if (bad === 0) console.log(`✅ 문법 검사 — 파일 ${files.length}개 모두 정상`);
process.exit(bad === 0 ? 0 : 1);
