/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 (무단 배포·상업적 이용 금지) */
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// 저작권 배너 — /*! 로 시작해야 압축 후에도 살아남는다.
const BANNER = '/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유. 학교 수업 목적으로만 이용해 주세요. */';

export default defineConfig({
  base: './', // GitHub Pages 하위 경로에서도 자원이 열리도록 상대 경로
  plugins: [viteSingleFile()],
  esbuild: {
    legalComments: 'inline', // /*! 주석을 결과물에 남긴다
  },
  build: {
    target: 'es2020',
    assetsInlineLimit: 100000000,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        banner: BANNER,
        inlineDynamicImports: true,
      },
    },
  },
});
