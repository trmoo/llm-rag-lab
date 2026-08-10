/*! LLM·RAG 실습실 — © 2026 티쳐무 · 모든 권리 보유 */
/* ============================================================================
 * pymini.js — 교실용 미니 파이썬 해석기
 *
 * 왜 직접 만들었나?
 *   교실 인터넷이 끊겨도 학생이 쓴 코드가 그 자리에서 돌아가야 한다.
 *   그래서 수업에 나오는 문법만 골라서 해석하는 작은 파이썬을 직접 만들었다.
 *   (파이썬 자체나 다른 라이브러리의 코드를 가져다 쓰지 않았다.)
 *
 * 지원하는 문법
 *   class / def / __init__ / self / return / if·elif·else / while
 *   for x in range(...) / f-문자열 / 리스트 컴프리헨션 / print / 산술·비교·논리 연산
 *   비공개 속성(__속성명)의 이름 맹글링과 AttributeError 까지 그대로 재현한다.
 *
 * 텍스트 분석 수업을 위해 더한 것 (이 앱에서 추가)
 *   내장 함수  sorted · enumerate · zip · set · any · all · reversed · dict · tuple
 *   모듈       re (sub·findall·split·search·match)
 *              collections (Counter → .most_common)
 *              korean (교실용 간이 형태소 분석기 — nouns·morphs·pos·josa)
 *   ⚠️ set( ) 은 「중복을 없앤 리스트」를 돌려준다. 진짜 집합과 다르지만
 *      수업에서 쓰는 자리(어휘 만들기)에서는 결과가 같다.
 *
 * 크게 세 단계로 동작한다.
 *   ① tokenize() — 소스를 토큰(낱말)으로 쪼갠다. 들여쓰기도 토큰으로 만든다.
 *   ② parse()    — 토큰을 문법 나무(AST)로 조립한다.
 *   ③ run()      — 문법 나무를 위에서부터 하나씩 실행한다.
 * ========================================================================== */

import * as KOREAN from './korean.js';

/* ────────────────────────────── 파이썬 값 표현 ─────────────────────────────
 * 파이썬의 값을 자바스크립트 객체 하나로 통일해서 다룬다.
 * k(kind)가 종류, v가 실제 값이다. 정수와 실수를 구분해 두어야
 * 파이썬처럼 5 와 5.0 을 다르게 출력할 수 있다.
 * ------------------------------------------------------------------------- */
const INT = (v) => ({ k: 'int', v });
const FLT = (v) => ({ k: 'float', v });
const STR = (v) => ({ k: 'str', v });
const BOOL = (v) => ({ k: 'bool', v: !!v });
const NONE = { k: 'none', v: null };
const LIST = (v) => ({ k: 'list', v });
const DICT = (v) => ({ k: 'dict', v }); // v: Map (키는 원시값)

const isNum = (x) => x.k === 'int' || x.k === 'float';

/** 딕셔너리 키(원시값)를 다시 파이썬 값으로 감싼다 */
const wrapKey = (k) => (typeof k === 'string' ? STR(k)
  : typeof k === 'boolean' ? BOOL(k)
    : Number.isInteger(k) ? INT(k) : FLT(k));

/** 파이썬 오류를 흉내낸 예외. type 은 ValueError·AttributeError 같은 이름. */
export class PyError extends Error {
  constructor(type, msg, line, hint) {
    super(msg);
    this.type = type;
    this.line = line;
    this.hint = hint || '';
  }
  /** 화면에 보여 줄 문구 (파이썬 형식 + 한국어 도움말) */
  format() {
    const head = this.line ? `${this.line}번째 줄 — ${this.type}: ${this.message}` : `${this.type}: ${this.message}`;
    return this.hint ? `${head}\n  ↳ ${this.hint}` : head;
  }
}
const err = (type, msg, line, hint) => {
  throw new PyError(type, msg, line, hint);
};

/* ══════════════════════════════ ① 토큰화 ══════════════════════════════════ */

const KEYWORDS = new Set([
  'class', 'def', 'return', 'if', 'elif', 'else', 'while', 'for', 'in', 'not',
  'and', 'or', 'pass', 'break', 'continue', 'import', 'from', 'True', 'False',
  'None', 'is', 'global', 'del',
]);

// 긴 연산자를 먼저 찾아야 '==' 가 '=' 두 개로 쪼개지지 않는다.
const OPS = [
  '**=', '//=', '...',
  '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '->',
  '+', '-', '*', '/', '%', '<', '>', '=', '(', ')', '[', ']', '{', '}',
  ',', ':', '.', ';',
];

/** 교안 PDF에서 복사한 코드에 섞여 있는 특수 문자를 보통 문자로 바꾼다. */
function normalize(src) {
  return src
    .replace(/\r\n?/g, '\n')
    .replace(/[‘’ʼ′]/g, "'")   // ‘ ’ → '
    .replace(/[“”″]/g, '"')          // “ ” → "
    .replace(/[ ​　]/g, ' ')          // 특수 공백 → 보통 공백
    .replace(/\t/g, '    ');                        // 탭 → 공백 4칸
}

function tokenize(srcRaw) {
  const src = normalize(srcRaw);
  const toks = [];
  const indents = [0];
  let i = 0, line = 1, depth = 0, atLineStart = true;

  const push = (type, value) => toks.push({ type, value, line });

  while (i < src.length) {
    // ── 줄 맨 앞: 들여쓰기 계산 ──
    if (atLineStart && depth === 0) {
      let ws = 0;
      while (i < src.length && src[i] === ' ') { ws++; i++; }
      // 빈 줄이나 주석만 있는 줄은 들여쓰기를 따지지 않는다.
      if (src[i] === '\n' || src[i] === '#' || i >= src.length) {
        if (src[i] === '#') { while (i < src.length && src[i] !== '\n') i++; }
        if (src[i] === '\n') { i++; line++; }
        continue;
      }
      if (ws > indents[indents.length - 1]) {
        indents.push(ws);
        push('INDENT', ws);
      } else {
        while (ws < indents[indents.length - 1]) {
          indents.pop();
          push('DEDENT', ws);
        }
        if (ws !== indents[indents.length - 1]) {
          err('IndentationError', '들여쓰기 칸 수가 맞지 않습니다', line,
            '같은 블록 안의 문장은 들여쓰기 칸 수를 똑같이 맞춰 주세요.');
        }
      }
      atLineStart = false;
      continue;
    }

    const c = src[i];

    if (c === '\n') {
      i++;
      if (depth === 0) { push('NEWLINE', '\n'); atLineStart = true; }
      line++;
      continue;
    }
    if (c === ' ') { i++; continue; }
    if (c === '#') { while (i < src.length && src[i] !== '\n') i++; continue; }
    if (c === '\\' && src[i + 1] === '\n') { i += 2; line++; continue; } // 줄 이어쓰기

    // ── 문자열 (f-string · r-string 포함) ──
    // r'...' 은 역슬래시를 그대로 두는 「날 문자열」이다. 정규표현식을 쓸 때 꼭 필요하다.
    const isQuote = (ch) => ch === '"' || ch === "'";
    const fPrefix = (c === 'f' || c === 'F') && isQuote(src[i + 1]);
    const rPrefix = (c === 'r' || c === 'R') && isQuote(src[i + 1]);
    if (c === '"' || c === "'" || fPrefix || rPrefix) {
      const start = line;
      if (fPrefix || rPrefix) i++;
      const q = src[i];
      const triple = src.slice(i, i + 3) === q.repeat(3);
      i += triple ? 3 : 1;
      let out = '';
      for (;;) {
        if (i >= src.length) err('SyntaxError', '문자열의 따옴표가 닫히지 않았습니다', start);
        if (!triple && src[i] === '\n') err('SyntaxError', '문자열의 따옴표가 닫히지 않았습니다', start);
        if (triple && src.slice(i, i + 3) === q.repeat(3)) { i += 3; break; }
        if (!triple && src[i] === q) { i++; break; }
        if (src[i] === '\\') {
          const e = src[i + 1];
          if (rPrefix) {
            // 날 문자열에서는 \d, \s 처럼 역슬래시를 그대로 남긴다 (따옴표 탈출만 처리)
            out += (e === q ? e : '\\' + e);
            i += 2;
            continue;
          }
          out += e === 'n' ? '\n' : e === 't' ? '\t' : e === '\\' ? '\\'
            : e === "'" ? "'" : e === '"' ? '"' : e === '\n' ? '' : '\\' + e;
          if (e === '\n') line++;
          i += 2;
          continue;
        }
        if (src[i] === '\n') line++;
        out += src[i++];
      }
      push(fPrefix ? 'FSTRING' : 'STRING', out);
      continue;
    }

    // ── 숫자 ──
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1]))) {
      let s = '';
      while (i < src.length && /[0-9_]/.test(src[i])) s += src[i++];
      let float = false;
      if (src[i] === '.' && /[0-9]/.test(src[i + 1] || '')) {
        float = true; s += src[i++];
        while (i < src.length && /[0-9_]/.test(src[i])) s += src[i++];
      } else if (src[i] === '.' && !/[a-zA-Z_]/.test(src[i + 1] || '')) {
        float = true; s += src[i++];
      }
      if (src[i] === 'e' || src[i] === 'E') {
        float = true; s += src[i++];
        if (src[i] === '+' || src[i] === '-') s += src[i++];
        while (i < src.length && /[0-9]/.test(src[i])) s += src[i++];
      }
      push('NUMBER', { v: parseFloat(s.replace(/_/g, '')), float });
      continue;
    }

    // ── 이름·키워드 (한글 이름도 허용) ──
    if (/[A-Za-z_가-힣ㄱ-ㆎ]/.test(c)) {
      let s = '';
      while (i < src.length && /[A-Za-z0-9_가-힣ㄱ-ㆎ]/.test(src[i])) s += src[i++];
      push(KEYWORDS.has(s) ? 'KEY' : 'NAME', s);
      continue;
    }

    // ── 연산자·구두점 ──
    const op = OPS.find((o) => src.startsWith(o, i));
    if (op) {
      if ('([{'.includes(op)) depth++;
      if (')]}'.includes(op)) depth = Math.max(0, depth - 1);
      i += op.length;
      push('OP', op);
      continue;
    }

    err('SyntaxError', `알 수 없는 문자 '${c}' 입니다`, line,
      '한글 자판으로 입력된 기호가 섞였는지 확인해 보세요.');
  }

  if (toks.length && toks[toks.length - 1].type !== 'NEWLINE') push('NEWLINE', '\n');
  while (indents.length > 1) { indents.pop(); push('DEDENT', 0); }
  push('EOF', null);
  return toks;
}

/* ══════════════════════════════ ② 구문 분석 ═══════════════════════════════ */

class Parser {
  constructor(toks) { this.toks = toks; this.p = 0; }

  peek(n = 0) { return this.toks[this.p + n]; }
  get line() { return this.peek().line; }
  next() { return this.toks[this.p++]; }
  at(type, value) {
    const t = this.peek();
    return t.type === type && (value === undefined || t.value === value);
  }
  eat(type, value) { if (this.at(type, value)) { return this.next(); } return null; }
  expect(type, value, what) {
    if (this.at(type, value)) return this.next();
    const t = this.peek();
    const got = t.type === 'NEWLINE' ? '줄바꿈' : t.type === 'EOF' ? '코드 끝'
      : t.type === 'INDENT' ? '들여쓰기' : t.type === 'DEDENT' ? '내어쓰기' : `'${t.value}'`;
    err('SyntaxError', `${what || `'${value ?? type}'`} 이(가) 필요한데 ${got} 을(를) 만났습니다`, t.line);
  }
  skipNewlines() { while (this.eat('NEWLINE')) { /* 빈 줄 넘기기 */ } }

  parseProgram() {
    const body = [];
    this.skipNewlines();
    while (!this.at('EOF')) {
      body.push(...this.statement());
      this.skipNewlines();
    }
    return { type: 'Module', body };
  }

  /** ':' 다음의 블록. 여러 줄이거나 한 줄로 붙여 쓴 형태 모두 받는다. */
  block(what) {
    this.expect('OP', ':', `'${what}' 뒤의 콜론(:)`);
    if (this.eat('NEWLINE')) {
      this.skipNewlines();
      if (!this.eat('INDENT')) {
        err('IndentationError', `'${what}' 아래에 들여 쓴 문장이 없습니다`, this.line,
          `${what} 다음 줄은 스페이스 4칸으로 들여 써야 합니다.`);
      }
      const body = [];
      this.skipNewlines();
      while (!this.at('DEDENT') && !this.at('EOF')) {
        body.push(...this.statement());
        this.skipNewlines();
      }
      this.eat('DEDENT');
      return body;
    }
    return this.simpleLine(); // 예) if x : print(x)
  }

  /** 세미콜론으로 이어 쓴 한 줄 */
  simpleLine() {
    const out = [this.simpleStatement()];
    while (this.eat('OP', ';')) {
      if (this.at('NEWLINE')) break;
      out.push(this.simpleStatement());
    }
    this.eat('NEWLINE');
    return out;
  }

  /* 교안 범위를 넘는 문법들 — 그냥 SyntaxError 를 내면 학생이 원인을 못 찾으므로
   * 무엇이 지원되지 않는지 이름을 대어 알려 준다. */
  checkUnsupported() {
    const t = this.peek();
    if (t.type !== 'NAME') return;
    const guide = {
      try: '예외 처리(try / except)', except: '예외 처리(try / except)', finally: '예외 처리(try / finally)',
      raise: '오류 일으키기(raise)', assert: '단정문(assert)', with: 'with 문',
      yield: 'yield', nonlocal: 'nonlocal 선언', async: 'async 문법', await: 'await 문법',
    }[t.value];
    if (!guide) return;
    // 조사를 붙이면 「…(try / except) 을(를)」 처럼 어색해지므로 줄표로 잇는다
    err('지원하지 않는 문법', `${guide} — 이 앱에서는 쓸 수 없습니다`, t.line,
      t.value === 'try' || t.value === 'except' || t.value === 'finally'
        ? '교안 범위에서는 if 문으로 값을 미리 검사하세요. 예) if age < 0 : print(\'잘못된 값\')'
        : '교안에 나오는 문법(class · def · if · while · for · print)으로 바꿔 써 보세요.');
  }

  statement() {
    this.checkUnsupported();
    const t = this.peek();
    if (t.type === 'KEY') {
      switch (t.value) {
        case 'class': return [this.classDef()];
        case 'def': return [this.funcDef()];
        case 'if': return [this.ifStmt()];
        case 'while': return [this.whileStmt()];
        case 'for': return [this.forStmt()];
      }
    }
    return this.simpleLine();
  }

  classDef() {
    const line = this.line;
    this.next(); // class
    const name = this.expect('NAME', undefined, '클래스 이름').value;

    /* 상속은 교안 범위 밖이라 지원하지 않는다.
     * 그런데 조용히 무시하면 물려받은 메소드를 부를 때
     * 「그런 메소드가 없다」는 엉뚱한 오류가 나서 학생이 헤맨다.
     * 그래서 여기서 분명히 알려 준다. class Foo(object) : 는 예외로 통과시킨다. */
    if (this.eat('OP', '(')) {
      const bases = [];
      let d = 1;
      while (d > 0 && !this.at('EOF')) {
        const x = this.next();
        if (x.type === 'OP' && x.value === '(') d++;
        else if (x.type === 'OP' && x.value === ')') d--;
        else if (x.type === 'NAME' && d === 1) bases.push(x.value);
      }
      const real = bases.filter((b) => b !== 'object');
      if (real.length) {
        err('지원하지 않는 문법', `class ${name}(${real.join(', ')}) — 상속은 이 앱에서 지원하지 않습니다`, line,
          `괄호와 그 안의 이름을 지우고 class ${name} : 로 쓰세요. 필요한 속성과 메소드는 이 클래스 안에 직접 정의해야 합니다. (상속은 교안 범위를 넘는 내용입니다)`);
      }
    }

    const body = this.block(`class ${name}`);
    return { type: 'ClassDef', name, body, line };
  }

  funcDef() {
    const line = this.line;
    this.next(); // def
    const nt = this.peek();
    if (nt.type !== 'NAME' && nt.type !== 'KEY') this.expect('NAME', undefined, '메소드(함수) 이름');
    const name = this.next().value;
    this.expect('OP', '(', `'def ${name}' 뒤의 여는 괄호`);
    const params = [];
    if (!this.at('OP', ')')) {
      do {
        if (this.eat('OP', '*')) { this.eat('OP', '*'); this.expect('NAME'); continue; } // *args 무시
        const pn = this.expect('NAME', undefined, '매개변수 이름').value;
        let def = null;
        if (this.eat('OP', '=')) def = this.expr();
        params.push({ name: pn, def });
      } while (this.eat('OP', ','));
    }
    this.expect('OP', ')', '닫는 괄호');
    const body = this.block(`def ${name}`);
    return { type: 'FuncDef', name, params, body, line };
  }

  ifStmt() {
    const line = this.line;
    this.next();
    const test = this.expr();
    const body = this.block('if');
    let orelse = [];
    this.skipNewlines();
    if (this.at('KEY', 'elif')) orelse = [this.ifStmt()];
    else if (this.at('KEY', 'else')) { this.next(); orelse = this.block('else'); }
    return { type: 'If', test, body, orelse, line };
  }

  whileStmt() {
    const line = this.line;
    this.next();
    const test = this.expr();
    const body = this.block('while');
    return { type: 'While', test, body, line };
  }

  forStmt() {
    const line = this.line;
    this.next();
    const targets = [this.expect('NAME', undefined, '반복 변수 이름').value];
    while (this.eat('OP', ',')) targets.push(this.expect('NAME').value);
    this.expect('KEY', 'in', "'in'");
    const iter = this.expr();
    const body = this.block('for');
    return { type: 'For', targets, iter, body, line };
  }

  simpleStatement() {
    const line = this.line;
    const t = this.peek();

    if (t.type === 'KEY') {
      if (t.value === 'pass') { this.next(); return { type: 'Pass', line }; }
      if (t.value === 'break') { this.next(); return { type: 'Break', line }; }
      if (t.value === 'continue') { this.next(); return { type: 'Continue', line }; }
      if (t.value === 'global' || t.value === 'del') { // 교안 범위 밖 — 조용히 넘긴다
        while (!this.at('NEWLINE') && !this.at('EOF')) this.next();
        return { type: 'Pass', line };
      }
      if (t.value === 'return') {
        this.next();
        const value = this.at('NEWLINE') || this.at('EOF') || this.at('OP', ';') ? null : this.expr();
        return { type: 'Return', value, line };
      }
      if (t.value === 'import') {
        this.next();
        const names = [];
        do { names.push(this.expect('NAME', undefined, '모듈 이름').value); } while (this.eat('OP', ','));
        return { type: 'Import', names, line };
      }
      if (t.value === 'from') {
        this.next();
        const mod = this.expect('NAME', undefined, '모듈 이름').value;
        this.expect('KEY', 'import', "'import'");
        const names = [];
        if (this.eat('OP', '*')) names.push('*');
        else do { names.push(this.next().value); } while (this.eat('OP', ','));
        return { type: 'FromImport', mod, names, line };
      }
    }

    const first = this.expr();
    const aug = ['+=', '-=', '*=', '/=', '//=', '%=', '**='];
    for (const a of aug) {
      if (this.at('OP', a)) {
        this.next();
        return { type: 'AugAssign', target: first, op: a.slice(0, -1), value: this.expr(), line };
      }
    }
    if (this.at('OP', '=')) {
      const targets = [first];
      let value = null;
      while (this.eat('OP', '=')) {
        const e = this.expr();
        if (this.at('OP', '=')) targets.push(e); else value = e;
      }
      for (const tg of targets) {
        if (!['Name', 'Attribute', 'Subscript'].includes(tg.type)) {
          err('SyntaxError', '왼쪽에 값을 담을 수 없습니다', line,
            '= 왼쪽에는 변수 이름이나 self.속성명 같은 것만 올 수 있습니다.');
        }
      }
      return { type: 'Assign', targets, value, line };
    }
    return { type: 'Expr', value: first, line };
  }

  /* ── 식(expression) 파싱: 우선순위가 낮은 것부터 ── */
  expr() { return this.orTest(); }

  orTest() {
    let n = this.andTest();
    while (this.at('KEY', 'or')) { this.next(); n = { type: 'Bool', op: 'or', l: n, r: this.andTest() }; }
    return n;
  }
  andTest() {
    let n = this.notTest();
    while (this.at('KEY', 'and')) { this.next(); n = { type: 'Bool', op: 'and', l: n, r: this.notTest() }; }
    return n;
  }
  notTest() {
    if (this.at('KEY', 'not')) { const line = this.line; this.next(); return { type: 'Not', v: this.notTest(), line }; }
    return this.comparison();
  }
  comparison() {
    const line = this.line;
    let left = this.arith();
    const ops = [];
    for (;;) {
      let op = null;
      for (const o of ['==', '!=', '<=', '>=', '<', '>']) if (this.at('OP', o)) { op = o; this.next(); break; }
      if (!op && this.at('KEY', 'in')) { op = 'in'; this.next(); }
      if (!op && this.at('KEY', 'not') && this.peek(1).type === 'KEY' && this.peek(1).value === 'in') { op = 'not in'; this.next(); this.next(); }
      if (!op && this.at('KEY', 'is')) {
        this.next();
        op = this.eat('KEY', 'not') ? 'is not' : 'is';
      }
      if (!op) break;
      ops.push({ op, right: this.arith() });
    }
    return ops.length ? { type: 'Compare', left, ops, line } : left;
  }
  arith() {
    let n = this.term();
    for (;;) {
      const line = this.line;
      if (this.at('OP', '+')) { this.next(); n = { type: 'Bin', op: '+', l: n, r: this.term(), line }; }
      else if (this.at('OP', '-')) { this.next(); n = { type: 'Bin', op: '-', l: n, r: this.term(), line }; }
      else break;
    }
    return n;
  }
  term() {
    let n = this.factor();
    for (;;) {
      const line = this.line;
      let op = null;
      for (const o of ['*', '//', '/', '%']) if (this.at('OP', o)) { op = o; this.next(); break; }
      if (!op) break;
      n = { type: 'Bin', op, l: n, r: this.factor(), line };
    }
    return n;
  }
  factor() {
    const line = this.line;
    if (this.at('OP', '-')) { this.next(); return { type: 'Unary', op: '-', v: this.factor(), line }; }
    if (this.at('OP', '+')) { this.next(); return this.factor(); }
    return this.power();
  }
  power() {
    const base = this.atomExpr();
    if (this.at('OP', '**')) { const line = this.line; this.next(); return { type: 'Bin', op: '**', l: base, r: this.factor(), line }; }
    return base;
  }

  atomExpr() {
    let n = this.atom();
    for (;;) {
      const line = this.line;
      if (this.at('OP', '(')) {
        this.next();
        const args = [];
        if (!this.at('OP', ')')) {
          do {
            if (this.at('OP', ')')) break; // 뒤에 붙은 쉼표
            // 키워드 인수 예) setAge(age=17)
            if ((this.at('NAME')) && this.peek(1).type === 'OP' && this.peek(1).value === '='
                && !(this.peek(2).type === 'OP' && this.peek(2).value === '=')) {
              const kw = this.next().value; this.next();
              args.push({ kw, value: this.expr() });
            } else args.push({ kw: null, value: this.expr() });
          } while (this.eat('OP', ','));
        }
        this.expect('OP', ')', '닫는 괄호');
        n = { type: 'Call', func: n, args, line };
      } else if (this.at('OP', '.')) {
        this.next();
        const nt = this.peek();
        if (nt.type !== 'NAME' && nt.type !== 'KEY') this.expect('NAME', undefined, '점(.) 뒤의 이름');
        n = { type: 'Attribute', obj: n, attr: this.next().value, line };
      } else if (this.at('OP', '[')) {
        this.next();
        // 잘라내기(슬라이스) 지원 — 글[2:5], 글[:5], 글[2:]
        const start = this.at('OP', ':') ? null : this.expr();
        if (this.at('OP', ':')) {
          this.next();
          const end = this.at('OP', ']') ? null : this.expr();
          this.expect('OP', ']', '닫는 대괄호');
          n = { type: 'Slice', obj: n, start, end, line };
        } else {
          this.expect('OP', ']', '닫는 대괄호');
          n = { type: 'Subscript', obj: n, index: start, line };
        }
      } else break;
    }
    return n;
  }

  atom() {
    const t = this.peek();
    const line = t.line;
    if (t.type === 'NUMBER') { this.next(); return { type: 'Num', v: t.value.v, float: t.value.float, line }; }
    if (t.type === 'STRING') {
      this.next();
      let s = t.value;
      while (this.at('STRING')) s += this.next().value; // 붙여 쓴 문자열
      return { type: 'Str', v: s, line };
    }
    if (t.type === 'FSTRING') { this.next(); return parseFString(t.value, line); }
    if (t.type === 'NAME') {
      if (t.value === 'lambda') {
        err('지원하지 않는 문법', '이 앱은 lambda 를 지원하지 않습니다', line,
          'def 로 메소드를 정의해 주세요. 교안에서 쓰는 방식입니다.');
      }
      this.next();
      return { type: 'Name', id: t.value, line };
    }
    if (t.type === 'KEY') {
      if (t.value === 'True') { this.next(); return { type: 'Const', v: BOOL(true), line }; }
      if (t.value === 'False') { this.next(); return { type: 'Const', v: BOOL(false), line }; }
      if (t.value === 'None') { this.next(); return { type: 'Const', v: NONE, line }; }
    }
    if (t.type === 'OP' && t.value === '(') {
      this.next();
      if (this.eat('OP', ')')) return { type: 'ListLit', items: [], tuple: true, line };
      const e = this.expr();
      if (this.at('OP', ',')) {
        const items = [e];
        while (this.eat('OP', ',')) { if (this.at('OP', ')')) break; items.push(this.expr()); }
        this.expect('OP', ')', '닫는 괄호');
        return { type: 'ListLit', items, tuple: true, line };
      }
      this.expect('OP', ')', '닫는 괄호');
      return e;
    }
    if (t.type === 'OP' && t.value === '[') {
      this.next();
      if (this.at('OP', ']')) { this.next(); return { type: 'ListLit', items: [], line }; }
      const first = this.expr();
      // 리스트 컴프리헨션 — [토큰 for 토큰 in 토큰들 if 조건]
      if (this.at('KEY', 'for')) {
        this.next();
        const targets = [this.expect('NAME', undefined, '변수 이름').value];
        while (this.eat('OP', ',')) targets.push(this.expect('NAME', undefined, '변수 이름').value);
        this.expect('KEY', 'in', "'in'");
        const iter = this.expr();
        let cond = null;
        if (this.eat('KEY', 'if')) cond = this.expr();
        this.expect('OP', ']', '닫는 대괄호');
        return { type: 'ListComp', elt: first, targets, iter, cond, line };
      }
      const items = [first];
      while (this.eat('OP', ',')) { if (this.at('OP', ']')) break; items.push(this.expr()); }
      this.expect('OP', ']', '닫는 대괄호');
      return { type: 'ListLit', items, line };
    }
    if (t.type === 'OP' && t.value === '{') {
      this.next();
      const pairs = [];
      if (!this.at('OP', '}')) {
        do {
          if (this.at('OP', '}')) break;
          const k = this.expr();
          this.expect('OP', ':', '딕셔너리의 콜론(:)');
          pairs.push([k, this.expr()]);
        } while (this.eat('OP', ','));
      }
      this.expect('OP', '}', '닫는 중괄호');
      return { type: 'DictLit', pairs, line };
    }
    const got = t.type === 'NEWLINE' ? '줄바꿈' : t.type === 'EOF' ? '코드 끝' : `'${t.value}'`;
    err('SyntaxError', `값이 올 자리에 ${got} 이(가) 있습니다`, line);
  }
}

/** f'점수: {score}점' → '점수: ' + str(score) + '점' 으로 바꿔 준다. */
function parseFString(raw, line) {
  const parts = [];
  let buf = '';
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === '{' && raw[i + 1] === '{') { buf += '{'; i++; continue; }
    if (c === '}' && raw[i + 1] === '}') { buf += '}'; i++; continue; }
    if (c === '{') {
      if (buf) { parts.push({ type: 'Str', v: buf, line }); buf = ''; }
      let d = 1, code = '';
      i++;
      while (i < raw.length && d > 0) {
        if (raw[i] === '{') d++;
        if (raw[i] === '}') { d--; if (!d) break; }
        code += raw[i++];
      }
      const cut = code.search(/[:!](?![=])/);
      if (cut > 0) code = code.slice(0, cut); // 서식(:.2f)은 무시
      const inner = new Parser(tokenize(code)).parseProgram();
      const e = inner.body[0] && inner.body[0].type === 'Expr' ? inner.body[0].value : { type: 'Str', v: '', line };
      parts.push({ type: 'Call', func: { type: 'Name', id: 'str', line }, args: [{ kw: null, value: e }], line });
      continue;
    }
    buf += c;
  }
  if (buf) parts.push({ type: 'Str', v: buf, line });
  if (!parts.length) return { type: 'Str', v: '', line };
  return parts.reduce((l, r) => ({ type: 'Bin', op: '+', l, r, line }));
}

/** 소스 → 문법 나무. 평가요소 검사(rubric.js)도 이 나무를 읽는다. */
export function parse(src) {
  return new Parser(tokenize(src)).parseProgram();
}

/* ══════════════════════════════ ③ 실행 ════════════════════════════════════ */

const RETURN = Symbol('return');
const BREAK = Symbol('break');
const CONTINUE = Symbol('continue');
class Signal { constructor(kind, value) { this.kind = kind; this.value = value; } }

/** 비공개 속성 이름 맹글링: 클래스 Car 안의 __speed → _Car__speed */
function mangle(name, clsName) {
  if (!clsName) return name;
  if (!name.startsWith('__')) return name;
  if (name.endsWith('__')) return name; // __init__, __str__ 는 그대로
  return `_${clsName}${name}`;
}
/** 화면에 보여 줄 때는 _Car__speed 를 다시 __speed 로 되돌린다. */
export function unmangle(name) {
  return name.replace(/^_[A-Za-z_가-힣][A-Za-z0-9_가-힣]*__/, '__');
}

/** 파이썬처럼 실수는 5.0, 정수는 5 로 보이게 만든다. */
function numText(x) {
  if (x.k === 'int') return String(x.v);
  if (!isFinite(x.v)) return x.v > 0 ? 'inf' : (Number.isNaN(x.v) ? 'nan' : '-inf');
  if (Number.isInteger(x.v) && Math.abs(x.v) < 1e16) return `${x.v}.0`;
  const s = String(x.v);
  return s.includes('e') ? s : s;
}

export class Interp {
  constructor(opts = {}) {
    this.out = [];              // 출력이 끝난 줄 모음
    this.buf = '';              // 아직 줄바꿈을 만나지 않은 조각 (end='' 대응)
    this.globals = new Map();
    this.instances = [];        // 만들어진 인스턴스 (상자 그림용)
    this.trace = [];            // 메소드 호출 기록
    this.frames = [];           // 호출 중인 메소드 스택 (속성 변화 기록용)
    this.inputs = (opts.inputs || []).slice();
    this.steps = 0;
    this.maxSteps = opts.maxSteps || 400000;
    this.nextId = 1;
    this.depth = 0;
    // 교실용 한국어 분석기를 꽂아 준다 (import korean 으로 쓸 수 있게)
    this.korean = opts.korean || null;
  }

  write(s) {
    this.out.push(s);
    if (this.out.length > 3000) err('RuntimeError', '출력이 너무 많습니다 (3000줄 초과)', 0,
      '반복문이 끝나지 않는지 확인해 보세요.');
  }
  /** 글자를 출력 흐름에 흘려 보낸다. 줄바꿈을 만날 때마다 한 줄이 완성된다. */
  emit(s) {
    this.buf += s;
    let at;
    while ((at = this.buf.indexOf('\n')) >= 0) {
      this.write(this.buf.slice(0, at));
      this.buf = this.buf.slice(at + 1);
    }
  }
  /** 마지막에 남은 조각도 한 줄로 마무리한다. */
  flush() {
    if (this.buf !== '') { this.out.push(this.buf); this.buf = ''; }
  }
  tick(line) {
    if (++this.steps > this.maxSteps) {
      err('RuntimeError', '실행이 너무 오래 걸려 멈췄습니다', line,
        'while 문의 조건이 언제 거짓이 되는지 확인해 보세요. (무한 반복 의심)');
    }
  }

  /* ── 값 → 글자 ── */
  str(x, line) {
    switch (x.k) {
      case 'int': case 'float': return numText(x);
      case 'str': return x.v;
      case 'bool': return x.v ? 'True' : 'False';
      case 'none': return 'None';
      case 'list': return `[${x.v.map((e) => this.repr(e, line)).join(', ')}]`;
      case 'tuple': return `(${x.v.map((e) => this.repr(e, line)).join(', ')}${x.v.length === 1 ? ',' : ''})`;
      // 딕셔너리의 키는 원시값(문자열·숫자)으로 저장되므로 파이썬 값으로 되돌려 찍는다
      case 'dict': return `{${[...x.v.entries()].map(([k, v]) => `${this.repr(wrapKey(k), line)}: ${this.repr(v, line)}`).join(', ')}}`;
      case 'class': return `<class '__main__.${x.name}'>`;
      case 'typeobj': return `<class '${x.name}'>`;
      case 'func': case 'bound': case 'builtin': return `<function ${x.name}>`;
      case 'module': return `<module '${x.name}'>`;
      case 'obj': {
        const m = this.findMethod(x.cls, '__str__');
        if (m) {
          const r = this.callFunc(m, [x], line);
          if (r.k !== 'str') {
            err('TypeError', `__str__ 이 문자열이 아니라 ${this.typeName(r)} 을(를) 돌려주었습니다`, line,
              '__str__ 메소드는 반드시 문자열을 return 해야 합니다. 숫자는 str( ) 로 감싸 주세요.');
          }
          return r.v;
        }
        return `<__main__.${x.cls.name} object at 0x${(0x1a2b000 + x.id * 0x30).toString(16)}>`;
      }
      default: return String(x.v);
    }
  }
  repr(x, line) { return x.k === 'str' ? `'${x.v}'` : this.str(x, line); }
  typeName(x) {
    return { int: 'int', float: 'float', str: 'str', bool: 'bool', none: 'NoneType', list: 'list',
      tuple: 'tuple', dict: 'dict', class: 'type', obj: x.k === 'obj' ? x.cls.name : 'object' }[x.k] || x.k;
  }
  truthy(x) {
    switch (x.k) {
      case 'none': return false;
      case 'bool': return x.v;
      case 'int': case 'float': return x.v !== 0;
      case 'str': return x.v.length > 0;
      case 'list': case 'tuple': return x.v.length > 0;
      case 'dict': return x.v.size > 0;
      default: return true;
    }
  }

  /* ── 프로그램 실행 ── */
  run(src) {
    const ast = parse(src);
    this.execBlock(ast.body, this.globals, null);
    return this.result();
  }

  result() {
    this.flush();
    // 전역 변수 이름을 인스턴스에 붙여 준다 (상자 그림의 이름표)
    for (const inst of this.instances) inst.names = [];
    for (const [name, val] of this.globals) {
      if (val && val.k === 'obj') val.names.push(name);
    }
    return {
      output: this.out.join('\n'),
      lines: this.out.slice(),
      instances: this.instances.map((o) => ({
        id: o.id,
        cls: o.cls.name,
        names: o.names.slice(),
        attrs: [...o.attrs.entries()].map(([k, v]) => ({
          name: unmangle(k),
          private: unmangle(k) !== k || k.startsWith('__'),
          value: this.str(v, 0),
        })),
        methods: [...o.cls.methods.keys()],
      })),
      trace: this.trace,
    };
  }

  execBlock(body, env, clsName) {
    for (const st of body) {
      const sig = this.exec(st, env, clsName);
      if (sig) return sig;
    }
    return null;
  }

  exec(st, env, clsName) {
    this.tick(st.line);
    switch (st.type) {
      case 'Expr': this.eval(st.value, env, clsName); return null;
      case 'Pass': return null;
      case 'Break': return new Signal(BREAK);
      case 'Continue': return new Signal(CONTINUE);
      case 'Return': return new Signal(RETURN, st.value ? this.eval(st.value, env, clsName) : NONE);

      case 'Assign': {
        const v = this.eval(st.value, env, clsName);
        for (const t of st.targets) this.assign(t, v, env, clsName);
        return null;
      }
      case 'AugAssign': {
        const cur = this.eval(st.target, env, clsName);
        const v = this.binop(st.op, cur, this.eval(st.value, env, clsName), st.line);
        this.assign(st.target, v, env, clsName);
        return null;
      }

      case 'If': {
        if (this.truthy(this.eval(st.test, env, clsName))) return this.execBlock(st.body, env, clsName);
        return this.execBlock(st.orelse, env, clsName);
      }
      case 'While': {
        while (this.truthy(this.eval(st.test, env, clsName))) {
          this.tick(st.line);
          const sig = this.execBlock(st.body, env, clsName);
          if (sig) {
            if (sig.kind === BREAK) break;
            if (sig.kind === CONTINUE) continue;
            return sig;
          }
        }
        return null;
      }
      case 'For': {
        const it = this.iterate(this.eval(st.iter, env, clsName), st.line);
        for (const item of it) {
          this.tick(st.line);
          if (st.targets.length === 1) env.set(st.targets[0], item);
          else {
            const parts = item.k === 'list' || item.k === 'tuple' ? item.v : [item];
            st.targets.forEach((n, i) => env.set(n, parts[i] ?? NONE));
          }
          const sig = this.execBlock(st.body, env, clsName);
          if (sig) {
            if (sig.kind === BREAK) break;
            if (sig.kind === CONTINUE) continue;
            return sig;
          }
        }
        return null;
      }

      case 'FuncDef': {
        env.set(st.name, { k: 'func', name: st.name, params: st.params, body: st.body, env, clsName, line: st.line });
        return null;
      }
      case 'ClassDef': {
        const methods = new Map();
        const classEnv = new Map();
        for (const m of st.body) {
          if (m.type === 'FuncDef') {
            methods.set(m.name, { k: 'func', name: m.name, params: m.params, body: m.body, env: classEnv, clsName: st.name, line: m.line });
          } else if (m.type === 'Assign') {
            // 클래스 변수는 교안 범위 밖이지만 오류를 내지 않고 담아 둔다
            this.exec(m, classEnv, st.name);
          }
        }
        env.set(st.name, { k: 'class', name: st.name, methods, classEnv, line: st.line });
        return null;
      }

      case 'Import': {
        for (const n of st.names) {
          const m = this.module(n, st.line);
          env.set(n, m);
        }
        return null;
      }
      case 'FromImport': {
        const m = this.module(st.mod, st.line);
        if (st.names.includes('*')) for (const [k, v] of m.members) env.set(k, v);
        else for (const n of st.names) {
          if (!m.members.has(n)) err('ImportError', `${st.mod} 모듈에 '${n}' 이(가) 없습니다`, st.line);
          env.set(n, m.members.get(n));
        }
        return null;
      }
      default: err('SyntaxError', `아직 지원하지 않는 문장입니다 (${st.type})`, st.line);
    }
  }

  assign(target, value, env, clsName) {
    if (target.type === 'Name') { env.set(target.id, value); return; }
    if (target.type === 'Attribute') {
      const obj = this.eval(target.obj, env, clsName);
      const real = mangle(target.attr, clsName);
      if (obj.k !== 'obj') {
        err('AttributeError', `${this.typeName(obj)} 에는 속성을 새로 만들 수 없습니다`, target.line,
          '속성은 클래스로 만든 인스턴스(객체)에만 붙일 수 있습니다.');
      }
      // 메소드 호출 기록에 "무엇이 어떻게 바뀌었는지" 남긴다
      const frame = this.frames[this.frames.length - 1];
      if (frame && frame.self === obj) {
        const before = obj.attrs.has(real) ? this.str(obj.attrs.get(real), target.line) : '(없음)';
        frame.changes.push({ attr: unmangle(real), before, after: this.str(value, target.line) });
      }
      obj.attrs.set(real, value);
      return;
    }
    if (target.type === 'Subscript') {
      const obj = this.eval(target.obj, env, clsName);
      const idx = this.eval(target.index, env, clsName);
      if (obj.k === 'list') {
        let i = this.intOf(idx, target.line);
        if (i < 0) i += obj.v.length;
        if (i < 0 || i >= obj.v.length) err('IndexError', '리스트 범위를 벗어난 자리입니다', target.line);
        obj.v[i] = value;
        return;
      }
      if (obj.k === 'dict') { obj.v.set(this.key(idx, target.line), value); return; }
      err('TypeError', `${this.typeName(obj)} 에는 [ ] 로 값을 넣을 수 없습니다`, target.line);
    }
  }

  /* ── 식 계산 ── */
  eval(e, env, clsName) {
    this.tick(e.line);
    switch (e.type) {
      case 'Num': return e.float ? FLT(e.v) : INT(e.v);
      case 'Str': return STR(e.v);
      case 'Const': return e.v;
      case 'ListComp': {
        // [바꿀값 for 변수 in 무엇 if 조건] — 일반 for 문과 결과가 똑같다.
        const src = this.iterate(this.eval(e.iter, env, clsName), e.line);
        const out = [];
        for (const item of src) {
          this.tick(e.line);
          if (e.targets.length === 1) env.set(e.targets[0], item);
          else {
            const parts = item.k === 'list' || item.k === 'tuple' ? item.v : [item];
            e.targets.forEach((n, i) => env.set(n, parts[i] ?? NONE));
          }
          if (e.cond && !this.truthy(this.eval(e.cond, env, clsName))) continue;
          out.push(this.eval(e.elt, env, clsName));
        }
        return LIST(out);
      }
      case 'ListLit': {
        const items = e.items.map((x) => this.eval(x, env, clsName));
        return e.tuple ? { k: 'tuple', v: items } : LIST(items);
      }
      case 'DictLit': {
        const m = new Map();
        for (const [k, v] of e.pairs) m.set(this.key(this.eval(k, env, clsName), e.line), this.eval(v, env, clsName));
        return DICT(m);
      }
      case 'Name': {
        if (env.has(e.id)) return env.get(e.id);
        if (this.globals.has(e.id)) return this.globals.get(e.id);
        const b = this.builtin(e.id);
        if (b) return b;
        err('NameError', `이름 '${e.id}' 이(가) 정의되지 않았습니다`, e.line,
          '변수·클래스 이름의 철자와 대소문자를 확인해 보세요. 클래스는 먼저 정의한 뒤에 사용해야 합니다.');
      }
      case 'Not': return BOOL(!this.truthy(this.eval(e.v, env, clsName)));
      case 'Bool': {
        const l = this.eval(e.l, env, clsName);
        if (e.op === 'and') return this.truthy(l) ? this.eval(e.r, env, clsName) : l;
        return this.truthy(l) ? l : this.eval(e.r, env, clsName);
      }
      case 'Unary': {
        const v = this.eval(e.v, env, clsName);
        if (!isNum(v)) err('TypeError', `${this.typeName(v)} 앞에는 - 를 붙일 수 없습니다`, e.line);
        return v.k === 'int' ? INT(-v.v) : FLT(-v.v);
      }
      case 'Bin': return this.binop(e.op, this.eval(e.l, env, clsName), this.eval(e.r, env, clsName), e.line);
      case 'Compare': {
        let left = this.eval(e.left, env, clsName);
        for (const { op, right } of e.ops) {
          const r = this.eval(right, env, clsName);
          if (!this.compare(op, left, r, e.line)) return BOOL(false);
          left = r;
        }
        return BOOL(true);
      }
      case 'Slice': {
        const obj = this.eval(e.obj, env, clsName);
        const len = obj.k === 'str' ? [...obj.v].length
          : (obj.k === 'list' || obj.k === 'tuple') ? obj.v.length : -1;
        if (len < 0) err('TypeError', `${this.typeName(obj)} 은(는) 잘라낼 수 없습니다`, e.line, '문자열과 리스트만 [a:b] 로 자를 수 있습니다.');
        const norm = (node, dflt) => {
          if (!node) return dflt;
          let i = this.intOf(this.eval(node, env, clsName), e.line);
          if (i < 0) i += len;
          return Math.max(0, Math.min(len, i));
        };
        const a = norm(e.start, 0);
        const b = norm(e.end, len);
        if (obj.k === 'str') return STR([...obj.v].slice(a, b).join(''));
        const cut = obj.v.slice(a, b);
        return obj.k === 'tuple' ? { k: 'tuple', v: cut } : LIST(cut);
      }
      case 'Subscript': {
        const obj = this.eval(e.obj, env, clsName);
        const idx = this.eval(e.index, env, clsName);
        if (obj.k === 'str' || obj.k === 'list' || obj.k === 'tuple') {
          let i = this.intOf(idx, e.line);
          const len = obj.k === 'str' ? obj.v.length : obj.v.length;
          if (i < 0) i += len;
          if (i < 0 || i >= len) err('IndexError', `${obj.k === 'str' ? '문자열' : '리스트'} 범위를 벗어난 자리입니다`, e.line);
          return obj.k === 'str' ? STR(obj.v[i]) : obj.v[i];
        }
        if (obj.k === 'dict') {
          const key = this.key(idx, e.line);
          if (!obj.v.has(key)) err('KeyError', `키 ${this.repr(idx, e.line)} 이(가) 없습니다`, e.line);
          return obj.v.get(key);
        }
        err('TypeError', `${this.typeName(obj)} 에는 [ ] 를 쓸 수 없습니다`, e.line);
      }
      case 'Attribute': return this.getAttr(this.eval(e.obj, env, clsName), e.attr, e.line, clsName);
      case 'Call': return this.evalCall(e, env, clsName);
      default: err('SyntaxError', `아직 지원하지 않는 식입니다 (${e.type})`, e.line);
    }
  }

  /* ── 속성 읽기: 여기서 정보 은닉이 살아난다 ── */
  getAttr(obj, attr, line, clsName) {
    const real = mangle(attr, clsName);

    if (obj.k === 'obj') {
      if (obj.attrs.has(real)) return obj.attrs.get(real);
      const m = this.findMethod(obj.cls, real) || this.findMethod(obj.cls, attr);
      if (m) return { k: 'bound', name: attr, self: obj, func: m };
      if (obj.cls.classEnv && obj.cls.classEnv.has(real)) return obj.cls.classEnv.get(real);

      // __로 시작하는 이름을 클래스 밖에서 쓴 경우 → 바로 이 지점이 '정보 은닉'
      const hidden = attr.startsWith('__') && !attr.endsWith('__') && !clsName;
      const bare = attr.replace(/^__/, '');
      const getter = `get${bare.charAt(0).toUpperCase()}${bare.slice(1)}`;
      err('AttributeError', `'${obj.cls.name}' object has no attribute '${attr}'`, line,
        hidden
          ? `${attr} 는 비공개 속성입니다. 클래스 밖에서는 이렇게 꺼낼 수 없으니 ${getter}( ) 같은 접근자 메소드를 만들어 쓰세요.`
          : `${obj.cls.name} 클래스에 '${attr}' 속성이나 메소드가 있는지 확인해 보세요.`);
    }
    if (obj.k === 'class') {
      if (obj.methods.has(attr)) return obj.methods.get(attr);
      if (obj.classEnv && obj.classEnv.has(attr)) return obj.classEnv.get(attr);
      err('AttributeError', `type object '${obj.name}' has no attribute '${attr}'`, line);
    }
    if (obj.k === 'module') {
      if (obj.members.has(attr)) return obj.members.get(attr);
      err('AttributeError', `module '${obj.name}' has no attribute '${attr}'`, line,
        `${obj.name} 모듈에서 이 앱이 지원하는 기능만 쓸 수 있습니다.`);
    }
    const m = this.dataMethod(obj, attr, line);
    if (m) return m;
    err('AttributeError', `'${this.typeName(obj)}' object has no attribute '${attr}'`, line);
  }

  findMethod(cls, name) { return cls.methods.get(name) || null; }

  /* ── 호출 ── */
  evalCall(e, env, clsName) {
    const args = [];
    const kwargs = new Map();
    for (const a of e.args) {
      const v = this.eval(a.value, env, clsName);
      if (a.kw) kwargs.set(a.kw, v); else args.push(v);
    }
    const callee = this.eval(e.func, env, clsName);
    return this.callValue(callee, args, kwargs, e.line, e);
  }

  callValue(callee, args, kwargs, line, node) {
    switch (callee.k) {
      case 'builtin': return callee.fn(args, kwargs, line);
      case 'bound': {
        if (callee.func.k === 'builtin') return callee.func.fn(args, kwargs, line);
        return this.callFunc(callee.func, [callee.self, ...args], line, kwargs);
      }
      case 'func': return this.callFunc(callee, args, line, kwargs);
      case 'class': return this.instantiate(callee, args, kwargs, line);
      default:
        err('TypeError', `${this.typeName(callee)} 은(는) ( ) 로 호출할 수 없습니다`, line,
          node && node.func && node.func.type === 'Attribute'
            ? `'${node.func.attr}' 이(가) 메소드가 아니라 속성일 수 있습니다. 괄호를 빼 보세요.`
            : '함수·메소드·클래스 이름이 맞는지 확인해 보세요.');
    }
  }

  instantiate(cls, args, kwargs, line) {
    const inst = { k: 'obj', cls, attrs: new Map(), id: this.nextId++, names: [] };
    this.instances.push(inst);
    if (this.instances.length > 200) err('RuntimeError', '인스턴스를 너무 많이 만들었습니다 (200개 초과)', line);
    const init = this.findMethod(cls, '__init__');
    if (init) this.callFunc(init, [inst, ...args], line, kwargs, `${cls.name}()`);
    else if (args.length) {
      err('TypeError', `${cls.name}() 는 값을 받지 않습니다`, line,
        `값을 받고 싶다면 def __init__(self, 매개변수) : 형태로 생성자를 만들어야 합니다.`);
    }
    return inst;
  }

  callFunc(fn, args, line, kwargs, label) {
    if (fn.k === 'builtin') return fn.fn(args, kwargs, line);
    if (++this.depth > 60) { this.depth--; err('RecursionError', '메소드가 자기 자신을 너무 깊이 불렀습니다', line); }

    const env = new Map();
    const ps = fn.params;
    const self = ps.length && ps[0].name === 'self' && args[0] && args[0].k === 'obj' ? args[0] : null;

    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (i < args.length) { env.set(p.name, args[i]); continue; }
      if (kwargs && kwargs.has(p.name)) { env.set(p.name, kwargs.get(p.name)); continue; }
      if (p.def) { env.set(p.name, this.eval(p.def, env, fn.clsName)); continue; }
      this.depth--;
      const need = ps.filter((q) => !q.def).length;
      err('TypeError', `${label || fn.name}() 에 넘긴 값의 개수가 맞지 않습니다 (${need}개 필요, ${args.length}개 받음)`, line,
        ps[0] && ps[0].name === 'self'
          ? `self 는 자동으로 넘어갑니다. def ${fn.name}(${ps.map((q) => q.name).join(', ')}) 이므로 괄호 안에 ${Math.max(0, need - 1)}개를 넣어 주세요.`
          : `def ${fn.name}(${ps.map((q) => q.name).join(', ')}) 의 매개변수 개수와 맞춰 주세요.`);
    }
    if (args.length > ps.length) {
      this.depth--;
      err('TypeError', `${label || fn.name}() 에 값을 너무 많이 넘겼습니다 (${ps.length}개 자리에 ${args.length}개)`, line,
        'def 줄의 매개변수 개수를 확인해 보세요.');
    }

    // 메소드 호출 기록 (인스턴스 메소드일 때만)
    let frame = null;
    if (self) {
      frame = {
        self,
        cls: fn.clsName || self.cls.name,
        method: fn.name,
        args: args.slice(1).map((a) => this.repr(a, line)),
        changes: [],
        returned: null,
      };
      this.frames.push(frame);
    }

    let ret = NONE;
    try {
      const sig = this.execBlock(fn.body, env, fn.clsName);
      if (sig && sig.kind === RETURN) ret = sig.value;
    } finally {
      this.depth--;
      if (frame) {
        this.frames.pop();
        frame.returned = ret.k === 'none' ? null : this.repr(ret, line);
        if (this.trace.length < 200) this.trace.push(frame);
      }
    }
    return ret;
  }

  /* ── 연산 ── */
  binop(op, a, b, line) {
    if (op === '+') {
      if (a.k === 'str' && b.k === 'str') return STR(a.v + b.v);
      if (a.k === 'list' && b.k === 'list') return LIST([...a.v, ...b.v]);
      if (a.k === 'str' || b.k === 'str') {
        const other = a.k === 'str' ? b : a;
        err('TypeError', `문자열과 ${this.typeName(other)} 은(는) + 로 이을 수 없습니다`, line,
          `숫자를 문자열에 붙이려면 str( ) 로 감싸 주세요. 예) '점수: ' + str(${this.str(other, line)})`);
      }
    }
    if (isNum(a) && isNum(b)) {
      const x = a.v, y = b.v;
      const bothInt = a.k === 'int' && b.k === 'int';
      switch (op) {
        case '+': return bothInt ? INT(x + y) : FLT(x + y);
        case '-': return bothInt ? INT(x - y) : FLT(x - y);
        case '*': return bothInt ? INT(x * y) : FLT(x * y);
        case '/':
          if (y === 0) err('ZeroDivisionError', 'division by zero', line, '0으로는 나눌 수 없습니다.');
          return FLT(x / y); // 파이썬의 / 는 언제나 실수
        case '//':
          if (y === 0) err('ZeroDivisionError', 'integer division or modulo by zero', line, '0으로는 나눌 수 없습니다.');
          return bothInt ? INT(Math.floor(x / y)) : FLT(Math.floor(x / y));
        case '%':
          if (y === 0) err('ZeroDivisionError', 'integer division or modulo by zero', line, '0으로는 나눌 수 없습니다.');
          return bothInt ? INT(((x % y) + y) % y) : FLT(((x % y) + y) % y);
        case '**': {
          const r = Math.pow(x, y);
          return bothInt && y >= 0 ? INT(r) : FLT(r);
        }
      }
    }
    if (op === '*' && ((a.k === 'str' && b.k === 'int') || (a.k === 'int' && b.k === 'str'))) {
      const s = a.k === 'str' ? a.v : b.v, n = a.k === 'int' ? a.v : b.v;
      return STR(n > 0 ? s.repeat(n) : '');
    }
    if (op === '*' && ((a.k === 'list' && b.k === 'int') || (a.k === 'int' && b.k === 'list'))) {
      const l = a.k === 'list' ? a.v : b.v, n = a.k === 'int' ? a.v : b.v;
      const out = [];
      for (let i = 0; i < n; i++) out.push(...l);
      return LIST(out);
    }
    err('TypeError', `${this.typeName(a)} 와 ${this.typeName(b)} 사이에는 ${op} 연산을 할 수 없습니다`, line,
      a.k === 'obj' || b.k === 'obj'
        ? '객체끼리 계산하려면 접근자 메소드로 속성값을 꺼낸 뒤에 계산해야 합니다.'
        : '두 값의 자료형을 확인해 보세요.');
  }

  compare(op, a, b, line) {
    if (op === 'is') return a === b || (a.k === 'none' && b.k === 'none');
    if (op === 'is not') return !(a === b || (a.k === 'none' && b.k === 'none'));
    if (op === 'in' || op === 'not in') {
      let found = false;
      if (b.k === 'str') found = a.k === 'str' && b.v.includes(a.v);
      else if (b.k === 'list' || b.k === 'tuple') found = b.v.some((x) => this.eq(x, a));
      else if (b.k === 'dict') found = b.v.has(this.key(a, line));
      else err('TypeError', `${this.typeName(b)} 안에서는 in 을 쓸 수 없습니다`, line);
      return op === 'in' ? found : !found;
    }
    if (op === '==') return this.eq(a, b);
    if (op === '!=') return !this.eq(a, b);
    if (isNum(a) && isNum(b)) {
      switch (op) { case '<': return a.v < b.v; case '<=': return a.v <= b.v; case '>': return a.v > b.v; case '>=': return a.v >= b.v; }
    }
    if (a.k === 'str' && b.k === 'str') {
      switch (op) { case '<': return a.v < b.v; case '<=': return a.v <= b.v; case '>': return a.v > b.v; case '>=': return a.v >= b.v; }
    }
    err('TypeError', `${this.typeName(a)} 와 ${this.typeName(b)} 은(는) ${op} 로 크기를 비교할 수 없습니다`, line,
      '숫자끼리 또는 문자열끼리만 크기를 비교할 수 있습니다.');
  }
  eq(a, b) {
    if (isNum(a) && isNum(b)) return a.v === b.v;
    if (a.k === 'bool' && isNum(b)) return (a.v ? 1 : 0) === b.v;
    if (isNum(a) && b.k === 'bool') return a.v === (b.v ? 1 : 0);
    if (a.k !== b.k) return false;
    if (a.k === 'str' || a.k === 'bool') return a.v === b.v;
    if (a.k === 'none') return true;
    if (a.k === 'list' || a.k === 'tuple') return a.v.length === b.v.length && a.v.every((x, i) => this.eq(x, b.v[i]));
    return a === b;
  }
  key(x, line) {
    if (x.k === 'str' || x.k === 'int' || x.k === 'float' || x.k === 'bool') return x.v;
    err('TypeError', `${this.typeName(x)} 은(는) 딕셔너리 키로 쓸 수 없습니다`, line);
  }
  intOf(x, line) {
    if (x.k === 'int' || x.k === 'bool') return x.k === 'bool' ? (x.v ? 1 : 0) : x.v;
    if (x.k === 'float' && Number.isInteger(x.v)) return x.v;
    err('TypeError', '자리 번호(인덱스)는 정수여야 합니다', line);
  }
  iterate(x, line) {
    if (x.k === 'list' || x.k === 'tuple') return x.v.slice();
    if (x.k === 'str') return [...x.v].map(STR);
    if (x.k === 'dict') return [...x.v.keys()].map((k) => (typeof k === 'string' ? STR(k) : INT(k)));
    err('TypeError', `${this.typeName(x)} 은(는) for 문으로 하나씩 꺼낼 수 없습니다`, line,
      'range( ), 리스트, 문자열만 for 문에 쓸 수 있습니다.');
  }

  /* ── 내장 함수 ── */
  builtin(name) {
    const self = this;
    const B = (fn) => ({ k: 'builtin', name, fn });
    const need = (args, n, line) => {
      if (args.length < n) err('TypeError', `${name}() 에 넣을 값이 부족합니다`, line);
    };
    switch (name) {
      case 'print': return B((args, kw, line) => {
        const sep = kw && kw.has('sep') ? self.str(kw.get('sep'), line) : ' ';
        const end = kw && kw.has('end') ? self.str(kw.get('end'), line) : '\n';
        self.emit(args.map((a) => self.str(a, line)).join(sep) + end);
        return NONE;
      });
      case 'len': return B((args, kw, line) => {
        need(args, 1, line);
        const x = args[0];
        if (x.k === 'str' || x.k === 'list' || x.k === 'tuple') return INT(x.v.length);
        if (x.k === 'dict') return INT(x.v.size);
        err('TypeError', `${self.typeName(x)} 의 길이는 구할 수 없습니다`, line);
      });
      case 'str': return B((args, kw, line) => STR(args.length ? self.str(args[0], line) : ''));
      case 'int': return B((args, kw, line) => {
        if (!args.length) return INT(0);
        const x = args[0];
        if (isNum(x)) return INT(Math.trunc(x.v));
        if (x.k === 'bool') return INT(x.v ? 1 : 0);
        if (x.k === 'str') {
          const t = x.v.trim();
          if (!/^[+-]?\d+$/.test(t)) err('ValueError', `invalid literal for int() with base 10: '${x.v}'`, line,
            '숫자로 된 문자열만 int( ) 로 바꿀 수 있습니다.');
          return INT(parseInt(t, 10));
        }
        err('TypeError', `${self.typeName(x)} 은(는) int( ) 로 바꿀 수 없습니다`, line);
      });
      case 'float': return B((args, kw, line) => {
        if (!args.length) return FLT(0);
        const x = args[0];
        if (isNum(x)) return FLT(x.v);
        if (x.k === 'str' && x.v.trim() !== '' && !isNaN(Number(x.v))) return FLT(Number(x.v));
        err('ValueError', `could not convert string to float: '${x.v}'`, line);
      });
      case 'bool': return B((args) => BOOL(args.length ? self.truthy(args[0]) : false));
      case 'abs': return B((args, kw, line) => {
        need(args, 1, line);
        const x = args[0];
        if (!isNum(x)) err('TypeError', 'abs( ) 에는 숫자를 넣어야 합니다', line);
        return x.k === 'int' ? INT(Math.abs(x.v)) : FLT(Math.abs(x.v));
      });
      case 'round': return B((args, kw, line) => {
        need(args, 1, line);
        const x = args[0], d = args[1] ? self.intOf(args[1], line) : 0;
        if (!isNum(x)) err('TypeError', 'round( ) 에는 숫자를 넣어야 합니다', line);
        const f = Math.pow(10, d);
        // 파이썬처럼 .5 는 짝수 쪽으로 (은행가 반올림)
        const scaled = x.v * f;
        let r = Math.round(scaled);
        if (Math.abs(scaled % 1) === 0.5 && r % 2 !== 0) r -= 1;
        return args[1] ? FLT(r / f) : INT(r / f);
      });
      case 'sum': return B((args, kw, line) => {
        need(args, 1, line);
        const l = self.iterate(args[0], line);
        let acc = INT(0);
        for (const v of l) acc = self.binop('+', acc, v, line);
        return acc;
      });
      case 'max': case 'min': return B((args, kw, line) => {
        const list = args.length === 1 ? self.iterate(args[0], line) : args;
        if (!list.length) err('ValueError', `${name}() arg is an empty sequence`, line);
        return list.reduce((a, b) => (self.compare(name === 'max' ? '>' : '<', b, a, line) ? b : a));
      });
      case 'range': return B((args, kw, line) => {
        need(args, 1, line);
        const n = args.map((a) => self.intOf(a, line));
        const [start, stop, step] = n.length === 1 ? [0, n[0], 1] : n.length === 2 ? [n[0], n[1], 1] : n;
        if (step === 0) err('ValueError', 'range() arg 3 must not be zero', line);
        const count = Math.max(0, Math.ceil((stop - start) / step));
        if (count > 200000) err('RuntimeError', 'range( ) 가 너무 큽니다 (20만 초과)', line);
        const out = [];
        for (let i = 0, v = start; i < count; i++, v += step) out.push(INT(v));
        return LIST(out);
      });
      case 'list': return B((args, kw, line) => LIST(args.length ? self.iterate(args[0], line) : []));
      case 'type': return B((args, kw, line) => {
        need(args, 1, line);
        const x = args[0];
        return { k: 'typeobj', name: x.k === 'obj' ? `__main__.${x.cls.name}` : self.typeName(x) };
      });
      case 'input': return B((args, kw, line) => {
        if (args.length) self.emit(self.str(args[0], line)); // 안내 문구는 줄바꿈 없이
        if (!self.inputs.length) {
          err('EOFError', '입력할 값이 준비되어 있지 않습니다', line,
            '아래 「미리 넣어 둘 입력값」 칸에 한 줄씩 값을 적어 두면 input( ) 이 위에서부터 읽어 갑니다.');
        }
        return STR(self.inputs.shift());
      });
      case 'isinstance': return B((args, kw, line) => {
        need(args, 2, line);
        const [x, t] = args;
        if (t.k === 'class') return BOOL(x.k === 'obj' && x.cls === t);
        if (t.k === 'builtin') return BOOL(self.typeName(x) === t.name);
        return BOOL(false);
      });

      /* ── 텍스트 분석 수업에서 꼭 필요한 것들을 더 넣었다 ───────────────── */
      case 'sorted': return B((args, kw, line) => {
        need(args, 1, line);
        const list = self.iterate(args[0], line);
        const keyFn = kw && kw.get('key');
        const rev = kw && kw.get('reverse') ? self.truthy(kw.get('reverse')) : false;
        const keyed = list.map((v) => ({ v, k: keyFn ? self.callValue(keyFn, [v], new Map(), line) : v }));
        keyed.sort((a, bb) => {
          if (self.eq(a.k, bb.k)) return 0;
          return self.compare('<', a.k, bb.k, line) ? -1 : 1;
        });
        if (rev) keyed.reverse();
        return LIST(keyed.map((x) => x.v));
      });
      case 'reversed': return B((args, kw, line) => {
        need(args, 1, line);
        return LIST(self.iterate(args[0], line).reverse());
      });
      case 'enumerate': return B((args, kw, line) => {
        need(args, 1, line);
        const start = args[1] ? self.intOf(args[1], line) : (kw && kw.has('start') ? self.intOf(kw.get('start'), line) : 0);
        return LIST(self.iterate(args[0], line).map((v, i) => ({ k: 'tuple', v: [INT(i + start), v] })));
      });
      case 'zip': return B((args, kw, line) => {
        if (args.length < 2) err('TypeError', 'zip( ) 에는 두 개 이상을 넣어야 합니다', line);
        const lists = args.map((a) => self.iterate(a, line));
        const n = Math.min(...lists.map((l) => l.length));
        const out = [];
        for (let i = 0; i < n; i++) out.push({ k: 'tuple', v: lists.map((l) => l[i]) });
        return LIST(out);
      });
      case 'set': return B((args, kw, line) => {
        // ⚠️ 이 앱의 set( ) 은 「중복을 없앤 리스트」를 돌려준다.
        //    파이썬의 진짜 집합과 달리 순서가 유지되고 리스트처럼 쓸 수 있다.
        //    수업에서 set( ) 을 쓰는 자리(어휘 만들기)에서는 결과가 같다.
        if (!args.length) return LIST([]);
        const out = [];
        for (const v of self.iterate(args[0], line)) {
          if (!out.some((x) => self.eq(x, v))) out.push(v);
        }
        return LIST(out);
      });
      case 'any': case 'all': return B((args, kw, line) => {
        need(args, 1, line);
        const l = self.iterate(args[0], line);
        return BOOL(name === 'any' ? l.some((v) => self.truthy(v)) : l.every((v) => self.truthy(v)));
      });
      case 'tuple': return B((args, kw, line) => ({ k: 'tuple', v: args.length ? self.iterate(args[0], line) : [] }));
      case 'dict': return B((args, kw, line) => {
        const m = new Map();
        if (args.length) {
          for (const pair of self.iterate(args[0], line)) {
            const p = pair.k === 'tuple' || pair.k === 'list' ? pair.v : null;
            if (!p || p.length !== 2) err('TypeError', 'dict( ) 에는 (키, 값) 짝을 넣어야 합니다', line);
            m.set(self.key(p[0], line), p[1]);
          }
        }
        if (kw) for (const [k, v] of kw) m.set(k, v);
        return DICT(m);
      });
      default: return null;
    }
  }

  /* ── 문자열·리스트·딕셔너리의 메소드 ── */
  dataMethod(obj, attr, line) {
    const self = this;
    const B = (fn) => ({ k: 'bound', name: attr, self: obj, func: { k: 'builtin', name: attr, fn } });

    if (obj.k === 'str') {
      const s = obj.v;
      switch (attr) {
        case 'upper': return B(() => STR(s.toUpperCase()));
        case 'lower': return B(() => STR(s.toLowerCase()));
        case 'title': return B(() => STR(s.replace(/\S+/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase())));
        case 'strip': return B(() => STR(s.trim()));
        case 'count': return B((a) => INT(a[0] && a[0].v ? s.split(a[0].v).length - 1 : 0));
        case 'replace': return B((a) => STR(s.split(a[0].v).join(a[1].v)));
        case 'find': return B((a) => INT(s.indexOf(a[0].v)));
        case 'startswith': return B((a) => BOOL(s.startsWith(a[0].v)));
        case 'endswith': return B((a) => BOOL(s.endsWith(a[0].v)));
        case 'split': return B((a) => LIST((a.length ? s.split(a[0].v) : s.trim().split(/\s+/)).filter((x, i, arr) => a.length || x !== '' || arr.length === 1).map(STR)));
        case 'join': return B((a, kw, ln) => STR(self.iterate(a[0], ln).map((x) => {
          if (x.k !== 'str') err('TypeError', 'join( ) 에는 문자열 리스트만 넣을 수 있습니다', ln);
          return x.v;
        }).join(s)));
        case 'isdigit': return B(() => BOOL(/^\d+$/.test(s)));
        case 'format': return B((a) => { let i = 0; return STR(s.replace(/\{\}/g, () => self.str(a[i++] || NONE, line))); });
      }
      return null;
    }
    if (obj.k === 'list') {
      const l = obj.v;
      switch (attr) {
        case 'append': return B((a, kw, ln) => { if (!a.length) err('TypeError', 'append( ) 에 넣을 값이 없습니다', ln); l.push(a[0]); return NONE; });
        case 'pop': return B((a, kw, ln) => {
          if (!l.length) err('IndexError', 'pop from empty list', ln);
          const i = a.length ? self.intOf(a[0], ln) : l.length - 1;
          return l.splice(i < 0 ? i + l.length : i, 1)[0];
        });
        case 'insert': return B((a, kw, ln) => { l.splice(self.intOf(a[0], ln), 0, a[1]); return NONE; });
        case 'remove': return B((a, kw, ln) => {
          const i = l.findIndex((x) => self.eq(x, a[0]));
          if (i < 0) err('ValueError', 'list.remove(x): x not in list', ln);
          l.splice(i, 1); return NONE;
        });
        case 'index': return B((a, kw, ln) => {
          const i = l.findIndex((x) => self.eq(x, a[0]));
          if (i < 0) err('ValueError', `${self.repr(a[0], ln)} is not in list`, ln);
          return INT(i);
        });
        case 'count': return B((a) => INT(l.filter((x) => self.eq(x, a[0])).length));
        case 'sort': return B((a, kw, ln) => { l.sort((x, y) => (self.compare('<', x, y, ln) ? -1 : self.eq(x, y) ? 0 : 1)); return NONE; });
        case 'reverse': return B(() => { l.reverse(); return NONE; });
        case 'clear': return B(() => { l.length = 0; return NONE; });
      }
      return null;
    }
    if (obj.k === 'dict') {
      const d = obj.v;
      const wrap = (k) => (typeof k === 'string' ? STR(k) : typeof k === 'boolean' ? BOOL(k) : Number.isInteger(k) ? INT(k) : FLT(k));
      switch (attr) {
        case 'keys': return B(() => LIST([...d.keys()].map(wrap)));
        case 'values': return B(() => LIST([...d.values()]));
        case 'items': return B(() => LIST([...d.entries()].map(([k, v]) => ({ k: 'tuple', v: [wrap(k), v] }))));
        case 'get': return B((a, kw, ln) => (d.has(self.key(a[0], ln)) ? d.get(self.key(a[0], ln)) : (a[1] || NONE)));
        case 'pop': return B((a, kw, ln) => { const key = self.key(a[0], ln); const v = d.get(key); d.delete(key); return v ?? NONE; });
        // Counter( ) 로 만든 딕셔너리에서 「많이 나온 순서」를 뽑을 때 쓴다
        case 'most_common': return B((a, kw, ln) => {
          const n = a.length ? self.intOf(a[0], ln) : d.size;
          const rows = [...d.entries()]
            .filter(([, v]) => isNum(v))
            .sort((x, y) => y[1].v - x[1].v || String(x[0]).localeCompare(String(y[0]), 'ko'))
            .slice(0, n);
          return LIST(rows.map(([k, v]) => ({ k: 'tuple', v: [wrap(k), v] })));
        });
      }
      return null;
    }
    return null;
  }

  /* ── 모듈 ── */
  module(name, line) {
    const self = this;
    const B = (n, fn) => ({ k: 'builtin', name: n, fn });
    const M = (members) => ({ k: 'module', name, members: new Map(Object.entries(members)) });

    if (name === 'random') {
      return M({
        randint: B('randint', (a, kw, ln) => {
          const lo = self.intOf(a[0], ln), hi = self.intOf(a[1], ln);
          if (hi < lo) err('ValueError', 'empty range for randrange()', ln);
          return INT(lo + Math.floor(Math.random() * (hi - lo + 1)));
        }),
        random: B('random', () => FLT(Math.random())),
        choice: B('choice', (a, kw, ln) => {
          const l = self.iterate(a[0], ln);
          if (!l.length) err('IndexError', 'Cannot choose from an empty sequence', ln);
          return l[Math.floor(Math.random() * l.length)];
        }),
        randrange: B('randrange', (a, kw, ln) => {
          const n = a.map((x) => self.intOf(x, ln));
          const [lo, hi] = n.length === 1 ? [0, n[0]] : n;
          return INT(lo + Math.floor(Math.random() * (hi - lo)));
        }),
        shuffle: B('shuffle', (a, kw, ln) => {
          if (a[0].k !== 'list') err('TypeError', 'shuffle( ) 에는 리스트를 넣어야 합니다', ln);
          const l = a[0].v;
          for (let i = l.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [l[i], l[j]] = [l[j], l[i]]; }
          return NONE;
        }),
      });
    }
    if (name === 'math') {
      const f1 = (n, fn) => B(n, (a, kw, ln) => {
        if (!a.length || !isNum(a[0])) err('TypeError', `math.${n}( ) 에는 숫자를 넣어야 합니다`, ln);
        return FLT(fn(a[0].v));
      });
      return M({
        pi: FLT(Math.PI), e: FLT(Math.E),
        sqrt: B('sqrt', (a, kw, ln) => {
          if (!a.length || !isNum(a[0])) err('TypeError', 'math.sqrt( ) 에는 숫자를 넣어야 합니다', ln);
          if (a[0].v < 0) err('ValueError', 'math domain error', ln, '음수의 제곱근은 구할 수 없습니다.');
          return FLT(Math.sqrt(a[0].v));
        }),
        floor: B('floor', (a, kw, ln) => INT(Math.floor(a[0].v))),
        ceil: B('ceil', (a, kw, ln) => INT(Math.ceil(a[0].v))),
        trunc: B('trunc', (a, kw, ln) => INT(Math.trunc(a[0].v))),
        pow: B('pow', (a, kw, ln) => FLT(Math.pow(a[0].v, a[1].v))),
        sin: f1('sin', Math.sin), cos: f1('cos', Math.cos), tan: f1('tan', Math.tan),
        log: B('log', (a, kw, ln) => FLT(a.length > 1 ? Math.log(a[0].v) / Math.log(a[1].v) : Math.log(a[0].v))),
        log10: f1('log10', Math.log10), fabs: f1('fabs', Math.abs),
      });
    }
    /* ── re : 정규표현식 — 전처리 실습의 핵심 도구 ─────────────────────── */
    if (name === 're') {
      const toRe = (pat, extra) => {
        // 파이썬 표기를 자바스크립트 표기로 살짝 옮긴다
        const src = String(pat).replace(/\(\?P<([A-Za-z_]\w*)>/g, '(?<$1>');
        try { return new RegExp(src, 'g' + (extra || '')); } catch (e2) {
          err('ValueError', `정규표현식을 이해할 수 없습니다: ${pat}`, line, e2.message);
        }
        return null;
      };
      const toRepl = (r) => String(r).replace(/\$/g, '$$$$').replace(/\\g<(\d+)>/g, '$$$1').replace(/\\(\d)/g, '$$$1');
      const S = (a, i, ln) => {
        if (!a[i] || a[i].k !== 'str') err('TypeError', 're 함수에는 문자열을 넣어야 합니다', ln);
        return a[i].v;
      };
      return M({
        sub: B('sub', (a, kw, ln) => STR(S(a, 2, ln).replace(toRe(S(a, 0, ln)), toRepl(S(a, 1, ln))))),
        findall: B('findall', (a, kw, ln) => {
          const out = [];
          const re2 = toRe(S(a, 0, ln));
          let m;
          while ((m = re2.exec(S(a, 1, ln)))) {
            out.push(STR(m.length > 1 ? (m[1] ?? '') : m[0]));
            if (m.index === re2.lastIndex) re2.lastIndex += 1;
          }
          return LIST(out);
        }),
        split: B('split', (a, kw, ln) => LIST(S(a, 1, ln).split(toRe(S(a, 0, ln))).map((x) => STR(x ?? '')))),
        search: B('search', (a, kw, ln) => {
          // 진짜 파이썬은 매치 객체를 돌려주지만, 여기서는 찾은 글자(없으면 None)를 돌려준다.
          const m = toRe(S(a, 0, ln)).exec(S(a, 1, ln));
          return m ? STR(m[0]) : NONE;
        }),
        match: B('match', (a, kw, ln) => {
          const m = toRe('^(?:' + S(a, 0, ln) + ')').exec(S(a, 1, ln));
          return m ? STR(m[0]) : NONE;
        }),
      });
    }

    /* ── collections : Counter — 빈도 세기 ─────────────────────────────── */
    if (name === 'collections') {
      return M({ Counter: B('Counter', (a, kw, ln) => makeCounter(self, a[0], ln)) });
    }

    /* ── korean : 교실용 간이 형태소 분석기 ────────────────────────────── */
    /* ⚠️ 진짜 KoNLPy 가 아니다. 자바 없이 브라우저에서 돌도록 사전과 규칙으로
       만든 아주 작은 분석기이며, 결과가 다를 수 있다는 것을 화면에 적어 두었다. */
    if (name === 'korean' || name === 'konlpy') {
      const KO = self.korean;
      if (!KO) err('ImportError', '이 화면에서는 korean 모듈을 쓸 수 없습니다', line);
      const text = (a, ln) => {
        if (!a.length || a[0].k !== 'str') err('TypeError', '문자열을 넣어야 합니다', ln);
        return a[0].v;
      };
      return M({
        nouns: B('nouns', (a, kw, ln) => LIST(KO.nouns(text(a, ln)).map(STR))),
        morphs: B('morphs', (a, kw, ln) => {
          const stem = kw && kw.has('stem') ? self.truthy(kw.get('stem')) : false;
          return LIST(KO.morphs(text(a, ln), { stem }).map(STR));
        }),
        pos: B('pos', (a, kw, ln) => {
          const stem = kw && kw.has('stem') ? self.truthy(kw.get('stem')) : false;
          return LIST(KO.pos(text(a, ln), { stem }).map((x) => ({ k: 'tuple', v: [STR(x.t), STR(x.p)] })));
        }),
        josa: B('josa', (a, kw, ln) => STR(KO.josa(text(a, ln), a[1] ? a[1].v : '은'))),
      });
    }

    if (name === 'sklearn' || name === 'gensim' || name === 'torch' || name === 'numpy'
      || name === 'pandas' || name === 'nltk' || name === 'langchain' || name === 'transformers') {
      err('ImportError', `이 앱 안에서는 ${name} 을(를) 쓸 수 없습니다`, line,
        '진짜 라이브러리는 브라우저에서 돌지 않습니다. 이 수업에서는 같은 계산을 직접 만들어 봅니다 — '
        + 're, collections(Counter), math, korean 모듈을 쓸 수 있습니다.');
    }
    if (name === 'time' || name === 'datetime' || name === 'turtle' || name === 'tkinter') {
      err('ImportError', `이 앱에서는 ${name} 모듈을 쓸 수 없습니다`, line,
        're, collections, math, random, korean 을 지원합니다.');
    }
    err('ImportError', `'${name}' 모듈을 찾을 수 없습니다`, line,
      're, collections, math, random, korean 을 지원합니다.');
  }
}

/** collections.Counter( ) — 무엇이 몇 번 나왔는지 센 딕셔너리를 만든다. */
function makeCounter(self, arg, line) {
  const m = new Map();
  if (arg) {
    for (const v of self.iterate(arg, line)) {
      const k = self.key(v, line);
      const prev = m.get(k);
      m.set(k, INT((prev ? prev.v : 0) + 1));
    }
  }
  return DICT(m);
}

/** 소스를 실행하고 결과(출력·인스턴스·호출기록·오류)를 돌려준다. */
export function runPython(src, opts = {}) {
  const it = new Interp({ korean: KOREAN, ...opts });
  try {
    return { ok: true, ...it.run(src) };
  } catch (e) {
    if (e instanceof PyError) {
      const partial = it.result();
      return {
        ok: false, error: e.format(), errorType: e.type, errorLine: e.line,
        errorMsg: e.message, errorHint: e.hint, ...partial,
      };
    }
    if (e instanceof RangeError) {
      return { ok: false, error: '실행이 너무 깊어져 멈췄습니다 (메소드가 자기 자신을 계속 부르고 있는지 확인해 보세요)', ...it.result() };
    }
    return { ok: false, error: `예상하지 못한 오류: ${e.message}`, ...it.result() };
  }
}
