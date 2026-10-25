"""Private stdin/stdout worker for the studio's isolated JavaScript data nodes.

This file is intentionally executable by absolute filename under ``python -I``.
It imports no application module and accepts no Python source or import paths.
The only untrusted source passed to an evaluator is JavaScript inside V8.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


MAX_RESPONSE = 1024 * 1024
MAX_REQUEST = MAX_RESPONSE + 6 * 64 * 1024 + 1024

# Compile untrusted text as a function *argument*, never interpolate it into this
# wrapper's source.  Capture serialization intrinsics before running that body.
# The function constructor gets the global scope, not this trusted closure.
_WRAPPER = r"""
(function(code, payload) {
  'use strict';
  const JSONParse = JSON.parse.bind(JSON), JSONStringify = JSON.stringify.bind(JSON);
  const OwnKeys = Reflect.ownKeys.bind(Reflect);
  const GetDescriptor = Object.getOwnPropertyDescriptor.bind(Object);
  const GetPrototype = Object.getPrototypeOf.bind(Object);
  const Create = Object.create.bind(Object);
  const IsArray = Array.isArray.bind(Array);
  const Finite = Number.isFinite.bind(Number);
  const SameValue = Object.is.bind(Object);
  const Integer = Number.isInteger.bind(Number), SafeInteger = Number.isSafeInteger.bind(Number);
  const RawJSON = typeof JSON.rawJSON === 'function' ? JSON.rawJSON.bind(JSON) : null;
  const BigInteger = BigInt;
  const StringValue = String;
  const Plain = Object.prototype, ArrayProto = Array.prototype;
  const FunctionBody = Function;
  const logs = [];
  let logBytes = 0, values = 0;
  const seen = new Set(), Has = seen.has.bind(seen), Add = seen.add.bind(seen), Delete = seen.delete.bind(seen);
  const forbidden = ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'queueMicrotask',
    'WebAssembly', 'ArrayBuffer', 'SharedArrayBuffer', 'DataView', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
    'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array', 'Float16Array', 'Float32Array', 'Float64Array',
    'BigInt64Array', 'BigUint64Array'];
  for (const name of forbidden) {
    Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
  }
  // Frozen intrinsics prevent a script's prototype changes from affecting the
  // trusted JSON copy.  Per-invocation plain local objects remain mutable.
  for (const value of [Object.prototype, Array.prototype, String.prototype, Number.prototype,
    Boolean.prototype, BigInt.prototype, Function.prototype, Set.prototype, JSON, Reflect, Object, Array, Number, String]) {
    Object.freeze(value);
  }
  function copy(value, depth) {
    if (++values > 500000) throw new Error('Output contains too many values');
    if (depth > 64) throw new Error('Output exceeds maximum nesting depth of 64');
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
    if (typeof value === 'number') {
      if (!Finite(value)) throw new Error('Outputs must contain finite numbers');
      return value;
    }
    if (typeof value !== 'object') throw new Error('Outputs must contain only JSON values');
    if (Has(value)) throw new Error('Output contains a circular reference');
    const array = IsArray(value), prototype = GetPrototype(value);
    if (!array && prototype !== Plain && prototype !== null) throw new Error('Outputs must contain plain JSON objects');
    Add(value);
    const result = array ? [] : Create(null);
    if (array && value.length > 100000) throw new Error('Output array exceeds 100000 entries');
    for (const key of OwnKeys(value)) {
      if (array && key === 'length') continue;
      if (typeof key !== 'string') throw new Error('Output symbol keys are unsupported');
      const descriptor = GetDescriptor(value, key);
      if (!descriptor || !('value' in descriptor)) throw new Error('Output accessors are unsupported');
      if (!descriptor.enumerable) throw new Error('Output contains hidden non-JSON properties');
      if (array && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= value.length))
        throw new Error('Output arrays must contain indexed JSON values only');
      result[key] = copy(descriptor.value, depth + 1);
    }
    if (array) {
      if (OwnKeys(result).length !== value.length + 1) throw new Error('Sparse output arrays are unsupported');
    }
    Delete(value);
    return result;
  }
  function log(level, args) {
    if (logs.length >= 100 || logBytes >= 16384) return;
    let line = level + ': ';
    for (let i = 0; i < args.length && i < 16; ++i) {
      const value = args[i];
      line += (i ? ' ' : '') + (value !== null && typeof value === 'object' ? '[object]' : StringValue(value));
      if (line.length > 2048) break;
    }
    line = line.slice(0, Math.min(2048, 16384 - logBytes));
    logBytes += line.length;
    logs.push(line);
  }
  const console = Object.freeze({
    log: (...args) => log('log', args), info: (...args) => log('info', args),
    warn: (...args) => log('warn', args), error: (...args) => log('error', args)
  });
  const input = JSONParse(payload);
  const output = FunctionBody('inputs', 'params', 'console', '"use strict";\n' + code)(input.inputs, input.params, console);
  if (output === null || typeof output !== 'object' || IsArray(output))
    throw new Error('Script must return an object of named outputs');
  const normalized = copy(output, 0);
  const encoded = JSONStringify({outputs: normalized, logs: logs}, (_key, value) => {
    // The sign of zero is meaningful to this float32 observatory.  Standard
    // stringify would silently replace -0 with the unsigned integer token 0.
    if (typeof value === 'number' && SameValue(value, -0)) {
      if (RawJSON === null) throw new Error('Runtime cannot serialize negative zero exactly');
      return RawJSON('-0.0');
    }
    // Standard JSON.stringify emits the shortest round-tripping Number spelling,
    // e.g. 2**60 -> 1152921504606847000.  Python would parse that token as a
    // different exact integer.  Preserve the Number's actual integral value.
    if (typeof value === 'number' && Integer(value) && !SafeInteger(value)) {
      if (RawJSON === null) throw new Error('Runtime cannot serialize wide integers exactly; return a decimal string');
      return RawJSON(BigInteger(value).toString());
    }
    return value;
  });
  if (encoded.length > 1000000) throw new Error('Script output exceeds 1 MiB');
  return encoded;
})
"""


def _run(request: dict[str, Any], context: Any) -> dict[str, Any]:
    code = request["code"]
    if (not isinstance(code, str) or len(code.encode("utf-8")) > 64 * 1024
            or type(request["timeout_ms"]) is not int or not 10 <= request["timeout_ms"] <= 10000
            or type(request["memory_mb"]) is not int or not 16 <= request["memory_mb"] <= 64):
        raise ValueError("Invalid worker execution limits")
    timeout = request["timeout_ms"] / 1000
    arguments = json.dumps({"inputs": request["inputs"], "params": request["params"]}, allow_nan=False,
                           ensure_ascii=True, separators=(",", ":"))
    if len(arguments.encode("utf-8")) > MAX_RESPONSE:
        raise ValueError("Worker input exceeds 1 MiB")
    context.set_hard_memory_limit(request["memory_mb"] * 1024 * 1024)
    expression = f"{_WRAPPER}({json.dumps(code)}, {json.dumps(arguments)})"
    started = time.monotonic()
    result_json = context.eval(expression, timeout_sec=timeout)
    if not isinstance(result_json, str) or len(result_json.encode("utf-8")) > MAX_RESPONSE - 256:
        raise ValueError("Script output exceeds 1 MiB")
    result = json.loads(result_json)
    result["runtime"] = f"V8 {context.v8_version} / mini-racer"
    result["execution_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


def main() -> int:
    startup_started = time.monotonic()
    stage = "worker bootstrap"

    def phase(name: str) -> None:
        sys.stdout.write(json.dumps({"phase": name}) + "\n")
        sys.stdout.flush()

    try:
        phase(stage)
        # -S skips .pth and sitecustomize execution.  This trusted command-line
        # directory comes from the parent's installed mini-racer module spec,
        # never from plugin data.  Standard-library paths retain precedence.
        if len(sys.argv) != 2:
            raise ValueError("Worker requires its installed runtime package directory")
        runtime = Path(sys.argv[1]).resolve()
        if runtime.name != "py_mini_racer" or not (runtime / "__init__.py").is_file():
            raise ValueError("Worker runtime package directory is invalid")
        sys.path.append(str(runtime.parent))
        stage = "runtime import"
        phase(stage)
        from py_mini_racer import mini_racer  # noqa: PLC0415 -- only inside disposable isolated worker

        stage = "V8 initialization"
        phase(stage)
        with mini_racer() as context:
            context.set_hard_memory_limit(64 * 1024 * 1024)
            initialization_ms = round((time.monotonic() - startup_started) * 1000, 3)
            sys.stdout.write(json.dumps({"ready": True, "initialization_ms": initialization_ms}) + "\n")
            sys.stdout.flush()
            stage = "request decoding"
            raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
            if len(raw) > MAX_REQUEST:
                raise ValueError("Worker request exceeds its size limit")
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("Worker request must be an object")
            stage = "script execution"
            response = {"ok": True, "result": _run(request, context)}
    except Exception as exc:  # The trusted parent receives a bounded error, never a traceback or environment.
        error = f"{stage}: {type(exc).__name__}: {exc}"[:4000]
        response = {"ok": False, "error": error}
    encoded = json.dumps(response, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RESPONSE:
        encoded = b'{"ok":false,"error":"Script output exceeds 1 MiB"}'
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
