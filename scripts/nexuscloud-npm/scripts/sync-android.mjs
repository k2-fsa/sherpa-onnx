/**
 * Copy Android jniLibs + Tts.kt from @nexuscloud/sherpa-onnx into a Capacitor app.
 *
 * Usage (from consumer app root):
 *   node node_modules/@nexuscloud/sherpa-onnx/scripts/sync-android.mjs
 *   node node_modules/@nexuscloud/sherpa-onnx/scripts/sync-android.mjs --force
 */
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, '..');
const force = process.argv.includes('--force');
const appRoot = process.cwd();

const pkgJson = JSON.parse(readFileSync(path.join(pkgRoot, 'package.json'), 'utf8'));
const version = pkgJson.version;

const srcJni = path.join(pkgRoot, 'android', 'jniLibs');
const srcKotlin = path.join(pkgRoot, 'android', 'kotlin-api', 'Tts.kt');
const destJni = path.join(appRoot, 'android', 'app', 'src', 'main', 'jniLibs');
const destKotlinDir = path.join(
  appRoot,
  'android',
  'app',
  'src',
  'main',
  'java',
  'com',
  'k2fsa',
  'sherpa',
  'onnx',
);
const stampPath = path.join(destJni, '.sherpa-onnx-version');
const assetsVersion = path.join(
  appRoot,
  'android',
  'app',
  'src',
  'main',
  'assets',
  'sherpa-onnx-version.json',
);

function hasExpectedLibs() {
  const abis = ['arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64'];
  const required = ['libsherpa-onnx-jni.so', 'libonnxruntime.so'];
  for (const abi of abis) {
    const dir = path.join(destJni, abi);
    if (!existsSync(dir)) return false;
    for (const lib of required) {
      if (!existsSync(path.join(dir, lib))) return false;
    }
  }
  return true;
}

function copyAbiLibs() {
  if (!existsSync(srcJni)) {
    throw new Error('Package missing android/jniLibs — rebuild @nexuscloud/sherpa-onnx first.');
  }
  for (const abi of readdirSync(srcJni, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)) {
    const src = path.join(srcJni, abi);
    const dest = path.join(destJni, abi);
    mkdirSync(dest, { recursive: true });
    for (const f of readdirSync(dest)) {
      if (f.endsWith('.so')) rmSync(path.join(dest, f), { force: true });
    }
    for (const f of readdirSync(src)) {
      if (!f.endsWith('.so')) continue;
      copyFileSync(path.join(src, f), path.join(dest, f));
    }
    console.log(`[sync-android] ${abi}: copied .so libs`);
  }
}

function copyKotlinApi() {
  if (!existsSync(srcKotlin)) throw new Error('Package missing android/kotlin-api/Tts.kt');
  mkdirSync(destKotlinDir, { recursive: true });
  const banner = `// AUTO-SYNCED from @nexuscloud/sherpa-onnx — DO NOT EDIT
// version: ${version}

`;
  const body = readFileSync(srcKotlin, 'utf8').replace(/^\/\/ AUTO-SYNCED[\s\S]*?\n\n/, '');
  writeFileSync(path.join(destKotlinDir, 'Tts.kt'), banner + body, 'utf8');
  console.log(`[sync-android] Wrote Tts.kt (${version})`);
}

function writeMetadata() {
  writeFileSync(stampPath, `${version}\n`, 'utf8');
  mkdirSync(path.dirname(assetsVersion), { recursive: true });
  writeFileSync(
    assetsVersion,
    JSON.stringify(
      {
        version,
        npmPackage: '@nexuscloud/sherpa-onnx',
        syncedAt: new Date().toISOString(),
        note: 'Android jniLibs + Tts.kt from @nexuscloud/sherpa-onnx (ZipVoice espeakVoice fork)',
      },
      null,
      2,
    ),
    'utf8',
  );
}

const current = existsSync(stampPath) ? readFileSync(stampPath, 'utf8').trim() : '';
if (!force && current === version && hasExpectedLibs() && existsSync(path.join(destKotlinDir, 'Tts.kt'))) {
  console.log(`[sync-android] Already synced to ${version}`);
  process.exit(0);
}

console.log(`[sync-android] Syncing Android → @nexuscloud/sherpa-onnx@${version}`);
copyAbiLibs();
copyKotlinApi();
writeMetadata();
console.log('[sync-android] Done');
