import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [{
    name: "bundle-dependency-notices",
    writeBundle(options, bundle) {
      for (const name of Object.keys(bundle)) {
        if (/\.(js|css)$/.test(name)) {
          const path = resolve(options.dir, name);
          writeFileSync(path + ".gz", gzipSync(readFileSync(path)));
        }
      }
      writeFileSync(resolve(options.dir, "protocol.txt"), "4\n");
      const dir = resolve(options.dir, "assets/licenses");
      mkdirSync(dir, {recursive: true});
      for (const [name, files] of [
        ["three", ["LICENSE"]],
        ["lucide", ["LICENSE"]],
        ["gaussian-splat-lite", ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"]],
      ]) {
        for (const file of files) {
          copyFileSync(resolve("node_modules", name, file), resolve(dir, `${name}-${file}`));
        }
      }
    },
  }],
});
