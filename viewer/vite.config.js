import { copyFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [{
    name: "bundle-dependency-notices",
    writeBundle(options) {
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
