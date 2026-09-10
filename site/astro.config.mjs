import { defineConfig } from "astro/config";
export default defineConfig({
  site: "https://walker.purr.io",
  // keep whitespace between inline elements ("a <strong>referee</strong> that")
  compressHTML: false,
});
