import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    include: ["react", "react-dom", "react-dom/client", "react/jsx-runtime"],
  },
  build: {
    // sherpa-onnx-wasm-main-fa.data can be large; keep the default chunking.
    chunkSizeWarningLimit: 4096,
  },
});
