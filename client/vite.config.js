import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import process from "process";
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 4000,
  },
  define: {
    "process.env": {}, // Define process.env for compatibility
  },
  resolve: {
    alias: {
      process: "process/browser", // Polyfill for process
    },
  },
});
