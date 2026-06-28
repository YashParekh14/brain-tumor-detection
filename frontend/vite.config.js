import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to FastAPI backend
    // So frontend (5173) can talk to API (8000) without CORS issues
    proxy: {
      '/predict': 'http://localhost:8000',
      '/health':  'http://localhost:8000',
    }
  }
})
