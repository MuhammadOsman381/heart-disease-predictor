/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "rgb(var(--background) / <alpha-value>)",
        foreground: "rgb(var(--foreground) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        card: "rgb(var(--card) / <alpha-value>)",
        panel: "rgb(var(--panel) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        primary: "rgb(var(--primary) / <alpha-value>)",
        primarySoft: "rgb(var(--primary-soft) / <alpha-value>)",
        success: "rgb(var(--success) / <alpha-value>)",
        warning: "rgb(var(--warning) / <alpha-value>)",
        danger: "rgb(var(--danger) / <alpha-value>)"
      },
      boxShadow: {
        glow: "0 14px 40px rgb(var(--primary) / 0.12)",
        soft: "0 10px 30px rgb(15 23 42 / 0.06)"
      },
      keyframes: {
        shimmer: {
          "100%": { transform: "translateX(100%)" }
        }
      },
      animation: {
        shimmer: "shimmer 1.8s infinite"
      }
    }
  },
  plugins: []
};
