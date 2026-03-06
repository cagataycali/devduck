import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";

// ============================================================
// 🎨 Shared styles & helpers
// ============================================================

const COLORS = {
  bg: "#0d1117",
  bgAlt: "#161b22",
  gold: "#FFD700",
  goldDark: "#B8860B",
  white: "#f0f6fc",
  gray: "#8b949e",
  green: "#3fb950",
  blue: "#58a6ff",
  purple: "#bc8cff",
  pink: "#f778ba",
  orange: "#FF6B35",
  cyan: "#79c0ff",
  red: "#f85149",
};

const FONT =
  '"SF Mono", "JetBrains Mono", "Fira Code", "Cascadia Code", monospace';
const SANS = '"Inter", "SF Pro Display", -apple-system, sans-serif';

const ease = (frame: number, from: number, to: number, start: number, end: number) =>
  interpolate(frame, [start, end], [from, to], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.25, 0.1, 0.25, 1),
  });

// ============================================================
// Scene 1: THE LOGO DROP (0-4s) — Duck drops in with "ray tracing" glow
// ============================================================
const SceneLogo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const drop = spring({ frame, fps, config: { damping: 12, stiffness: 80 } });
  const glow = interpolate(frame, [30, 90], [0, 1], { extrapolateRight: "clamp" });
  const textOp = ease(frame, 0, 1, 40, 70);
  const tagOp = ease(frame, 0, 1, 60, 90);
  const shimmer = Math.sin(frame * 0.1) * 0.3 + 0.7;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 40%, #1a1f2e 0%, ${COLORS.bg} 70%)`,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {/* Ambient particles */}
      {Array.from({ length: 20 }).map((_, i) => {
        const x = 15 + (i * 73) % 70;
        const y = 10 + (i * 47) % 80;
        const delay = i * 4;
        const op = interpolate((frame + delay) % 90, [0, 45, 90], [0, 0.6, 0]);
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: `${x}%`,
              top: `${y}%`,
              width: 4,
              height: 4,
              borderRadius: "50%",
              background: COLORS.gold,
              opacity: op * glow,
              boxShadow: `0 0 8px ${COLORS.gold}`,
            }}
          />
        );
      })}

      {/* The Duck */}
      <div
        style={{
          transform: `translateY(${interpolate(drop, [0, 1], [-300, 0])}px) scale(${interpolate(drop, [0, 1], [0.3, 1])})`,
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width={220}
          height={220}
          style={{
            filter: `drop-shadow(0 0 ${20 + shimmer * 30}px ${COLORS.gold}) drop-shadow(0 0 ${60 * glow}px rgba(255,215,0,0.4))`,
          }}
        >
          <defs>
            <radialGradient id="duckFill" cx="40%" cy="30%" r="70%">
              <stop offset="0%" stopColor="#FFE566" />
              <stop offset="50%" stopColor="#FFD700" />
              <stop offset="100%" stopColor="#B8860B" />
            </radialGradient>
          </defs>
          <path
            d="M8.5 5A1.5 1.5 0 0 0 7 6.5 1.5 1.5 0 0 0 8.5 8 1.5 1.5 0 0 0 10 6.5 1.5 1.5 0 0 0 8.5 5M10 2a5 5 0 0 1 5 5c0 1.7-.85 3.2-2.14 4.1 1.58.15 3.36.51 5.14 1.4 3 1.5 4-.5 4-.5s-1 9-7 9H9s-5 0-5-5c0-3 3-4 2-6-4 0-4-3.5-4-3.5 1 .5 2.24.5 3 .15A5.02 5.02 0 0 1 10 2"
            fill="url(#duckFill)"
          />
        </svg>
      </div>

      {/* Title */}
      <div
        style={{
          position: "absolute",
          bottom: 280,
          opacity: textOp,
          fontSize: 82,
          fontFamily: SANS,
          fontWeight: 800,
          color: COLORS.white,
          letterSpacing: -2,
          textShadow: `0 0 40px rgba(255,215,0,${shimmer * 0.5})`,
        }}
      >
        DevDuck
      </div>

      {/* Tagline */}
      <div
        style={{
          position: "absolute",
          bottom: 220,
          opacity: tagOp,
          fontSize: 28,
          fontFamily: SANS,
          color: COLORS.gray,
          letterSpacing: 2,
        }}
      >
        SELF-MODIFYING AI AGENT
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 2: THE PROBLEM (4-10s) — "You" vs "Your agent"
// ============================================================
const SceneProblem: React.FC = () => {
  const frame = useCurrentFrame();

  const youOp = ease(frame, 0, 1, 0, 20);
  const agentOp = ease(frame, 0, 1, 30, 50);
  const vsOp = ease(frame, 0, 1, 15, 35);
  const strikeW = ease(frame, 0, 100, 80, 110);
  const fixOp = ease(frame, 0, 1, 120, 150);

  const problems = [
    "🔧 restart after every change",
    "📦 install deps manually",
    "🧠 forgets everything",
    "🏝️ runs in isolation",
    "⏱️ stops when you stop",
  ];

  return (
    <AbsoluteFill
      style={{
        background: COLORS.bg,
        justifyContent: "center",
        alignItems: "center",
        padding: 100,
      }}
    >
      {/* Left: "Normal agents" */}
      <div
        style={{
          position: "absolute",
          left: 120,
          top: 150,
          opacity: youOp,
        }}
      >
        <div
          style={{
            fontSize: 42,
            fontFamily: SANS,
            fontWeight: 700,
            color: COLORS.red,
            marginBottom: 40,
          }}
        >
          Normal AI Agents
        </div>
        {problems.map((p, i) => {
          const itemOp = ease(frame, 0, 1, 20 + i * 15, 40 + i * 15);
          return (
            <div
              key={i}
              style={{
                fontSize: 30,
                fontFamily: SANS,
                color: COLORS.gray,
                marginBottom: 18,
                opacity: itemOp,
                position: "relative",
              }}
            >
              {p}
              {/* Strike-through */}
              <div
                style={{
                  position: "absolute",
                  top: "50%",
                  left: 0,
                  width: `${strikeW}%`,
                  height: 3,
                  background: COLORS.red,
                  opacity: 0.8,
                }}
              />
            </div>
          );
        })}
      </div>

      {/* VS */}
      <div
        style={{
          position: "absolute",
          fontSize: 70,
          fontFamily: SANS,
          fontWeight: 900,
          color: COLORS.gold,
          opacity: vsOp,
          textShadow: `0 0 30px rgba(255,215,0,0.5)`,
        }}
      >
        vs
      </div>

      {/* Right: DevDuck */}
      <div
        style={{
          position: "absolute",
          right: 120,
          top: 150,
          opacity: agentOp,
        }}
      >
        <div
          style={{
            fontSize: 42,
            fontFamily: SANS,
            fontWeight: 700,
            color: COLORS.green,
            marginBottom: 40,
          }}
        >
          🦆 DevDuck
        </div>
        {[
          "🔥 hot-reloads its own code",
          "📦 installs deps at runtime",
          "🧠 remembers via Knowledge Base",
          "🌐 mesh: CLI + browser + cloud",
          "🌙 thinks while you sleep",
        ].map((p, i) => {
          const itemOp = ease(frame, 0, 1, 50 + i * 15, 70 + i * 15);
          return (
            <div
              key={i}
              style={{
                fontSize: 30,
                fontFamily: SANS,
                color: COLORS.white,
                marginBottom: 18,
                opacity: itemOp,
              }}
            >
              {p}
            </div>
          );
        })}
      </div>

      {/* Bottom fix text */}
      <div
        style={{
          position: "absolute",
          bottom: 100,
          fontSize: 26,
          fontFamily: FONT,
          color: COLORS.gold,
          opacity: fixOp,
        }}
      >
        pipx install devduck && devduck
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 3: TERMINAL MAGIC (10-22s) — Typing animation showing features
// ============================================================
const TerminalLine: React.FC<{
  text: string;
  color?: string;
  delay: number;
  frame: number;
  prefix?: string;
}> = ({ text, color = COLORS.white, delay, frame, prefix = "$ " }) => {
  const localFrame = frame - delay;
  if (localFrame < 0) return null;

  const chars = Math.min(Math.floor(localFrame * 1.5), text.length);
  const displayed = text.substring(0, chars);
  const cursor = localFrame % 16 < 10 && chars < text.length;

  return (
    <div
      style={{
        fontFamily: FONT,
        fontSize: 24,
        color,
        marginBottom: 6,
        opacity: ease(frame, 0, 1, delay, delay + 5),
      }}
    >
      <span style={{ color: COLORS.green }}>{prefix}</span>
      {displayed}
      {cursor && (
        <span
          style={{
            background: COLORS.gold,
            width: 10,
            height: 24,
            display: "inline-block",
            marginLeft: 2,
          }}
        />
      )}
    </div>
  );
};

const SceneTerminal: React.FC = () => {
  const frame = useCurrentFrame();

  const lines: { text: string; color?: string; delay: number; prefix?: string }[] = [
    { text: "devduck", delay: 0 },
    { text: "🦆 Using Bedrock", color: COLORS.gold, delay: 20, prefix: "" },
    { text: '🦆 ✓ Zenoh peer: macbook-a1b2c3', color: COLORS.cyan, delay: 35, prefix: "" },
    { text: '🦆 ✓ WebSocket server: localhost:10001', color: COLORS.cyan, delay: 45, prefix: "" },
    { text: '🦆 ✓ AgentCore proxy: ws://localhost:10000', color: COLORS.cyan, delay: 55, prefix: "" },
    { text: "", delay: 70, prefix: "" },
    { text: '"create a weather tool"', delay: 80 },
    { text: "🦆 Creating ./tools/weather.py...", color: COLORS.gray, delay: 110, prefix: "" },
    { text: "✅ Tool saved → auto-loaded in 0.3s", color: COLORS.green, delay: 140, prefix: "" },
    { text: "", delay: 155, prefix: "" },
    { text: '"deploy to cloud"', delay: 165 },
    { text: "🦆 devduck deploy --name weather-duck --launch", color: COLORS.gray, delay: 195, prefix: "" },
    { text: "✅ Agent deployed to AgentCore!", color: COLORS.green, delay: 230, prefix: "" },
    { text: "", delay: 245, prefix: "" },
    { text: "# Meanwhile, in another terminal...", color: COLORS.gray, delay: 260, prefix: "" },
    { text: "devduck", delay: 280 },
    { text: '🦆 Discovered peer: macbook-a1b2c3 (auto!)', color: COLORS.purple, delay: 300, prefix: "" },
    { text: 'zenoh_peer(action="broadcast", message="sync all")', delay: 320 },
    { text: "📡 2 peers responded — all synced!", color: COLORS.green, delay: 350, prefix: "" },
  ];

  return (
    <AbsoluteFill
      style={{
        background: "#0d1117",
        padding: 60,
        justifyContent: "center",
      }}
    >
      {/* Terminal chrome */}
      <div
        style={{
          background: "#161b22",
          borderRadius: 16,
          border: "1px solid #30363d",
          overflow: "hidden",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        }}
      >
        {/* Title bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            padding: "12px 20px",
            background: "#21262d",
            borderBottom: "1px solid #30363d",
            gap: 8,
          }}
        >
          <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#f85149" }} />
          <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#e3b341" }} />
          <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#3fb950" }} />
          <span style={{ marginLeft: 20, color: COLORS.gray, fontFamily: SANS, fontSize: 16 }}>
            devduck — bash — 80×24
          </span>
        </div>

        {/* Content */}
        <div style={{ padding: "24px 30px", minHeight: 600 }}>
          {lines.map((line, i) => (
            <TerminalLine
              key={i}
              text={line.text}
              color={line.color}
              delay={line.delay}
              frame={frame}
              prefix={line.prefix}
            />
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 4: THE MESH (22-42s) — Funny mesh network visualization
// ============================================================
const SceneMesh: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title
  const titleOp = ease(frame, 0, 1, 0, 20);

  // Nodes
  const nodes = [
    { id: "cli1", label: "🖥️ Terminal 1", x: 300, y: 300, color: COLORS.green, delay: 30 },
    { id: "cli2", label: "🖥️ Terminal 2", x: 300, y: 700, color: COLORS.green, delay: 60 },
    { id: "browser", label: "🌐 Browser", x: 960, y: 200, color: COLORS.blue, delay: 90 },
    { id: "cloud", label: "☁️ AgentCore", x: 1600, y: 300, color: COLORS.purple, delay: 120 },
    { id: "phone", label: "📱 Telegram", x: 1600, y: 700, color: COLORS.pink, delay: 150 },
    { id: "mesh", label: "🕸️ Mesh Relay\n:10000", x: 960, y: 500, color: COLORS.gold, delay: 20 },
  ];

  // Edges (from mesh center)
  const edges = [
    { from: "mesh", to: "cli1", delay: 40 },
    { from: "mesh", to: "cli2", delay: 70 },
    { from: "mesh", to: "browser", delay: 100 },
    { from: "mesh", to: "cloud", delay: 130 },
    { from: "mesh", to: "phone", delay: 160 },
    { from: "cli1", to: "cli2", delay: 80, dashed: true },
  ];

  // Flying messages (the fun part!)
  const messages = [
    { text: "🦆 quack!", from: "cli1", to: "mesh", start: 180, color: COLORS.gold },
    { text: "git pull", from: "mesh", to: "cli2", start: 210, color: COLORS.green },
    { text: "git pull", from: "mesh", to: "cloud", start: 210, color: COLORS.purple },
    { text: "✅ synced", from: "cli2", to: "mesh", start: 260, color: COLORS.green },
    { text: "✅ synced", from: "cloud", to: "mesh", start: 270, color: COLORS.purple },
    { text: "🧠 ring ctx", from: "browser", to: "mesh", start: 310, color: COLORS.blue },
    { text: "📡 broadcast!", from: "mesh", to: "cli1", start: 340, color: COLORS.gold },
    { text: "📡 broadcast!", from: "mesh", to: "cli2", start: 340, color: COLORS.gold },
    { text: "📡 broadcast!", from: "mesh", to: "browser", start: 340, color: COLORS.gold },
    { text: "📡 broadcast!", from: "mesh", to: "cloud", start: 340, color: COLORS.gold },
    { text: "📡 broadcast!", from: "mesh", to: "phone", start: 340, color: COLORS.gold },
    { text: "💬 reply", from: "phone", to: "mesh", start: 380, color: COLORS.pink },
  ];

  const nodeMap: Record<string, { x: number; y: number }> = {};
  nodes.forEach((n) => (nodeMap[n.id] = { x: n.x, y: n.y }));

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, #131820 0%, ${COLORS.bg} 100%)`,
      }}
    >
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 40,
          width: "100%",
          textAlign: "center",
          fontSize: 52,
          fontFamily: SANS,
          fontWeight: 800,
          color: COLORS.white,
          opacity: titleOp,
        }}
      >
        The Unified Mesh{" "}
        <span style={{ color: COLORS.gold }}>— Zero Config</span>
      </div>

      <div
        style={{
          position: "absolute",
          top: 100,
          width: "100%",
          textAlign: "center",
          fontSize: 24,
          fontFamily: SANS,
          color: COLORS.gray,
          opacity: titleOp,
        }}
      >
        Every duck finds every other duck. Automatically. 🦆🦆🦆
      </div>

      {/* SVG for edges */}
      <svg
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
        viewBox="0 0 1920 1080"
      >
        {edges.map((e, i) => {
          const from = nodeMap[e.from];
          const to = nodeMap[e.to];
          const progress = ease(frame, 0, 1, e.delay, e.delay + 20);
          return (
            <line
              key={i}
              x1={from.x}
              y1={from.y}
              x2={from.x + (to.x - from.x) * progress}
              y2={from.y + (to.y - from.y) * progress}
              stroke={COLORS.gray}
              strokeWidth={2}
              strokeDasharray={(e as any).dashed ? "8,8" : "none"}
              opacity={0.4 * progress}
            />
          );
        })}
      </svg>

      {/* Nodes */}
      {nodes.map((n) => {
        const s = spring({
          frame: frame - n.delay,
          fps,
          config: { damping: 14, stiffness: 100 },
        });
        const scale = frame > n.delay ? s : 0;
        const isMesh = n.id === "mesh";
        const pulse = isMesh ? Math.sin(frame * 0.08) * 0.05 + 1 : 1;

        return (
          <div
            key={n.id}
            style={{
              position: "absolute",
              left: n.x - 80,
              top: n.y - 45,
              width: 160,
              textAlign: "center",
              transform: `scale(${scale * pulse})`,
            }}
          >
            <div
              style={{
                background: isMesh ? `${COLORS.gold}22` : `${n.color}15`,
                border: `2px solid ${n.color}`,
                borderRadius: 16,
                padding: "14px 10px",
                boxShadow: `0 0 ${isMesh ? 30 : 15}px ${n.color}40`,
              }}
            >
              <div
                style={{
                  fontSize: isMesh ? 22 : 20,
                  fontFamily: SANS,
                  fontWeight: 700,
                  color: n.color,
                  whiteSpace: "pre-line",
                }}
              >
                {n.label}
              </div>
            </div>
          </div>
        );
      })}

      {/* Flying messages */}
      {messages.map((m, i) => {
        const from = nodeMap[m.from];
        const to = nodeMap[m.to];
        const localFrame = frame - m.start;
        if (localFrame < 0 || localFrame > 40) return null;

        const t = interpolate(localFrame, [0, 30], [0, 1], { extrapolateRight: "clamp" });
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        const op = interpolate(localFrame, [0, 5, 25, 40], [0, 1, 1, 0], {
          extrapolateRight: "clamp",
        });

        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: x - 50,
              top: y - 25,
              fontSize: 18,
              fontFamily: FONT,
              color: m.color,
              background: `${COLORS.bg}dd`,
              padding: "4px 12px",
              borderRadius: 8,
              border: `1px solid ${m.color}40`,
              opacity: op,
              whiteSpace: "nowrap",
            }}
          >
            {m.text}
          </div>
        );
      })}

      {/* Bottom caption */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          width: "100%",
          textAlign: "center",
          fontSize: 22,
          fontFamily: FONT,
          color: COLORS.gray,
          opacity: ease(frame, 0, 1, 350, 380),
        }}
      >
        Zenoh multicast + WebSocket relay + AgentCore = One mesh to rule them all
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 5: AMBIENT MODE (42-55s) — The sleeping duck that keeps working
// ============================================================
const SceneAmbient: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const moonY = ease(frame, -100, 80, 0, 40);
  const moonOp = ease(frame, 0, 1, 10, 30);
  const zOp1 = ease(frame, 0, 1, 60, 80);
  const zOp2 = ease(frame, 0, 1, 80, 100);
  const zOp3 = ease(frame, 0, 1, 100, 120);

  const thoughtOp = ease(frame, 0, 1, 130, 160);
  const thoughts = [
    "🔍 exploring edge cases...",
    "📊 validating assumptions...",
    "🛡️ checking security...",
    "💡 found optimization!",
    "✅ 3 improvements stored",
  ];

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(180deg, #0a0e1a 0%, #151d2e 50%, #1a2332 100%)`,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {/* Stars */}
      {Array.from({ length: 40 }).map((_, i) => {
        const x = (i * 51) % 100;
        const y = (i * 37) % 60;
        const twinkle = interpolate((frame + i * 7) % 60, [0, 30, 60], [0.2, 0.8, 0.2]);
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: `${x}%`,
              top: `${y}%`,
              width: 3,
              height: 3,
              borderRadius: "50%",
              background: "white",
              opacity: twinkle,
            }}
          />
        );
      })}

      {/* Moon */}
      <div
        style={{
          position: "absolute",
          right: 200,
          top: moonY,
          opacity: moonOp,
          fontSize: 120,
        }}
      >
        🌙
      </div>

      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 80,
          width: "100%",
          textAlign: "center",
          fontSize: 48,
          fontFamily: SANS,
          fontWeight: 800,
          color: COLORS.white,
          opacity: ease(frame, 0, 1, 0, 20),
        }}
      >
        You Sleep. Duck Works. 🌙
      </div>

      {/* Sleeping duck */}
      <div
        style={{
          transform: `translateY(${Math.sin(frame * 0.05) * 8}px)`,
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width={160}
          height={160}
          style={{ filter: `drop-shadow(0 0 20px rgba(255,215,0,0.3))` }}
        >
          <defs>
            <radialGradient id="sleepFill" cx="40%" cy="30%" r="70%">
              <stop offset="0%" stopColor="#FFE566" />
              <stop offset="100%" stopColor="#B8860B" />
            </radialGradient>
          </defs>
          <path
            d="M8.5 5A1.5 1.5 0 0 0 7 6.5 1.5 1.5 0 0 0 8.5 8 1.5 1.5 0 0 0 10 6.5 1.5 1.5 0 0 0 8.5 5M10 2a5 5 0 0 1 5 5c0 1.7-.85 3.2-2.14 4.1 1.58.15 3.36.51 5.14 1.4 3 1.5 4-.5 4-.5s-1 9-7 9H9s-5 0-5-5c0-3 3-4 2-6-4 0-4-3.5-4-3.5 1 .5 2.24.5 3 .15A5.02 5.02 0 0 1 10 2"
            fill="url(#sleepFill)"
          />
        </svg>
      </div>

      {/* Z's floating up */}
      <div
        style={{
          position: "absolute",
          left: "54%",
          top: "32%",
          fontSize: 40,
          color: COLORS.blue,
          opacity: zOp1,
          transform: `translateY(${-ease(frame, 0, 60, 60, 200)}px)`,
        }}
      >
        z
      </div>
      <div
        style={{
          position: "absolute",
          left: "57%",
          top: "28%",
          fontSize: 50,
          color: COLORS.blue,
          opacity: zOp2,
          transform: `translateY(${-ease(frame, 0, 80, 80, 220)}px)`,
        }}
      >
        Z
      </div>
      <div
        style={{
          position: "absolute",
          left: "60%",
          top: "24%",
          fontSize: 60,
          color: COLORS.blue,
          opacity: zOp3,
          transform: `translateY(${-ease(frame, 0, 100, 100, 240)}px)`,
        }}
      >
        Z
      </div>

      {/* Thought bubbles */}
      <div
        style={{
          position: "absolute",
          right: 200,
          top: 300,
          opacity: thoughtOp,
          background: `${COLORS.bgAlt}ee`,
          border: `1px solid ${COLORS.blue}40`,
          borderRadius: 16,
          padding: "20px 30px",
          maxWidth: 400,
        }}
      >
        <div
          style={{
            fontSize: 18,
            fontFamily: FONT,
            color: COLORS.blue,
            marginBottom: 12,
          }}
        >
          🌙 [ambient] iteration 3/3
        </div>
        {thoughts.map((t, i) => {
          const tOp = ease(frame, 0, 1, 160 + i * 25, 180 + i * 25);
          return (
            <div
              key={i}
              style={{
                fontSize: 20,
                fontFamily: SANS,
                color: i === thoughts.length - 1 ? COLORS.green : COLORS.gray,
                marginBottom: 8,
                opacity: tOp,
              }}
            >
              {t}
            </div>
          );
        })}
      </div>

      {/* Bottom */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          width: "100%",
          textAlign: "center",
          fontSize: 26,
          fontFamily: SANS,
          color: COLORS.gray,
          opacity: ease(frame, 0, 1, 280, 310),
        }}
      >
        Your background work is injected into your next query 💉
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 6: FEATURE SHOWCASE (55-75s) — Quick hits
// ============================================================
const SceneFeatures: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const features = [
    { icon: "🔥", title: "Hot-Reload", desc: "Edit code → agent restarts instantly", color: COLORS.orange },
    { icon: "🛠️", title: "54+ Tools", desc: "Shell, editor, GitHub, Spotify, macOS...", color: COLORS.green },
    { icon: "🌐", title: "MCP Server", desc: "Works with Claude Desktop natively", color: COLORS.blue },
    { icon: "🧠", title: "Auto-RAG", desc: "Knowledge Base memory across sessions", color: COLORS.purple },
    { icon: "☁️", title: "One-Click Deploy", desc: "devduck deploy --launch → AgentCore", color: COLORS.cyan },
    { icon: "🎬", title: "Time Travel", desc: "Record sessions, resume from any point", color: COLORS.pink },
    { icon: "🔗", title: "Zenoh P2P", desc: "Auto-discover peers, broadcast commands", color: COLORS.gold },
    { icon: "💬", title: "Messaging", desc: "Telegram, Slack, WhatsApp auto-reply", color: COLORS.green },
  ];

  return (
    <AbsoluteFill
      style={{
        background: COLORS.bg,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          fontSize: 48,
          fontFamily: SANS,
          fontWeight: 800,
          color: COLORS.white,
          marginBottom: 60,
          opacity: ease(frame, 0, 1, 0, 15),
        }}
      >
        Everything You Need. Nothing You Don't.
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: 24,
          maxWidth: 1600,
        }}
      >
        {features.map((f, i) => {
          const s = spring({
            frame: frame - i * 8 - 20,
            fps,
            config: { damping: 14, stiffness: 120 },
          });
          const scale = frame > i * 8 + 20 ? s : 0;

          return (
            <div
              key={i}
              style={{
                width: 360,
                padding: "24px 28px",
                background: `${f.color}0a`,
                border: `1px solid ${f.color}30`,
                borderRadius: 16,
                transform: `scale(${scale})`,
              }}
            >
              <div style={{ fontSize: 36, marginBottom: 8 }}>{f.icon}</div>
              <div
                style={{
                  fontSize: 24,
                  fontFamily: SANS,
                  fontWeight: 700,
                  color: f.color,
                  marginBottom: 6,
                }}
              >
                {f.title}
              </div>
              <div
                style={{
                  fontSize: 18,
                  fontFamily: SANS,
                  color: COLORS.gray,
                }}
              >
                {f.desc}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// Scene 7: CTA / OUTRO (75-90s)
// ============================================================
const SceneCTA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const duckBounce = spring({ frame: frame - 10, fps, config: { damping: 8, stiffness: 60 } });
  const cmdOp = ease(frame, 0, 1, 40, 60);
  const linksOp = ease(frame, 0, 1, 80, 110);
  const shimmer = Math.sin(frame * 0.08) * 0.3 + 0.7;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 40%, #1a1f2e 0%, ${COLORS.bg} 70%)`,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {/* Glowing particles */}
      {Array.from({ length: 30 }).map((_, i) => {
        const angle = (i / 30) * Math.PI * 2 + frame * 0.01;
        const radius = 250 + Math.sin(frame * 0.03 + i) * 50;
        const x = 960 + Math.cos(angle) * radius;
        const y = 400 + Math.sin(angle) * radius * 0.6;
        const op = interpolate((frame + i * 8) % 80, [0, 40, 80], [0.1, 0.5, 0.1]);
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: x,
              top: y,
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: COLORS.gold,
              opacity: op,
              boxShadow: `0 0 10px ${COLORS.gold}`,
            }}
          />
        );
      })}

      {/* Duck */}
      <div
        style={{
          transform: `scale(${interpolate(duckBounce, [0, 1], [0, 1])}) translateY(${Math.sin(frame * 0.06) * 10}px)`,
          marginBottom: 30,
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width={180}
          height={180}
          style={{
            filter: `drop-shadow(0 0 ${30 + shimmer * 40}px ${COLORS.gold})`,
          }}
        >
          <defs>
            <radialGradient id="ctaFill" cx="40%" cy="30%" r="70%">
              <stop offset="0%" stopColor="#FFE566" />
              <stop offset="50%" stopColor="#FFD700" />
              <stop offset="100%" stopColor="#B8860B" />
            </radialGradient>
          </defs>
          <path
            d="M8.5 5A1.5 1.5 0 0 0 7 6.5 1.5 1.5 0 0 0 8.5 8 1.5 1.5 0 0 0 10 6.5 1.5 1.5 0 0 0 8.5 5M10 2a5 5 0 0 1 5 5c0 1.7-.85 3.2-2.14 4.1 1.58.15 3.36.51 5.14 1.4 3 1.5 4-.5 4-.5s-1 9-7 9H9s-5 0-5-5c0-3 3-4 2-6-4 0-4-3.5-4-3.5 1 .5 2.24.5 3 .15A5.02 5.02 0 0 1 10 2"
            fill="url(#ctaFill)"
          />
        </svg>
      </div>

      {/* Command */}
      <div
        style={{
          opacity: cmdOp,
          background: `${COLORS.bgAlt}`,
          border: `2px solid ${COLORS.gold}40`,
          borderRadius: 16,
          padding: "20px 50px",
          marginBottom: 40,
        }}
      >
        <span
          style={{
            fontSize: 38,
            fontFamily: FONT,
            color: COLORS.green,
          }}
        >
          ${" "}
        </span>
        <span
          style={{
            fontSize: 38,
            fontFamily: FONT,
            color: COLORS.white,
          }}
        >
          pipx install devduck && devduck
        </span>
      </div>

      {/* Links */}
      <div
        style={{
          opacity: linksOp,
          display: "flex",
          gap: 60,
          fontSize: 26,
          fontFamily: SANS,
        }}
      >
        <div>
          <span style={{ color: COLORS.gray }}>GitHub: </span>
          <span style={{ color: COLORS.blue }}>cagataycali/devduck</span>
        </div>
        <div>
          <span style={{ color: COLORS.gray }}>PyPI: </span>
          <span style={{ color: COLORS.blue }}>pip install devduck</span>
        </div>
        <div>
          <span style={{ color: COLORS.gray }}>Web: </span>
          <span style={{ color: COLORS.blue }}>duck.nyc</span>
        </div>
      </div>

      {/* Tagline */}
      <div
        style={{
          position: "absolute",
          bottom: 100,
          fontSize: 24,
          fontFamily: SANS,
          color: COLORS.gold,
          opacity: ease(frame, 0, 1, 120, 150),
          textShadow: `0 0 20px rgba(255,215,0,0.3)`,
        }}
      >
        One agent. Self-modifying. Mesh-connected. Unstoppable. 🦆
      </div>

      {/* Built with */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          fontSize: 18,
          fontFamily: SANS,
          color: COLORS.gray,
          opacity: ease(frame, 0, 1, 140, 170),
        }}
      >
        Built with Strands Agents SDK
      </div>
    </AbsoluteFill>
  );
};

// ============================================================
// 🎬 MAIN COMPOSITION — Sequence all scenes
// ============================================================
export const DevDuckIntro: React.FC = () => {
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ background: COLORS.bg }}>
      {/* Scene 1: Logo Drop (0-4s = 0-120f) */}
      <Sequence from={0} durationInFrames={fps * 4}>
        <SceneLogo />
      </Sequence>

      {/* Scene 2: The Problem (4-10s = 120-300f) */}
      <Sequence from={fps * 4} durationInFrames={fps * 6}>
        <SceneProblem />
      </Sequence>

      {/* Scene 3: Terminal Magic (10-22s = 300-660f) */}
      <Sequence from={fps * 10} durationInFrames={fps * 12}>
        <SceneTerminal />
      </Sequence>

      {/* Scene 4: The Mesh (22-42s = 660-1260f) */}
      <Sequence from={fps * 22} durationInFrames={fps * 20}>
        <SceneMesh />
      </Sequence>

      {/* Scene 5: Ambient Mode (42-55s = 1260-1650f) */}
      <Sequence from={fps * 42} durationInFrames={fps * 13}>
        <SceneAmbient />
      </Sequence>

      {/* Scene 6: Feature Showcase (55-75s = 1650-2250f) */}
      <Sequence from={fps * 55} durationInFrames={fps * 20}>
        <SceneFeatures />
      </Sequence>

      {/* Scene 7: CTA / Outro (75-90s = 2250-2700f) */}
      <Sequence from={fps * 75} durationInFrames={fps * 15}>
        <SceneCTA />
      </Sequence>
    </AbsoluteFill>
  );
};
