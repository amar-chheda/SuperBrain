/* Superbrain deck — Tweaks panel */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "palette": ["#0F1419","#E8B14C","#5BC2A8","#D4D2CC"],
  "monoFont": "'JetBrains Mono'",
  "density": "comfortable",
  "showCode": true
}/*EDITMODE-END*/;

const PALETTES = [
  ["#0F1419","#E8B14C","#5BC2A8","#D4D2CC"], // default deep ink + amber/teal
  ["#1A1A2E","#F5A623","#00C9A7","#E8E6E1"], // brief original
  ["#0B0E10","#D9A03B","#79C2A2","#CFCEC8"], // muted, more print-friendly
  ["#FAF7F0","#1A1A2E","#A36300","#1F8A5B"]  // inverted light
];

const FONTS = [
  { label: "JetBrains Mono", value: "'JetBrains Mono'" },
  { label: "Fira Code",      value: "'Fira Code'" },
  { label: "IBM Plex Mono",  value: "'IBM Plex Mono'" }
];

function SuperbrainTweaks() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  // Apply visual tweaks to the document
  React.useEffect(() => {
    const root = document.documentElement;
    const [bg, amber, teal, text] = t.palette;
    // Slight derive: surface = bg lightened, dim = text darkened
    root.style.setProperty("--bg", bg);
    root.style.setProperty("--amber", amber);
    root.style.setProperty("--teal", teal);
    root.style.setProperty("--text", text);
    // Adjust surface based on bg lightness (rough: lighten by 6%)
    const isLight = /^#[fF]/.test(bg);
    root.style.setProperty("--surface", isLight ? "#F0EDE5" : derive(bg, +0.04));
    root.style.setProperty("--surface-2", isLight ? "#E5E1D6" : derive(bg, +0.07));
    root.style.setProperty("--border", isLight ? "#D6D2C5" : derive(bg, +0.10));
    root.style.setProperty("--muted", isLight ? "#5C5A50" : "#8B8980");
    root.style.setProperty("--dim", isLight ? "#8A8780" : "#5C5A55");

    root.style.setProperty("--font-mono", `${t.monoFont}, ui-monospace, monospace`);

    // Density
    if (t.density === "compact") {
      root.style.setProperty("--type-title", "56px");
      root.style.setProperty("--type-subtitle", "38px");
      root.style.setProperty("--type-body", "30px");
      root.style.setProperty("--type-small", "24px");
      root.style.setProperty("--pad-top", "76px");
      root.style.setProperty("--pad-bottom", "76px");
      root.style.setProperty("--pad-x", "90px");
    } else {
      root.style.setProperty("--type-title", "64px");
      root.style.setProperty("--type-subtitle", "44px");
      root.style.setProperty("--type-body", "34px");
      root.style.setProperty("--type-small", "28px");
      root.style.setProperty("--pad-top", "92px");
      root.style.setProperty("--pad-bottom", "92px");
      root.style.setProperty("--pad-x", "110px");
    }

    document.body.classList.toggle("code-hidden", !t.showCode);
  }, [t]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Palette">
        <TweakColor
          value={t.palette}
          onChange={(v) => setTweak("palette", v)}
          options={PALETTES}
        />
      </TweakSection>

      <TweakSection label="Type">
        <TweakSelect
          label="Mono font"
          value={t.monoFont}
          onChange={(v) => setTweak("monoFont", v)}
          options={FONTS}
        />
        <TweakRadio
          label="Density"
          value={t.density}
          onChange={(v) => setTweak("density", v)}
          options={[
            { label: "Comfortable", value: "comfortable" },
            { label: "Compact",     value: "compact" }
          ]}
        />
      </TweakSection>

      <TweakSection label="Code">
        <TweakToggle
          label="Show code blocks"
          value={t.showCode}
          onChange={(v) => setTweak("showCode", v)}
        />
      </TweakSection>
    </TweaksPanel>
  );
}

// Simple hex lightener — clamps and converts. Operates on #RRGGBB only.
function derive(hex, amount) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  let r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
  const adj = Math.round(255 * amount);
  r = Math.max(0, Math.min(255, r + adj));
  g = Math.max(0, Math.min(255, g + adj));
  b = Math.max(0, Math.min(255, b + adj));
  return "#" + [r, g, b].map(x => x.toString(16).padStart(2, "0")).join("");
}

const tweaksMount = document.createElement("div");
document.body.appendChild(tweaksMount);
ReactDOM.createRoot(tweaksMount).render(<SuperbrainTweaks />);
