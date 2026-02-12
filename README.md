# Romanesco Consciousness Field

This project is a **single-page WebGL visualization** that renders animated, organic, fractal-like graphics inspired by brain waves, phase coupling, and nested oscillations — visually similar to Romanesco broccoli patterns.

It runs entirely in the browser using **HTML + JavaScript + a GPU fragment shader**.

---

## What this is (simple version)

Think of this as:

- A **screensaver-like field**
- Made of **multiple moving waves**
- Where **slow waves modulate faster waves**
- And everything is drawn **pixel-by-pixel on the GPU**

No data is being loaded.  
No AI is running.  
Nothing is being “learned”.  

It’s pure math + graphics.

---

## Core idea

The code simulates **five oscillating fields** layered together:

| Band | Name  | Rough meaning |
|----|------|---------------|
| 0 | Delta | Very slow, large-scale background |
| 1 | Theta | Medium-slow modulation |
| 2 | Alpha | Carrier rhythm |
| 3 | Beta  | Fast detail |
| 4 | Gamma | Very fast, fine texture |

Each band:
- Has its own **frequency**
- Has its own **spatial scale**
- Is shaped as **concentric / spiral eigenmodes**
- Is **amplitude-modulated by the slower band below it**

This nesting is what creates the **Romanesco / fractal feel**.

---

## Why the graphics look “alive”

Three main reasons:

### 1. Phase–Amplitude Coupling (PAC)

Slower waves control how strong faster waves become.

```text
slow phase ↑ → fast detail grows
slow phase ↓ → fast detail fades
```

This creates pulsing, breathing structures.

---

### 2. Moiré interference

When multiple wave patterns overlap slightly out of sync, they form:

- ripples
- lattices
- rotating flower shapes
- emergent textures

These are **not drawn explicitly** — they appear naturally from interference.

---

### 3. GPU fragment shader

The core math runs **once per pixel, 60 times per second** on your GPU.

That’s why:
- It’s fast
- It’s fluid
- It feels continuous instead of frame-based

---

## What the modes do

The buttons on the right change what you’re looking at:

- **Composite** – All bands combined into RGB
- **Delta → Gamma** – View one band alone
- **Moiré** – Shows interference stress between bands
- **Phase** – Shows how synchronized the bands are
- **Coupling** – Visualizes modulation strength

They’re all the *same field*, just viewed differently.

---

## What the sliders do

- **Band amplitudes** – How strong each layer is
- **PAC strength** – How much slow waves affect fast ones
- **Spatial scale** – Zoom level of the field
- **Time speed** – Animation speed
- **Nonlinearity** – Sharpening / thresholding
- **Moiré zoom** – How dense interference patterns are

Small changes can cause **qualitative shifts**, which is why it feels unstable or surprising.

---

## What this is NOT

- ❌ Not a brain simulator
- ❌ Not EEG decoding
- ❌ Not consciousness itself
- ❌ Not AI

It’s a **visual metaphor** built from known signal principles.

---

## How to run

1. Save the file as `romanesco.html`
2. Open it in a modern browser (Chrome, Edge, Firefox)
3. Make sure WebGL is enabled
4. Move sliders, switch modes, watch it breathe

No server required.

---

## Why it matters

This kind of code shows how:

- Simple oscillators
- Coupling rules
- And spatial structure

Can create **rich, lifelike behavior** without agents, symbols, or learning.

Fields alone are enough.

---

## License

Do whatever you want with it.
Explore. Break it. Remix it.
