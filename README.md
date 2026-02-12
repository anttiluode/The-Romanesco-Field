# The Romanesco Field

👉 **Live Demo:** https://anttiluode.github.io/The-Romanesco-Field/

This repo contains two separate things that share a canvas but are functionally independent:

1. **A multi-scale oscillatory field visualization** (the Romanesco field)
2. **A heuristic 3-SAT solver** with a WebGL background animation

They were developed in sequence — the field came first, the solver grew out of it through iteration. The connection between them is aesthetic, not computational.

---

## Part 1: The Romanesco Field (`index.html`)

A GPU fragment shader that renders five coupled oscillatory bands in real time.

Each band has its own frequency and spatial wavelength. Slower bands modulate faster ones through phase-amplitude coupling (PAC) — the same cross-frequency interaction observed in neural oscillations. The result is nested, fractal-like interference patterns that look organic because biology operates under the same scale constraints.

**What's in the shader:**

- 5 frequency bands (delta through gamma) with Bessel-like radial eigenmodes
- PAC chain: delta gates theta, theta gates alpha, alpha gates beta, beta gates gamma
- Sigmoid nonlinearity on fast bands (creates sharp transitions)
- Moiré stress computed as cross-band interference products
- 9 visualization modes: composite, per-band isolation, moiré, phase coherence, coupling strength

**What the sliders control:**

- Per-band amplitude
- PAC coupling strength
- Spatial scale, time speed, nonlinearity, moiré zoom

**What it is not:** It is not a brain simulator, not EEG analysis, not AI. It is a visualization of known signal processing principles — specifically, what happens when incompatible spatial scales are forced to coexist on a single grid.

---

## Part 2: The SAT Solver (`solver2.html` through `solver7.html`)

A stochastic local search solver for 3-SAT problems, presented with a WebGL spiral animation. You drop a JSON file containing clauses and it attempts to find a satisfying Boolean assignment.

### How it works

Each Boolean variable is encoded as a phase angle on a circle. `cos(phase) > 0` means TRUE, `cos(phase) < 0` means FALSE. On each frame:

1. Every clause is evaluated against the current phase assignment
2. Unsatisfied clauses apply torque to a randomly chosen literal, pushing its phase toward a satisfying orientation
3. Phases are integrated with noise
4. If stagnation is detected (300 frames without improvement), all phases are randomly perturbed (thermal restart)
5. When all clauses read as satisfied, a **deterministic Boolean verifier** checks every clause. Only if this passes does it report "CERTIFIED"

This is a variant of [WalkSAT](https://en.wikipedia.org/wiki/WalkSAT) (Selman et al., 1994) with continuous phase variables instead of discrete bit flips, plus random-walk noise and restart-on-stagnation.

### What the verification means

When the solver displays **Verified: YES**, the solution is mathematically correct. The verifier is a standard SAT certificate checker — it evaluates every clause against the final Boolean assignment. This is the textbook reason SAT is in NP: solutions are hard to find but easy to check.

### What it does not mean

- This does not prove P = NP
- This does not guarantee convergence on hard instances
- The solver will struggle at the SAT phase transition (~4.27 clauses per variable for random 3-SAT)
- The WebGL animation is cosmetic — it receives `satisfaction` and `calcium` as uniforms but does not participate in the computation

### Evolution of the solver files

| File | What changed |
|------|-------------|
| `solver2.html` | Fake solver. Satisfaction climbs on a timer. No clause evaluation. Pretty animation. |
| `solver3.html` | Added JSON drop zone. Still fake — satisfaction is a timer, not computed from clauses. |
| `solver4.html` | Added proof output. **Still fake** — truth values are `Math.random()`, not derived from any computation. |
| `solver5.html` | **First real solver.** Clauses are evaluated, unsatisfied clauses apply torque, phases are integrated. No verification pass. |
| `solver6.html` | Added deterministic verification (`verifyAssignment()`). Only reports "CERTIFIED" if the verifier passes. First honest version. |
| `solver7.html` | Added stagnation detection and automatic thermal restarts. The most complete version. |

### JSON format

```json
{
  "n_vars": 20,
  "clauses": [
    [1, 2, 3],
    [-1, -2, -3],
    [2, 3, 4]
  ]
}
```

Positive integers are positive literals, negative integers are negated literals. Each clause is an OR of its literals. The solver tries to make every clause true simultaneously.

### Test problems included

- `problem.json` — 30 variables, 129 clauses (near phase transition ratio)
- `problem2.json` — 81 variables, 24 clauses (heavily underconstrained, easy)
- `problem3.json` — 20 variables, 22 clauses (adversarial bottleneck structure)

---

## Honest assessment

**The Romanesco field** is a genuinely rich visualization instrument. Five independent bands with real PAC coupling, nine diagnostic views, and full parametric control. It demonstrates how nested oscillations produce emergent structure through interference — the same principle that operates in neural dynamics.

**The SAT solver** is a working heuristic solver with correct verification. It is not novel — WalkSAT has existed since 1994 and production solvers like MiniSat or CaDiCaL handle millions of variables. The solver works on small, underconstrained instances. It will fail or take a very long time on hard instances at the phase transition.

**The connection between them** is that they share a WebGL canvas. The README previously described a "Phase-Calcium-Latent transduction loop" where the field dynamics participate in the solving. That is not what happens in the code. The shader receives two numbers (`satisfaction` and `globalCa`) and renders a spiral. The solver runs independently in JavaScript. They do not interact.

---

## How to run

1. Open any `.html` file in a modern browser (Chrome, Edge, Firefox)
2. For the solver files, drag and drop a `.json` problem file onto the drop zone
3. No server required — everything runs client-side

---

## License

Do whatever you want with it.
