# 🌌 Three-Body Problem Simulator

A high-performance, aesthetically pleasing simulation of the notorious Three-Body Problem, featuring both a standalone **Python (Panda3D) application** and an interactive **Web Application (Three.js)**. 

### 🌐 Play the Web Version Here:
[**Launch the Three-Body Web Simulator**](https://skatnagallu.github.io/Three-Body-Problem/three_body_web/index.html)

---

## 🚀 Features

- **15 Stable Periodic Orbits:** Includes mathematically precise initial conditions from the groundbreaking Šuvakov & Dmitrašinović discoveries, including the *Figure-8, Butterfly, Moth, Bumblebee, Yin-Yang, Dragonfly, Goggles, and Yarn* families.
- **Cinematic Aesthetics:** Uses HDR bloom, additive blending, and neon color palettes to create a premium, visually striking deep-space aesthetic. Realistic spectral star classes (M, O/B, and G-class stars) are supported.
- **Interactive "Trisolaris" Drop:** Click anywhere to dynamically inject a fourth mass into the orbital plane via raycasting and watch the chaotic slingshot mechanics unfold.
- **Symplectic Physics:** Built on a custom Velocity Verlet integrator with intensive sub-stepping to guarantee orbit stability over long periods.
- **Background Music:** Features atmospheric background music (*Clair de Lune*) directly integrated into the web client.

---

## 🐍 Python Version (Panda3D)

The standalone Python version offers the same physics and aesthetics but runs locally via the Panda3D engine.

### Prerequisites
- Python 3.x
- Poetry (or standard pip)

### Installation
Clone the repository:
```bash
git clone https://github.com/skatnagallu/Three-Body-Problem.git
cd Three-Body-Problem
```

Install dependencies:
```bash
poetry install
# or
pip install panda3d numpy
```

### Usage
Run the simulation:
```bash
poetry run python three_body_problem/three_body_problem.py
```

### Controls
Use the interactive On-Screen Display (OSD) at the bottom to toggle between modes:
- **Random System:** Spawns a highly chaotic, randomized starting state.
- **Stable Orbits:** A dropdown menu containing 15 distinct stable orbital choreographies.
- **Add Trisolaris:** Dynamically injects a 4th body.
- **Pause/Restart:** Control time and reset simulations.

---

## 🌐 Web Version (Three.js)

The web version requires absolutely no installation and runs flawlessly in any modern web browser. It is fully interactive and supports advanced bloom/HDR rendering natively.

### Local Development
To run the web version locally without GitHub pages, simply use any standard local HTTP server:
```bash
cd three_body_web
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000/` in your browser.

---

## 🤝 Contributions
Contributions, issues, and feature requests are welcome. Feel free to check the issues page or create a pull request.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
