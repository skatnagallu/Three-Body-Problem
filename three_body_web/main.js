import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ---- Physics Engine (Symplectic Velocity Verlet) ----
class PhysicsEngine {
    constructor() {
        this.G = 1.0;
        this.timeScale = 1.0;
        this.positions = [];
        this.velocities = [];
        this.accelerations = [];
        this.masses = [];
        this.numBodies = 0;
    }

    init(mode) {
        this.positions = [];
        this.velocities = [];
        this.masses = [];
        
        if (mode === 'random') {
            this.G = 1.0;
            this.timeScale = 0.1;
            this.numBodies = 3;
            for(let i=0; i<3; i++) this.masses.push(1.0);

            let totalMass = 0;
            let cmVelocity = new THREE.Vector3();
            
            for(let i=0; i<this.numBodies; i++) {
                let pos = new THREE.Vector3(
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10
                );
                let vel = new THREE.Vector3(
                    (Math.random() - 0.5) * 1.5,
                    (Math.random() - 0.5) * 1.5,
                    (Math.random() - 0.5) * 1.5
                );
                
                this.positions.push(pos);
                this.velocities.push(vel);
                
                cmVelocity.addScaledVector(vel, this.masses[i]);
                totalMass += this.masses[i];
            }
            cmVelocity.divideScalar(totalMass);
            for(let i=0; i<this.numBodies; i++) {
                this.velocities[i].sub(cmVelocity);
            }
        } 
        else if (orbits[mode]) {
            let o = orbits[mode];
            this.G = 1.0;
            this.timeScale = 0.15;
            this.numBodies = 3;
            this.masses = [1.0, 1.0, 1.0];
            
            this.positions = [
                new THREE.Vector3(-1, 0, 0).multiplyScalar(2),
                new THREE.Vector3(1, 0, 0).multiplyScalar(2),
                new THREE.Vector3(0, 0, 0)
            ];
            
            this.velocities = [
                new THREE.Vector3(o.vx, o.vy, 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(o.vx, o.vy, 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(-2 * o.vx, -2 * o.vy, 0).divideScalar(Math.sqrt(2))
            ];
        }
        else if (mode === 'lagrange') {
            this.G = 1.0;
            this.timeScale = 0.15;
            this.numBodies = 3;
            this.masses = [1.0, 1.0, 1.0];
            
            let r = 1.0;
            let v_mag = Math.sqrt(1 / Math.sqrt(3));
            
            this.positions = [
                new THREE.Vector3(r, 0, 0).multiplyScalar(2),
                new THREE.Vector3(-r/2, r*Math.sqrt(3)/2, 0).multiplyScalar(2),
                new THREE.Vector3(-r/2, -r*Math.sqrt(3)/2, 0).multiplyScalar(2)
            ];
            
            this.velocities = [
                new THREE.Vector3(0, v_mag, 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(-v_mag*Math.sqrt(3)/2, -v_mag/2, 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(v_mag*Math.sqrt(3)/2, -v_mag/2, 0).divideScalar(Math.sqrt(2))
            ];
        }
        
        this.accelerations = this.computeAccelerations(this.positions);
    }

    computeAccelerations(positions) {
        let acc = Array(this.numBodies).fill().map(() => new THREE.Vector3());
        
        for (let i = 0; i < this.numBodies; i++) {
            for (let j = 0; j < this.numBodies; j++) {
                if (i === j) continue;
                
                let diff = new THREE.Vector3().subVectors(positions[j], positions[i]);
                let distSq = diff.lengthSq();
                let dist = Math.sqrt(distSq);
                
                // Softening to prevent singularity explosions
                if (dist < 0.1) distSq = 0.01;
                
                let forceMag = (this.G * this.masses[j]) / (distSq * dist);
                acc[i].add(diff.multiplyScalar(forceMag));
            }
        }
        return acc;
    }

    step(dt) {
        let scaledDt = dt * this.timeScale * 10.0;
        if (scaledDt > 0.1) scaledDt = 0.1; // clamp to prevent explosion
        
        // Velocity Verlet
        for (let i = 0; i < this.numBodies; i++) {
            this.positions[i].add(this.velocities[i].clone().multiplyScalar(scaledDt))
                           .add(this.accelerations[i].clone().multiplyScalar(0.5 * scaledDt * scaledDt));
        }
        
        let newAcc = this.computeAccelerations(this.positions);
        
        for (let i = 0; i < this.numBodies; i++) {
            let avgAcc = new THREE.Vector3().addVectors(this.accelerations[i], newAcc[i]).multiplyScalar(0.5);
            this.velocities[i].add(avgAcc.multiplyScalar(scaledDt));
        }
        
        this.accelerations = newAcc;
    }

    getBarycenter() {
        let center = new THREE.Vector3();
        let totalMass = 0;
        for (let i = 0; i < this.numBodies; i++) {
            center.addScaledVector(this.positions[i], this.masses[i]);
            totalMass += this.masses[i];
        }
        if (totalMass > 0) center.divideScalar(totalMass);
        return center;
    }
}


// ---- Graphics Engine ----
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020204);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Post-processing Bloom
const renderScene = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 3.5, 0.4, 0.0);
const composer = new EffectComposer(renderer);
composer.addPass(renderScene);
composer.addPass(bloomPass);

// App State
const orbits = {
    'figure8': { vx: 0.347111, vy: 0.532728 },
    'butterfly_I': { vx: 0.30689, vy: 0.12551 },
    'butterfly_II': { vx: 0.39295, vy: 0.09758 },
    'bumblebee': { vx: 0.18428, vy: 0.58719 },
    'moth_I': { vx: 0.46444, vy: 0.39606 },
    'moth_II': { vx: 0.43917, vy: 0.45297 },
    'butterfly_III': { vx: 0.40592, vy: 0.23016 },
    'moth_III': { vx: 0.38344, vy: 0.37736 },
    'goggles': { vx: 0.08330, vy: 0.12789 },
    'butterfly_IV': { vx: 0.350112, vy: 0.07934 },
    'dragonfly': { vx: 0.08058, vy: 0.58884 },
    'yarn': { vx: 0.55906, vy: 0.34919 },
    'yin_yang_I': { vx: 0.51394, vy: 0.30474 },
    'yin_yang_II': { vx: 0.41682, vy: 0.33033 }
};

let mode = 'random';
let physics = new PhysicsEngine();
let bodyMeshes = [];
let trailLines = [];
let trailPointsList = [];

// Colors for visual flair (Realistic Star Spectral Types)
const colors = [
    new THREE.Color(0xff4444), // Red Giant (M-Class)
    new THREE.Color(0x4488ff), // Blue Giant (O/B-Class)
    new THREE.Color(0xffdd88)  // Yellow-White (G-Class, like our Sun)
];

function initSimulation() {
    // Clear old
    bodyMeshes.forEach(m => scene.remove(m));
    trailLines.forEach(t => scene.remove(t));
    bodyMeshes = [];
    trailLines = [];
    trailPointsList = [];
    
    physics.init(mode);
    
    const baseScale = mode === 'random' ? 0.3 : 0.15;
    
    for (let i = 0; i < physics.numBodies; i++) {
        // Create Body
        const geo = new THREE.SphereGeometry(1, 32, 32);
        const mat = new THREE.MeshBasicMaterial({ color: colors[i] });
        const mesh = new THREE.Mesh(geo, mat);
        
        let scale = (i === 3) ? baseScale * 0.4 : baseScale;
        mesh.scale.set(scale, scale, scale);
        
        // Add point light to make them cast light
        const light = new THREE.PointLight(colors[i], 2, 50);
        mesh.add(light);
        
        scene.add(mesh);
        bodyMeshes.push(mesh);
        
        // Create Trail
        const maxTrail = 3000;
        trailPointsList.push({ points: [], max: maxTrail });
        
        const lineGeo = new THREE.BufferGeometry();
        const lineMat = new THREE.LineBasicMaterial({ 
            color: colors[i].clone().multiplyScalar(1.5), // HDR Boost for bloom
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        const line = new THREE.Line(lineGeo, lineMat);
        line.frustumCulled = false;
        scene.add(line);
        trailLines.push(line);
    }
    
    // Setup initial camera
    const center = physics.getBarycenter();
    camera.position.set(center.x, center.y - 15, center.z + 10);
    controls.target.copy(center);
    
    document.getElementById('loading').style.display = 'none';
}

// Main Loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    let dt = clock.getDelta();
    if (dt > 0.1) dt = 0.1;
    
    // Sub-stepping for stable orbits
    const subSteps = 10;
    for(let s=0; s<subSteps; s++) {
        physics.step(dt / subSteps);
    }
    
    const center = physics.getBarycenter();
    controls.target.lerp(center, 0.05); // Smoothly track barycenter
    controls.update();

    for (let i = 0; i < physics.numBodies; i++) {
        const p = physics.positions[i];
        bodyMeshes[i].position.copy(p);
        
        // Update Trail
        const trailData = trailPointsList[i];
        trailData.points.push(p.clone());
        if (trailData.points.length > trailData.max) {
            trailData.points.shift();
        }
        
        trailLines[i].geometry.setFromPoints(trailData.points);
    }
    
    composer.render();
}

// UI Setup
function setMode(newMode) {
    mode = newMode;
    document.getElementById('btn-random').classList.remove('active');
    if (newMode === 'random') {
        document.getElementById('btn-random').classList.add('active');
        document.getElementById('stable-orbits').value = "";
    } else {
        document.getElementById('stable-orbits').value = newMode;
    }
    initSimulation();
}

document.getElementById('btn-random').onclick = () => setMode('random');
document.getElementById('stable-orbits').onchange = (e) => setMode(e.target.value);

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

let mouseDownPos = new THREE.Vector2();

window.addEventListener('mousedown', (event) => {
    mouseDownPos.x = event.clientX;
    mouseDownPos.y = event.clientY;
});

window.addEventListener('mouseup', (event) => {
    if (event.target.tagName === 'BUTTON') return;
    
    let dx = event.clientX - mouseDownPos.x;
    let dy = event.clientY - mouseDownPos.y;
    if (Math.sqrt(dx*dx + dy*dy) > 5) return; // It was a drag, not a click
    
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    
    const planeZ = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const target = new THREE.Vector3();
    raycaster.ray.intersectPlane(planeZ, target);
    
    if (target) {
        physics.numBodies++;
        physics.masses.push(mode === 'random' ? 0.05 : 1e-4);
        physics.positions.push(target.clone());
        
        let center = physics.getBarycenter();
        let rVec = new THREE.Vector3().subVectors(target, center);
        let r = rVec.length();
        let M = physics.masses[0];
        let v_mag = Math.sqrt(physics.G * M / r);
        let dir = new THREE.Vector3(-rVec.y, rVec.x, 0).normalize();
        physics.velocities.push(dir.multiplyScalar(v_mag));
        physics.accelerations = physics.computeAccelerations(physics.positions);
        
        const baseScale = mode === 'random' ? 0.3 : 0.15;
        const geo = new THREE.SphereGeometry(1, 32, 32);
        
        // Random color for new planets
        const pColor = new THREE.Color().setHSL(Math.random(), 1.0, 0.6);
        
        // Use StandardMaterial instead of BasicMaterial so it reflects star light
        // rather than emitting its own glowing bloom
        const mat = new THREE.MeshStandardMaterial({ 
            color: pColor,
            roughness: 0.3,
            metalness: 0.1
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.scale.set(baseScale * 0.3, baseScale * 0.3, baseScale * 0.3);
        
        scene.add(mesh);
        bodyMeshes.push(mesh);
        
        trailPointsList.push({ points: [], max: 3000 });
        const lineGeo = new THREE.BufferGeometry();
        const lineMat = new THREE.LineBasicMaterial({ 
            color: pColor.clone().multiplyScalar(0.8), // dimmer trail for planets
            transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending
        });
        const line = new THREE.Line(lineGeo, lineMat);
        line.frustumCulled = false;
        scene.add(line);
        trailLines.push(line);
    }
});

const bgm = document.getElementById('bgm');
const btnMusic = document.getElementById('btn-music');
bgm.volume = 0.5;

// Try to autoplay on load
bgm.play().then(() => {
    btnMusic.innerText = '🎵 Pause Music';
    btnMusic.classList.add('active');
}).catch((e) => {
    console.log("Browser prevented autoplay. User must click play.", e);
});

btnMusic.onclick = () => {
    if (bgm.paused) {
        bgm.play();
        btnMusic.innerText = '🎵 Pause Music';
        btnMusic.classList.add('active');
    } else {
        bgm.pause();
        btnMusic.innerText = '🎵 Play Music';
        btnMusic.classList.remove('active');
    }
};

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
});

// Start
initSimulation();
animate();
