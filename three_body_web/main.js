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
                new THREE.Vector3(o.positions[0][0], o.positions[0][1], 0).multiplyScalar(2),
                new THREE.Vector3(o.positions[1][0], o.positions[1][1], 0).multiplyScalar(2),
                new THREE.Vector3(o.positions[2][0], o.positions[2][1], 0).multiplyScalar(2)
            ];
            
            this.velocities = [
                new THREE.Vector3(o.velocities[0][0], o.velocities[0][1], 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(o.velocities[1][0], o.velocities[1][1], 0).divideScalar(Math.sqrt(2)),
                new THREE.Vector3(o.velocities[2][0], o.velocities[2][1], 0).divideScalar(Math.sqrt(2))
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
    "figure8": {
        positions: [ [-1, 0], [1, 0], [0, 0] ],
        velocities: [ [0.347111, 0.532728], [0.347111, 0.532728], [-0.694222, -1.065456] ]
    },
    "Broucke_A11": {
        positions: [ [0.0132604844, 0], [1.4157286016, 0], [-1.4289890859, 0] ],
        velocities: [ [0, 1.054151921], [0, -0.2101466639], [0, -0.8440052572] ]
    },
    "Broucke_R4": {
        positions: [ [0.8733047091, 0], [-0.6254030288, 0], [-0.2479016803, 0] ],
        velocities: [ [0, 1.0107764436], [0, -1.6833533458], [0, 0.6725769022] ]
    },
    "Dragonfly": {
        positions: [ [-1, 0], [1, 0], [0, 0] ],
        velocities: [ [0.080584, 0.588836], [0.080584, 0.588836], [-0.161168, -1.177672] ]
    },
    "Loop_end_triangles": {
        positions: [ [0.6661637520772179, -0.081921852656887], [-0.025192663684493022, 0.45444857588251897], [-0.10301329374224, -0.765806200083609] ],
        velocities: [ [0.84120297540307, 0.029746212757039], [0.142642469612081, -0.492315648524683], [-0.98384544501151, 0.462569435774018] ]
    },
    "Broucke_A1": {
        positions: [ [-0.9892620043, 0], [2.2096177241, 0], [-1.2203557197, 0] ],
        velocities: [ [0, 1.9169244185], [0, 0.1910268738], [0, -2.1079512924] ]
    },
    "two_ovals": {
        positions: [ [0.486657678894505, 0.755041888583519], [-0.681737994414464, 0.29366023319721], [-0.02259632746864, -0.612645601255358] ],
        velocities: [ [-0.182709864466916, 0.363013287999004], [-0.579074922540872, -0.748157481446087], [0.761784787007641, 0.385144193447218] ]
    },
    "butterfly_I": {
        positions: [ [-1, 0], [1, 0], [0, 0] ],
        velocities: [ [0.306893, 0.125507], [0.306893, 0.125507], [-0.613786, -0.251014] ]
    },
    "catface": {
        positions: [ [0.53638707339, 0.054088605008], [-0.252099126491, 0.694527327749], [-0.275706601688, -0.335933589318] ],
        velocities: [ [-0.569379585581, 1.255291102531], [0.079644615252, -0.458625997341], [0.489734970329, -0.796665105189] ]
    },
    "Broucke_A4": {
        positions: [ [0.2843198916, 0.0], [0.8736097872, 0.0], [-1.1579296788, 0.0] ],
        velocities: [ [0.0, 1.377417957], [0.0, -0.4884226932], [0.0, -0.8889952638] ]
    },
    "Broucke_R1": {
        positions: [ [0.808310623, 0.0], [-0.4954148566, 0.0], [-0.3128957664, 0.0] ],
        velocities: [ [0.0, 0.9901979166], [0.0, -2.7171431768], [0.0, 1.7269452602] ]
    },
    "IVa_2_A": {
        positions: [ [-1, 0], [1, 0], [0, 0] ],
        velocities: [ [0.464445, 0.39606], [0.464445, 0.39606], [-0.92889, -0.79212] ]
    },
    "Broucke_A13": {
        positions: [ [-0.8965015243, 0], [3.2352526189, 0], [-2.3387510946, 0] ],
        velocities: [ [0, 0.8285556923], [0, -0.0056478094], [0, -0.8229078829] ]
    },
    "Broucke_A10": {
        positions: [ [-0.5426216182, 0], [2.5274928067, 0], [-1.9848711885, 0] ],
        velocities: [ [0, 0.8750200467], [0, -0.0526955841], [0, -0.8223244626] ]
    },
    "googles": {
        positions: [ [-1, 0], [1, 0], [0, 0] ],
        velocities: [ [0.0833, 0.127889], [0.0833, 0.127889], [-0.1666, -0.255778] ]
    },
    "Broucke_A7": {
        positions: [ [-0.1095519101, 0], [1.6613533905, 0], [-1.5518014804, 0] ],
        velocities: [ [0, 0.9913358338], [0, -0.1569959746], [0, -0.8343398592] ]
    }
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
