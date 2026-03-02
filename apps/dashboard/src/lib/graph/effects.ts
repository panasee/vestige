import * as THREE from 'three';

export interface PulseEffect {
	nodeId: string;
	intensity: number;
	color: THREE.Color;
	decay: number;
}

interface SpawnBurst {
	position: THREE.Vector3;
	age: number;
	particles: THREE.Points;
}

interface Shockwave {
	mesh: THREE.Mesh;
	age: number;
	maxAge: number;
}

interface ConnectionFlash {
	line: THREE.Line;
	intensity: number;
}

export class EffectManager {
	pulseEffects: PulseEffect[] = [];
	private spawnBursts: SpawnBurst[] = [];
	private shockwaves: Shockwave[] = [];
	private connectionFlashes: ConnectionFlash[] = [];
	private scene: THREE.Scene;

	constructor(scene: THREE.Scene) {
		this.scene = scene;
	}

	addPulse(nodeId: string, intensity: number, color: THREE.Color, decay: number) {
		this.pulseEffects.push({ nodeId, intensity, color, decay });
	}

	createSpawnBurst(position: THREE.Vector3, color: THREE.Color) {
		const count = 60;
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(count * 3);
		const velocities = new Float32Array(count * 3);

		for (let i = 0; i < count; i++) {
			positions[i * 3] = position.x;
			positions[i * 3 + 1] = position.y;
			positions[i * 3 + 2] = position.z;
			const theta = Math.random() * Math.PI * 2;
			const phi = Math.acos(2 * Math.random() - 1);
			const speed = 0.3 + Math.random() * 0.5;
			velocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed;
			velocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed;
			velocities[i * 3 + 2] = Math.cos(phi) * speed;
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

		const mat = new THREE.PointsMaterial({
			color,
			size: 0.6,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
			sizeAttenuation: true,
		});

		const pts = new THREE.Points(geo, mat);
		this.scene.add(pts);
		this.spawnBursts.push({ position: position.clone(), age: 0, particles: pts });
	}

	createShockwave(position: THREE.Vector3, color: THREE.Color, camera: THREE.Camera) {
		const geo = new THREE.RingGeometry(0.1, 0.5, 64);
		const mat = new THREE.MeshBasicMaterial({
			color,
			transparent: true,
			opacity: 0.8,
			side: THREE.DoubleSide,
			blending: THREE.AdditiveBlending,
		});
		const ring = new THREE.Mesh(geo, mat);
		ring.position.copy(position);
		ring.lookAt(camera.position);
		this.scene.add(ring);
		this.shockwaves.push({ mesh: ring, age: 0, maxAge: 60 });
	}

	createConnectionFlash(from: THREE.Vector3, to: THREE.Vector3, color: THREE.Color) {
		const points = [from.clone(), to.clone()];
		const geo = new THREE.BufferGeometry().setFromPoints(points);
		const mat = new THREE.LineBasicMaterial({
			color,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
		});
		const line = new THREE.Line(geo, mat);
		this.scene.add(line);
		this.connectionFlashes.push({ line, intensity: 1.0 });
	}

	update(nodeMeshMap: Map<string, THREE.Mesh>, camera: THREE.Camera) {
		// Pulse effects
		for (let i = this.pulseEffects.length - 1; i >= 0; i--) {
			const pulse = this.pulseEffects[i];
			pulse.intensity -= pulse.decay;
			if (pulse.intensity <= 0) {
				this.pulseEffects.splice(i, 1);
				continue;
			}
			const mesh = nodeMeshMap.get(pulse.nodeId);
			if (mesh) {
				const mat = mesh.material as THREE.MeshStandardMaterial;
				mat.emissive.lerp(pulse.color, pulse.intensity * 0.3);
				mat.emissiveIntensity = Math.max(mat.emissiveIntensity, pulse.intensity);
			}
		}

		// Spawn bursts
		for (let i = this.spawnBursts.length - 1; i >= 0; i--) {
			const burst = this.spawnBursts[i];
			burst.age++;
			if (burst.age > 120) {
				this.scene.remove(burst.particles);
				burst.particles.geometry.dispose();
				(burst.particles.material as THREE.Material).dispose();
				this.spawnBursts.splice(i, 1);
				continue;
			}
			const positions = burst.particles.geometry.attributes.position as THREE.BufferAttribute;
			const vels = burst.particles.geometry.attributes.velocity as THREE.BufferAttribute;
			for (let j = 0; j < positions.count; j++) {
				positions.setX(j, positions.getX(j) + vels.getX(j));
				positions.setY(j, positions.getY(j) + vels.getY(j));
				positions.setZ(j, positions.getZ(j) + vels.getZ(j));
				vels.setX(j, vels.getX(j) * 0.97);
				vels.setY(j, vels.getY(j) * 0.97);
				vels.setZ(j, vels.getZ(j) * 0.97);
			}
			positions.needsUpdate = true;
			const mat = burst.particles.material as THREE.PointsMaterial;
			mat.opacity = Math.max(0, 1 - burst.age / 120);
			mat.size = 0.6 * (1 - burst.age / 200);
		}

		// Shockwaves
		for (let i = this.shockwaves.length - 1; i >= 0; i--) {
			const sw = this.shockwaves[i];
			sw.age++;
			if (sw.age > sw.maxAge) {
				this.scene.remove(sw.mesh);
				sw.mesh.geometry.dispose();
				(sw.mesh.material as THREE.Material).dispose();
				this.shockwaves.splice(i, 1);
				continue;
			}
			const progress = sw.age / sw.maxAge;
			sw.mesh.scale.setScalar(1 + progress * 20);
			(sw.mesh.material as THREE.MeshBasicMaterial).opacity = 0.8 * (1 - progress);
			sw.mesh.lookAt(camera.position);
		}

		// Connection flashes
		for (let i = this.connectionFlashes.length - 1; i >= 0; i--) {
			const flash = this.connectionFlashes[i];
			flash.intensity -= 0.015;
			if (flash.intensity <= 0) {
				this.scene.remove(flash.line);
				flash.line.geometry.dispose();
				(flash.line.material as THREE.Material).dispose();
				this.connectionFlashes.splice(i, 1);
				continue;
			}
			(flash.line.material as THREE.LineBasicMaterial).opacity = flash.intensity;
		}
	}

	dispose() {
		for (const burst of this.spawnBursts) {
			this.scene.remove(burst.particles);
			burst.particles.geometry.dispose();
			(burst.particles.material as THREE.Material).dispose();
		}
		for (const sw of this.shockwaves) {
			this.scene.remove(sw.mesh);
			sw.mesh.geometry.dispose();
			(sw.mesh.material as THREE.Material).dispose();
		}
		for (const flash of this.connectionFlashes) {
			this.scene.remove(flash.line);
			flash.line.geometry.dispose();
			(flash.line.material as THREE.Material).dispose();
		}
		this.pulseEffects = [];
		this.spawnBursts = [];
		this.shockwaves = [];
		this.connectionFlashes = [];
	}
}
