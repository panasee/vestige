import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export interface SceneContext {
	scene: THREE.Scene;
	camera: THREE.PerspectiveCamera;
	renderer: THREE.WebGLRenderer;
	controls: OrbitControls;
	composer: EffectComposer;
	bloomPass: UnrealBloomPass;
	raycaster: THREE.Raycaster;
	mouse: THREE.Vector2;
	lights: {
		ambient: THREE.AmbientLight;
		point1: THREE.PointLight;
		point2: THREE.PointLight;
	};
}

export function createScene(container: HTMLDivElement): SceneContext {
	const scene = new THREE.Scene();
	scene.fog = new THREE.FogExp2(0x050510, 0.008);

	const camera = new THREE.PerspectiveCamera(
		60,
		container.clientWidth / container.clientHeight,
		0.1,
		2000
	);
	camera.position.set(0, 30, 80);

	const renderer = new THREE.WebGLRenderer({
		antialias: true,
		alpha: true,
		powerPreference: 'high-performance',
	});
	renderer.setSize(container.clientWidth, container.clientHeight);
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.2;
	container.appendChild(renderer.domElement);

	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	controls.dampingFactor = 0.05;
	controls.rotateSpeed = 0.5;
	controls.zoomSpeed = 0.8;
	controls.minDistance = 10;
	controls.maxDistance = 500;
	controls.autoRotate = true;
	controls.autoRotateSpeed = 0.3;

	const composer = new EffectComposer(renderer);
	composer.addPass(new RenderPass(scene, camera));
	const bloomPass = new UnrealBloomPass(
		new THREE.Vector2(container.clientWidth, container.clientHeight),
		0.8,
		0.4,
		0.85
	);
	composer.addPass(bloomPass);

	const ambient = new THREE.AmbientLight(0x1a1a3a, 0.5);
	scene.add(ambient);

	const point1 = new THREE.PointLight(0x6366f1, 1.5, 200);
	point1.position.set(50, 50, 50);
	scene.add(point1);

	const point2 = new THREE.PointLight(0xa855f7, 1, 200);
	point2.position.set(-50, -30, -50);
	scene.add(point2);

	const raycaster = new THREE.Raycaster();
	raycaster.params.Points = { threshold: 2 };
	const mouse = new THREE.Vector2();

	return {
		scene,
		camera,
		renderer,
		controls,
		composer,
		bloomPass,
		raycaster,
		mouse,
		lights: { ambient, point1, point2 },
	};
}

export function resizeScene(ctx: SceneContext, container: HTMLDivElement) {
	const w = container.clientWidth;
	const h = container.clientHeight;
	ctx.camera.aspect = w / h;
	ctx.camera.updateProjectionMatrix();
	ctx.renderer.setSize(w, h);
	ctx.composer.setSize(w, h);
}

export function disposeScene(ctx: SceneContext) {
	ctx.scene.traverse((obj: THREE.Object3D) => {
		if (obj instanceof THREE.Mesh || obj instanceof THREE.InstancedMesh) {
			obj.geometry?.dispose();
			if (Array.isArray(obj.material)) {
				obj.material.forEach((m: THREE.Material) => m.dispose());
			} else if (obj.material) {
				(obj.material as THREE.Material).dispose();
			}
		}
	});
	ctx.renderer.dispose();
	ctx.composer.dispose();
}
