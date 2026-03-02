<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { GraphNode, GraphEdge, VestigeEvent } from '$types';
	import { createScene, resizeScene, disposeScene, type SceneContext } from '$lib/graph/scene';
	import { ForceSimulation } from '$lib/graph/force-sim';
	import { NodeManager } from '$lib/graph/nodes';
	import { EdgeManager } from '$lib/graph/edges';
	import { ParticleSystem } from '$lib/graph/particles';
	import { EffectManager } from '$lib/graph/effects';
	import { DreamMode } from '$lib/graph/dream-mode';
	import { mapEventToEffects } from '$lib/graph/events';
	import { createNebulaBackground, updateNebula } from '$lib/graph/shaders/nebula.frag';
	import { createPostProcessing, updatePostProcessing, type PostProcessingStack } from '$lib/graph/shaders/post-processing';
	import type * as THREE from 'three';

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		centerId: string;
		events?: VestigeEvent[];
		isDreaming?: boolean;
		onSelect?: (nodeId: string) => void;
	}

	let { nodes, edges, centerId, events = [], isDreaming = false, onSelect }: Props = $props();

	let container: HTMLDivElement;
	let ctx: SceneContext;
	let animationId: number;

	// Modules
	let nodeManager: NodeManager;
	let edgeManager: EdgeManager;
	let particles: ParticleSystem;
	let effects: EffectManager;
	let forceSim: ForceSimulation;
	let dreamMode: DreamMode;
	let nebulaMaterial: THREE.ShaderMaterial;
	let postStack: PostProcessingStack;

	// Event tracking
	let processedEventCount = 0;

	onMount(() => {
		ctx = createScene(container);

		// Nebula background
		const nebula = createNebulaBackground(ctx.scene);
		nebulaMaterial = nebula.material;

		// Post-processing (added after bloom)
		postStack = createPostProcessing(ctx.composer);

		// Modules
		particles = new ParticleSystem(ctx.scene);
		nodeManager = new NodeManager();
		edgeManager = new EdgeManager();
		effects = new EffectManager(ctx.scene);
		dreamMode = new DreamMode();

		// Build graph
		const positions = nodeManager.createNodes(nodes);
		edgeManager.createEdges(edges, positions);
		forceSim = new ForceSimulation(positions);

		ctx.scene.add(edgeManager.group);
		ctx.scene.add(nodeManager.group);

		animate();

		window.addEventListener('resize', onResize);
		container.addEventListener('pointermove', onPointerMove);
		container.addEventListener('click', onClick);
	});

	onDestroy(() => {
		cancelAnimationFrame(animationId);
		window.removeEventListener('resize', onResize);
		container?.removeEventListener('pointermove', onPointerMove);
		container?.removeEventListener('click', onClick);
		effects?.dispose();
		particles?.dispose();
		nodeManager?.dispose();
		edgeManager?.dispose();
		if (ctx) disposeScene(ctx);
	});

	function animate() {
		animationId = requestAnimationFrame(animate);
		const time = performance.now() * 0.001;

		// Force simulation
		forceSim.tick(edges);

		// Update positions
		nodeManager.updatePositions();
		edgeManager.updatePositions(nodeManager.positions);

		// Animate
		particles.animate(time);
		nodeManager.animate(time, nodes, ctx.camera);

		// Dream mode
		dreamMode.setActive(isDreaming);
		dreamMode.update(ctx.scene, ctx.bloomPass, ctx.controls, ctx.lights, time);

		// Nebula + post-processing
		updateNebula(
			nebulaMaterial,
			time,
			dreamMode.current.nebulaIntensity,
			container.clientWidth,
			container.clientHeight
		);
		updatePostProcessing(postStack, time, dreamMode.current.nebulaIntensity);

		// Events + effects
		processEvents();
		effects.update(nodeManager.meshMap, ctx.camera);

		ctx.controls.update();
		ctx.composer.render();
	}

	function processEvents() {
		if (!events || events.length <= processedEventCount) return;

		const newEvents = events.slice(processedEventCount);
		processedEventCount = events.length;

		for (const event of newEvents) {
			mapEventToEffects(event, effects, nodeManager.positions, nodeManager.meshMap, ctx.camera);
		}
	}

	function onResize() {
		if (!container || !ctx) return;
		resizeScene(ctx, container);
	}

	function onPointerMove(event: PointerEvent) {
		const rect = container.getBoundingClientRect();
		ctx.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		ctx.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

		ctx.raycaster.setFromCamera(ctx.mouse, ctx.camera);
		const intersects = ctx.raycaster.intersectObjects(nodeManager.getMeshes());

		if (intersects.length > 0) {
			nodeManager.hoveredNode = intersects[0].object.userData.nodeId;
			container.style.cursor = 'pointer';
		} else {
			nodeManager.hoveredNode = null;
			container.style.cursor = 'grab';
		}
	}

	function onClick() {
		if (nodeManager.hoveredNode) {
			nodeManager.selectedNode = nodeManager.hoveredNode;
			onSelect?.(nodeManager.hoveredNode);

			const pos = nodeManager.positions.get(nodeManager.hoveredNode);
			if (pos) {
				ctx.controls.target.lerp(pos.clone(), 0.5);
			}
		}
	}
</script>

<div bind:this={container} class="w-full h-full"></div>
