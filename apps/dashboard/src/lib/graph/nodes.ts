import * as THREE from 'three';
import type { GraphNode } from '$types';
import { NODE_TYPE_COLORS } from '$types';

export class NodeManager {
	group: THREE.Group;
	meshMap = new Map<string, THREE.Mesh>();
	positions = new Map<string, THREE.Vector3>();
	labelSprites = new Map<string, THREE.Sprite>();
	hoveredNode: string | null = null;
	selectedNode: string | null = null;

	constructor() {
		this.group = new THREE.Group();
	}

	createNodes(nodes: GraphNode[]): Map<string, THREE.Vector3> {
		const phi = (1 + Math.sqrt(5)) / 2;
		const count = nodes.length;

		for (let i = 0; i < count; i++) {
			const node = nodes[i];

			// Fibonacci sphere distribution for initial positions
			const y = 1 - (2 * i) / (count - 1 || 1);
			const radius = Math.sqrt(1 - y * y);
			const theta = (2 * Math.PI * i) / phi;
			const spread = 30 + count * 0.5;

			const pos = new THREE.Vector3(
				radius * Math.cos(theta) * spread,
				y * spread,
				radius * Math.sin(theta) * spread
			);

			if (node.isCenter) pos.set(0, 0, 0);

			this.positions.set(node.id, pos);

			const size = 0.5 + node.retention * 2;
			const color = NODE_TYPE_COLORS[node.type] || '#8B95A5';

			// Node mesh
			const geometry = new THREE.SphereGeometry(size, 16, 16);
			const material = new THREE.MeshStandardMaterial({
				color: new THREE.Color(color),
				emissive: new THREE.Color(color),
				emissiveIntensity: 0.3 + node.retention * 0.5,
				roughness: 0.3,
				metalness: 0.1,
				transparent: true,
				opacity: 0.3 + node.retention * 0.7,
			});

			const mesh = new THREE.Mesh(geometry, material);
			mesh.position.copy(pos);
			mesh.userData = { nodeId: node.id, type: node.type, retention: node.retention };
			this.meshMap.set(node.id, mesh);
			this.group.add(mesh);

			// Glow sprite
			const spriteMat = new THREE.SpriteMaterial({
				color: new THREE.Color(color),
				transparent: true,
				opacity: 0.15 + node.retention * 0.2,
				blending: THREE.AdditiveBlending,
			});
			const sprite = new THREE.Sprite(spriteMat);
			sprite.scale.set(size * 4, size * 4, 1);
			sprite.position.copy(pos);
			sprite.userData = { isGlow: true, nodeId: node.id };
			this.group.add(sprite);

			// Text label sprite
			const labelText = node.label || node.type;
			const labelSprite = this.createTextSprite(labelText, '#e2e8f0');
			labelSprite.position.copy(pos);
			labelSprite.position.y += size * 2 + 1.5;
			labelSprite.userData = { isLabel: true, nodeId: node.id, offset: size * 2 + 1.5 };
			this.group.add(labelSprite);
			this.labelSprites.set(node.id, labelSprite);
		}

		return this.positions;
	}

	private createTextSprite(text: string, color: string): THREE.Sprite {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d')!;
		canvas.width = 512;
		canvas.height = 64;

		const label = text.length > 40 ? text.slice(0, 37) + '...' : text;

		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.font = 'bold 28px -apple-system, BlinkMacSystemFont, sans-serif';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
		ctx.shadowBlur = 6;
		ctx.shadowOffsetX = 0;
		ctx.shadowOffsetY = 2;
		ctx.fillStyle = color;
		ctx.fillText(label, canvas.width / 2, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		texture.needsUpdate = true;

		const mat = new THREE.SpriteMaterial({
			map: texture,
			transparent: true,
			opacity: 0,
			depthTest: false,
			sizeAttenuation: true,
		});

		const sprite = new THREE.Sprite(mat);
		sprite.scale.set(12, 1.5, 1);
		return sprite;
	}

	updatePositions() {
		this.group.children.forEach((child) => {
			if (child.userData.nodeId) {
				const pos = this.positions.get(child.userData.nodeId);
				if (!pos) return;

				if (child.userData.isGlow) {
					child.position.copy(pos);
				} else if (child.userData.isLabel) {
					child.position.copy(pos);
					child.position.y += child.userData.offset;
				} else if (child instanceof THREE.Mesh) {
					child.position.copy(pos);
				}
			}
		});
	}

	animate(time: number, nodes: GraphNode[], camera: THREE.PerspectiveCamera) {
		// Node breathing
		this.meshMap.forEach((mesh, id) => {
			const node = nodes.find((n) => n.id === id);
			if (!node) return;
			const breathe =
				1 + Math.sin(time * 1.5 + nodes.indexOf(node) * 0.5) * 0.15 * node.retention;
			mesh.scale.setScalar(breathe);

			const mat = mesh.material as THREE.MeshStandardMaterial;
			if (id === this.hoveredNode) {
				mat.emissiveIntensity = 1.0;
			} else if (id === this.selectedNode) {
				mat.emissiveIntensity = 0.8;
			} else {
				// Low-retention nodes breathe slower
				const baseIntensity = 0.3 + node.retention * 0.5;
				const breatheIntensity =
					baseIntensity + Math.sin(time * (0.8 + node.retention * 0.7)) * 0.1 * node.retention;
				mat.emissiveIntensity = breatheIntensity;
			}
		});

		// Distance-based label visibility
		this.labelSprites.forEach((sprite, id) => {
			const pos = this.positions.get(id);
			if (!pos) return;
			const dist = camera.position.distanceTo(pos);
			const mat = sprite.material as THREE.SpriteMaterial;
			const targetOpacity =
				id === this.hoveredNode || id === this.selectedNode
					? 1.0
					: dist < 40
						? 0.9
						: dist < 80
							? 0.9 * (1 - (dist - 40) / 40)
							: 0;
			mat.opacity += (targetOpacity - mat.opacity) * 0.1;
		});
	}

	getMeshes(): THREE.Mesh[] {
		return Array.from(this.meshMap.values());
	}

	dispose() {
		this.group.traverse((obj) => {
			if (obj instanceof THREE.Mesh) {
				obj.geometry?.dispose();
				(obj.material as THREE.Material)?.dispose();
			} else if (obj instanceof THREE.Sprite) {
				(obj.material as THREE.SpriteMaterial)?.map?.dispose();
				(obj.material as THREE.Material)?.dispose();
			}
		});
	}
}
