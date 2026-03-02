import * as THREE from 'three';
import type { GraphEdge } from '$types';

export class EdgeManager {
	group: THREE.Group;

	constructor() {
		this.group = new THREE.Group();
	}

	createEdges(edges: GraphEdge[], positions: Map<string, THREE.Vector3>) {
		for (const edge of edges) {
			const sourcePos = positions.get(edge.source);
			const targetPos = positions.get(edge.target);
			if (!sourcePos || !targetPos) continue;

			const points = [sourcePos, targetPos];
			const geometry = new THREE.BufferGeometry().setFromPoints(points);
			const material = new THREE.LineBasicMaterial({
				color: 0x4a4a7a,
				transparent: true,
				opacity: Math.min(0.1 + edge.weight * 0.5, 0.6),
				blending: THREE.AdditiveBlending,
			});

			const line = new THREE.Line(geometry, material);
			line.userData = { source: edge.source, target: edge.target };
			this.group.add(line);
		}
	}

	updatePositions(positions: Map<string, THREE.Vector3>) {
		this.group.children.forEach((child) => {
			const line = child as THREE.Line;
			const sourcePos = positions.get(line.userData.source);
			const targetPos = positions.get(line.userData.target);
			if (sourcePos && targetPos) {
				const attrs = line.geometry.attributes.position as THREE.BufferAttribute;
				attrs.setXYZ(0, sourcePos.x, sourcePos.y, sourcePos.z);
				attrs.setXYZ(1, targetPos.x, targetPos.y, targetPos.z);
				attrs.needsUpdate = true;
			}
		});
	}

	dispose() {
		this.group.children.forEach((child) => {
			const line = child as THREE.Line;
			line.geometry?.dispose();
			(line.material as THREE.Material)?.dispose();
		});
	}
}
