import * as THREE from 'three';
import type { VestigeEvent } from '$types';
import type { EffectManager } from './effects';

export function mapEventToEffects(
	event: VestigeEvent,
	effects: EffectManager,
	nodePositions: Map<string, THREE.Vector3>,
	nodeMeshMap: Map<string, THREE.Mesh>,
	camera: THREE.Camera
) {
	switch (event.type) {
		case 'MemoryCreated': {
			const nodeId = (event.data as { id?: string })?.id;
			const pos = nodeId ? nodePositions.get(nodeId) : null;
			const burstPos =
				pos?.clone() ??
				new THREE.Vector3(
					(Math.random() - 0.5) * 40,
					(Math.random() - 0.5) * 40,
					(Math.random() - 0.5) * 40
				);
			effects.createSpawnBurst(burstPos, new THREE.Color(0x00ffd1));
			effects.createShockwave(burstPos, new THREE.Color(0x00ffd1), camera);
			break;
		}
		case 'SearchPerformed': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 0.6 + Math.random() * 0.4, new THREE.Color(0x818cf8), 0.02);
			});
			break;
		}
		case 'DreamStarted': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 1.0, new THREE.Color(0xa855f7), 0.005);
			});
			break;
		}
		case 'DreamProgress': {
			const memoryId = (event.data as { memory_id?: string })?.memory_id;
			if (memoryId && nodeMeshMap.has(memoryId)) {
				effects.addPulse(memoryId, 1.5, new THREE.Color(0xc084fc), 0.01);
			}
			break;
		}
		case 'DreamCompleted': {
			effects.createSpawnBurst(new THREE.Vector3(0, 0, 0), new THREE.Color(0xa855f7));
			effects.createShockwave(new THREE.Vector3(0, 0, 0), new THREE.Color(0xa855f7), camera);
			break;
		}
		case 'ConnectionDiscovered': {
			const data = event.data as { source_id?: string; target_id?: string };
			const srcPos = data.source_id ? nodePositions.get(data.source_id) : null;
			const tgtPos = data.target_id ? nodePositions.get(data.target_id) : null;
			if (srcPos && tgtPos) {
				effects.createConnectionFlash(srcPos, tgtPos, new THREE.Color(0x00d4ff));
			}
			break;
		}
		case 'RetentionDecayed': {
			const decayId = (event.data as { id?: string })?.id;
			if (decayId && nodeMeshMap.has(decayId)) {
				effects.addPulse(decayId, 0.8, new THREE.Color(0xff4757), 0.03);
			}
			break;
		}
		case 'MemoryPromoted': {
			const promoId = (event.data as { id?: string })?.id;
			if (promoId && nodeMeshMap.has(promoId)) {
				effects.addPulse(promoId, 1.2, new THREE.Color(0x00ff88), 0.01);
				const promoPos = nodePositions.get(promoId);
				if (promoPos) effects.createShockwave(promoPos, new THREE.Color(0x00ff88), camera);
			}
			break;
		}
		case 'ConsolidationCompleted': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 0.4 + Math.random() * 0.3, new THREE.Color(0xffb800), 0.015);
			});
			break;
		}
		case 'ActivationSpread': {
			const spreadData = event.data as { source_id?: string; target_ids?: string[] };
			if (spreadData.source_id && spreadData.target_ids) {
				const srcPos = nodePositions.get(spreadData.source_id);
				if (srcPos) {
					for (const targetId of spreadData.target_ids) {
						const tgtPos = nodePositions.get(targetId);
						if (tgtPos) {
							effects.createConnectionFlash(srcPos, tgtPos, new THREE.Color(0x14e8c6));
						}
					}
				}
			}
			break;
		}
	}
}
