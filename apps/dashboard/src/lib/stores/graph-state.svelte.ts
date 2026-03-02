// Shared graph state using Svelte 5 $state runes
// This store manages temporal playback and dream mode state

export const graphState = createGraphState();

function createGraphState() {
	let temporalEnabled = $state(false);
	let temporalDate = $state<Date>(new Date());
	let temporalPlaying = $state(false);
	let temporalSpeed = $state(1); // days per second: 1, 7, 30
	let dreamMode = $state(false);

	return {
		get temporalEnabled() {
			return temporalEnabled;
		},
		set temporalEnabled(v: boolean) {
			temporalEnabled = v;
		},

		get temporalDate() {
			return temporalDate;
		},
		set temporalDate(v: Date) {
			temporalDate = v;
		},

		get temporalPlaying() {
			return temporalPlaying;
		},
		set temporalPlaying(v: boolean) {
			temporalPlaying = v;
		},

		get temporalSpeed() {
			return temporalSpeed;
		},
		set temporalSpeed(v: number) {
			temporalSpeed = v;
		},

		get dreamMode() {
			return dreamMode;
		},
		set dreamMode(v: boolean) {
			dreamMode = v;
		},
	};
}
