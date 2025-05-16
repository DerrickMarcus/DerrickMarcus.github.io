function timingTime(startTimeString) {
    const startTime = new Date(startTimeString).getTime();
    const now = Date.now();
    let diff = Math.floor((now - startTime) / 1000);
    const days = Math.floor(diff / 86400);
    diff %= 86400;
    const hours = Math.floor(diff / 3600);
    diff %= 3600;
    const minutes = Math.floor(diff / 60);
    const seconds = diff % 60;
    return `${days}days ${hours}h ${minutes}min ${seconds}s.`;
}

function updateRuntime(startTimeString, elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = timingTime(startTimeString);
    }
}

function updateCurrentTime(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const now = new Date();
    const options = {
        year: 'numeric',
        month: 'long',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        weekday: 'long',
        hour12: false,
    };

    const formatted = now.toLocaleString('en-US', options);
    el.textContent = formatted;
}

const startTime = "2024-11-02T13:22:38";

setInterval(() => {
    updateRuntime(startTime, "runtime-in-md");
    updateRuntime(startTime, "runtime-in-footer");
    updateCurrentTime("current-time");
}, 1000);
