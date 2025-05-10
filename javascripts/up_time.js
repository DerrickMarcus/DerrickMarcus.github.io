; (function () {
    // 1. 把下面的日期替换成你的网站“上线时间”，ISO 格式最好：
    const launchDate = new Date("2024-11-02T13:22:38+08:00");

    // 2. 获取 <div id="site-uptime">
    const el = document.getElementById("site-uptime");
    if (!el) return;

    function updateUptime() {
        const now = new Date();
        let diff = Math.floor((now - launchDate) / 1000); // 单位：秒

        const days = Math.floor(diff / 86400);
        const hours = Math.floor((diff % 86400) / 3600);
        const minutes = Math.floor((diff % 3600) / 60);
        const seconds = diff % 60;

        el.textContent = `The site has been running for ${days} days ${hours} h ${minutes} min ${seconds} s`;
    }

    updateUptime();
    setInterval(updateUptime, 1000);
})();
