function updateTimeDiff() {
    const targetDate = new Date("2024-11-02T13:22:38Z"); // 替换为目标时间
    const now = new Date();
    const diff = now - targetDate;

    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);

    document.getElementById("time-diff-zh").innerHTML = `本站已经运行 ${days} 天 ${hours} 小时 ${minutes} 分钟 ${seconds} 秒`;
    document.getElementById("time-diff-en").innerHTML = `This site has been running for ${days} days ${hours} hours ${minutes} minutes ${seconds} seconds`;
}

setInterval(updateTimeDiff, 1000); // 每秒更新一次