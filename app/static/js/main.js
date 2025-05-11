// Отображение времени
function updateTime() {
    const now = new Date();
    document.getElementById('current-time').textContent =
        now.toLocaleTimeString();
}
setInterval(updateTime, 1000);
updateTime();


// Полноэкранный режим
function toggleFullscreen() {
    const videoWrapper = document.querySelector('.video-wrapper');
    if (!document.fullscreenElement) {
        videoWrapper.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}