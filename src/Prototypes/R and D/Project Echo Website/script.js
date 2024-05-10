// Example of dynamic behavior: console log when button is clicked
document.querySelector('.btn-discover').addEventListener('click', function() {
    console.log('Discover more about Project Echo!');
});

function scrollToMission() {
    const missionSection = document.getElementById('mission');
    if (missionSection) {
        missionSection.scrollIntoView({ behavior: 'smooth' });
    }
}