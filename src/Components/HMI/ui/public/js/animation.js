const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

let interval = null;


function textFlex() {  
    console.log('event start!')
    let iteration = 0;
    clearInterval(interval);
    const name = document.getElementById("name");
    interval = setInterval(() => {
        name.innerHTML = name.innerHTML.split("").map((letter, index) => {
            if(index < iteration) {
            return name.dataset.value[index];
            }
        
            return letters[Math.floor(Math.random() * 26)]
        })
        .join("");
        
        if(iteration >= name.dataset.value.length){ 
        clearInterval(interval);
        }
        
        iteration += 1 / 3;
    }, 30);
}

function fadeAnim(){
    const screen = document.querySelector('.screen');
    const blur = document.getElementById('blur');
    const hamburger = document.querySelector('.hamburger-icon');
    screen.classList.toggle('change');
    blur.classList.toggle('change');
    setTimeout(()=>{
        hamburger.style.display = "block";
        screen.style.display = "none";
        blur.style.display = "none";
        
    },1000)
}