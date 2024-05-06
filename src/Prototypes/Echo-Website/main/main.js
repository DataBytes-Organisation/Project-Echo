function scrollFunction() {
    var header = document.getElementById("header");
    if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50) {
        header.style.height = "100px";
        header.querySelector('h1').style.display = "block";
    } else {
        header.style.height = "200px";
        header.querySelector('h1').style.display = "none";
    }
}
