'use strict';
{
  const StartBtn = document.getElementById("js-startButton")
  const TopHero = document.querySelector(".top-hero")
  const Fight = document.querySelector(".fight")

  StartBtn.addEventListener("click",function(){
    TopHero.classList.add("fadeout")
    TopHero.style.display = "none";
    Fight.style.display = "block";
    Fight.classList.add("fadein")
  })
  







}