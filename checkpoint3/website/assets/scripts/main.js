//* ======================== Slide Control ===================== */
// Function to initialize slide display controls
function initSlideDisplay(slideDisplayClass) {
  const slideDisplay = document.querySelector("." + slideDisplayClass);
  if (!slideDisplay) return; // Skip if slide display not found

  // Find slide menu - handle both direct ul#slide-menu and nested ul within a div.slide-menu
  let slideMenu = slideDisplay.querySelector("#slide-menu");
  if (!slideMenu) {
    const slideMenuContainer = slideDisplay.querySelector(".slide-menu");
    if (slideMenuContainer) {
      slideMenu = slideMenuContainer.querySelector("#slide-menu");
    }
  }

  if (!slideMenu) return; // Skip if slide menu not found

  const contents = slideDisplay.querySelectorAll(".slide-content");
  if (!contents.length) return; // Skip if no slide contents found

  // Set up click event for this specific slide menu

  slideMenu.addEventListener("click", function (e) {
    // Check if clicked on dot or li.dot
    const isLiDot =
      e.target.tagName === "LI" && e.target.classList.contains("dot");
    const isDot = e.target.classList.contains("dot");

    if (!isLiDot && !isDot) return; // Not clicked on a dot

    // Get index of clicked dot
    let idx;
    if (isLiDot) {
      // Get all li.dot elements
      const dots = [...this.querySelectorAll("li.dot")];
      idx = dots.indexOf(e.target);
    } else {
      // Get all .dot elements
      const dots = [...this.querySelectorAll(".dot")];
      idx = dots.indexOf(e.target);
    }

    if (idx >= 0) {
      // Remove active class from previously active dot in this menu
      const prev = slideDisplay.querySelector(".dot.active");
      if (prev) prev.classList.remove("active");

      // Add active class to clicked dot
      e.target.classList.add("active");

      // Show/hide slide contents for this slide display
      for (var i = 0; i < contents.length; i++) {
        if (i == idx) {
          contents[i].style.display = "block";
          contents[i].style.width = "100%";
          contents[i].style.height = "auto";
          contents[i].style.visibility = "visible";
        } else {
          contents[i].style.display = "none";
          // contents[i].style.visibility = "hidden";
        }
      }
    }
  });

  // Initialize first slide as active if none is active
  const dots = slideMenu.querySelectorAll(".dot");
  if (dots.length > 0) {
    if (!slideDisplay.querySelector(".dot.active")) {
      dots[0].classList.add("active");
      if (contents.length > 0) {
        contents[0].style.display = "block";
        contents[i].style.width = "100%";
        contents[i].style.height = "100%";
        contents[0].style.visibility = "visible";

        // Hide all other slides
        for (var i = 1; i < contents.length; i++) {
          contents[i].style.display = "none";
          // contents[i].style.visibility = "hidden";
        }
      }
    }
  }
}

//* ======================== Video Control ===================== */
function ToggleVideo(x) {
  var videos = document.getElementsByClassName(x + "-video");
  for (var i = 0; i < videos.length; i++) {
    if (videos[i].paused) {
      videos[i].play();
    } else {
      videos[i].pause();
    }
  }
}

function SlowVideo(x) {
  var videos = document.getElementsByClassName(x + "-video");
  for (var i = 0; i < videos.length; i++) {
    videos[i].playbackRate = videos[i].playbackRate * 0.9;
    videos[i].play();
  }

  var msg = document.getElementById(x + "-msg");
  msg.innerHTML = "Speed: " + "×" + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = "none";
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null;
}

function FastVideo(x) {
  var videos = document.getElementsByClassName(x + "-video");
  for (var i = 0; i < videos.length; i++) {
    videos[i].playbackRate = videos[i].playbackRate / 0.9;
    videos[i].play();
  }

  var msg = document.getElementById(x + "-msg");
  msg.innerHTML = "Speed: " + "×" + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = "none";
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null;
}

function RestartVideo(x) {
  var videos = document.getElementsByClassName(x + "-video");
  for (var i = 0; i < videos.length; i++) {
    videos[i].pause();
    videos[i].playbackRate = 1.0;
    videos[i].currentTime = 0;
    videos[i].play();
  }

  var msg = document.getElementById(x + "-msg");
  msg.innerHTML = "Speed: " + "×" + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = "none";
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null;
}

//* ======================== Slide Show Control ===================== */
// Object to store all slideshow intervals
const slideshowIntervals = {};

// Function to create slideshow instances
function initSlideshow(slideshowClass) {
  const slideshow = document.querySelector("." + slideshowClass);
  if (!slideshow) return; // Skip if slideshow not found

  const slider = slideshow.querySelector(".slider");
  if (!slider) return; // Skip if slider not found

  const prevBtn = slideshow.querySelector("#prev_btn");
  const nextBtn = slideshow.querySelector("#next_btn");

  if (!prevBtn || !nextBtn) return; // Skip if buttons not found

  // Set positions for this specific slideshow
  const setPositions = () =>
    [...slider.children].forEach(
      (item, i) => (item.style.left = `${(i - 1) * 440}px`)
    );

  // Initial setup
  setPositions();

  // Set transition speed for this specific slideshow
  const setTransitionSpeed = (speed) => {
    [...slider.children].forEach(
      (item) => (item.style.transitionDuration = speed)
    );
  };

  // Slide functions for this specific slideshow
  const next = (isAuto = false) => {
    setTransitionSpeed(isAuto ? "1.5s" : "0.2s");
    slider.appendChild(slider.firstElementChild);
    setPositions();
  };

  const prev = () => {
    setTransitionSpeed("0.2s");
    slider.prepend(slider.lastElementChild);
    setPositions();
  };

  // Auto slide for this specific slideshow
  const startAuto = () => {
    if (!slideshowIntervals[slideshowClass]) {
      slideshowIntervals[slideshowClass] = setInterval(() => next(true), 2000);
    }
  };

  const stopAuto = () => {
    clearInterval(slideshowIntervals[slideshowClass]);
    slideshowIntervals[slideshowClass] = null;
  };

  // Event listeners for this specific slideshow
  prevBtn.addEventListener("click", () => {
    prev();
    stopAuto(); // Stop auto when manually navigating
    setTimeout(startAuto, 5000); // Restart auto after a delay
  });

  nextBtn.addEventListener("click", () => {
    next(false);
    stopAuto(); // Stop auto when manually navigating
    setTimeout(startAuto, 5000); // Restart auto after a delay
  });

  // Mouse hover controls for this specific slideshow
  [slider, prevBtn, nextBtn].forEach((el) => {
    el.addEventListener("mouseover", stopAuto);
    el.addEventListener("mouseout", startAuto);
  });

  // Start auto slide for this specific slideshow
  startAuto();
}

// Initialize all slideshows and slide displays on the page
document.addEventListener("DOMContentLoaded", function () {
  // Initialize auto slideshows
  const slideshows = document.querySelectorAll(
    ".slideshow, .slideshow-2, .slideshow-3"
  );
  slideshows.forEach((slideshow, index) => {
    if (index === 0 && slideshow.classList.contains("slideshow")) {
      // First slideshow already has the correct class
      initSlideshow("slideshow");
    } else if (slideshow.classList.contains("slideshow-2")) {
      // Second slideshow already has the correct class
      initSlideshow("slideshow-2");
    } else if (slideshow.classList.contains("slideshow-3")) {
      // Third slideshow already has the correct class
      initSlideshow("slideshow-3");
    } else {
      // If we find more slideshows, add a unique class and initialize them
      const newClass = "slideshow-" + (index + 1);
      slideshow.classList.add(newClass);
      initSlideshow(newClass);
    }
  });

  // Initialize slide displays (dot navigation)
  const slideDisplays = document.querySelectorAll(
    ".slide-display, .slide-display-2, .slide-display-3"
  );
  slideDisplays.forEach((slideDisplay, index) => {
    if (index === 0 && slideDisplay.classList.contains("slide-display")) {
      // First slide display already has the correct class
      initSlideDisplay("slide-display");
    } else if (slideDisplay.classList.contains("slide-display-2")) {
      // Second slide display already has the correct class
      initSlideDisplay("slide-display-2");
    } else if (slideDisplay.classList.contains("slide-display-3")) {
      // Third slide display already has the correct class
      initSlideDisplay("slide-display-3");
    } else {
      // If we find more slide displays, add a unique class and initialize them
      const newClass = "slide-display-" + (index + 1);
      slideDisplay.classList.add(newClass);
      initSlideDisplay(newClass);
    }
  });
});
