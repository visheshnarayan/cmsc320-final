document.addEventListener("DOMContentLoaded", function () {
  const style = document.createElement("style");
  style.textContent = `
          .nav-container {
              position: fixed;
              z-index: 1000;
              backdrop-filter: blur(10px);
              -webkit-backdrop-filter: blur(10px);
          }
  
          /* Desktop styles */
          @media (min-width: 768px) {
              .nav-container {
                  position: fixed;
                  left: 20px;
                  top: 5%;
                  transform: translateY(-5%) translateX(5%);
                  padding: 15px;
                  border-radius: 16px;
                  background: rgba(255, 255, 255, 0.5);
                  opacity: 0;
                  pointer-events: none;
                  max-width: 300px;
                  width: 300px;
                  display: flex;
                  flex-direction: column;
              }
  
              .nav-container.visible {
                  opacity: 1;
                  pointer-events: all;
                  transition: all 1.3s ease;
              }
  
              .nav-container.collapsed {
                  width: auto;
                  min-width: 50px;
              }
  
              .nav-container.collapsed .nav-list {
                  display: none;
              }
  
              .desktop-header {
                  display: flex;
                  justify-content: space-between;
                  align-items: center;
                  gap: 10px;
              }
  
              .nav-container.collapsed .desktop-header {
                  padding: 0;
              }
  
              .nav-container:not(.collapsed) .desktop-header {
                  padding-bottom: 10px;
              }
  
              .nav-container.collapsed .current-section {
                  display: block;
                  flex-grow: 1;
                  font-size: 14px;
                  font-family: 'Poppins', sans-serif;
                  white-space: nowrap;
                  overflow: hidden;
                  text-overflow: ellipsis;
                  opacity: 0.7;
                  margin: 0 5px;
              }
  
              .nav-container:not(.collapsed) .desktop-header .current-section {
                  display: none;
              }
  
              .nav-collapse {
                  background: none;
                  border: none;
                  cursor: pointer;
                  padding: 5px;
                  margin: 0;
                  opacity: 0.3;
                  transition: opacity 0.2s ease;
                  flex-shrink: 0;
              }
  
              .nav-collapse:hover {
                  opacity: 1;
              }
  
              .expand-icon {
                  display: none;
              }
  
              .collapsed .expand-icon {
                  display: block;
              }
  
              .collapsed .collapse-icon {
                  display: none;
              }
  
              .nav-header {
                  display: none !important;
              }
  
              .nav-list {
                  list-style: none;
                  padding: 0;
                  margin: 0;
                  overflow-y: auto;
                  max-height: 70vh;
              }
  
              .nav-item {
                  margin: 10px 0;
                  cursor: pointer;
                  color: #666;
                  transition: color 0.3s ease;
                  font-size: 14px;
              }
  
              .nav-item.h2 {
                  padding-left: 15px;
                  font-size: 12px;
                  margin: 5px 0;
              }
              
              .nav-item.h3 {
                  padding-left: 30px;
                  font-size: 11px;
                  margin: 4px 0;
                  opacity: 0.8;
              }
  
              .nav-item.active {
                  color: #000;
                  font-weight: 500;
              }
          }
  
          /* Mobile styles */
          @media (max-width: 767px) {
              .nav-container {
                  position: fixed;
                  top: 0;
                  left: 0;
                  right: 0;
                  width: 100%;
                  background: rgba(255, 255, 255, 0.5);
                  transform: translateY(-100%);
                  transition: transform 0.3s ease;
                  height: auto;
              }
  
              .nav-container.visible {
                  transform: translateY(0);
              }
  
              .nav-container.expanded {
                  height: 100vh;
                  background: rgba(255, 255, 255, 0.95);
              }
  
              .nav-header {
                  display: flex;
                  justify-content: space-between;
                  align-items: center;
                  padding: 15px 20px;
                  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
              }
  
              .current-section {
                  font-size: 14px;
                  font-weight: 500;
                  color: #000;
                  font-family: 'Poppins', sans-serif;
                  white-space: nowrap;
                  overflow: hidden;
                  text-overflow: ellipsis;
                  max-width: calc(100% - 40px);
              }
  
              .nav-close {
                  display: none;
                  cursor: pointer;
              }
  
              @media (min-width: 768px) {
                  .nav-header {
                      display: none;
                  }
              }
  
              .expanded .nav-close {
                  display: block;
              }
  
              .nav-list {
                  display: none;
                  list-style: none;
                  padding: 20px;
                  margin: 0;
                  max-height: calc(100vh - 60px);
                  overflow-y: auto;
              }
  
              .expanded .nav-list {
                  display: block;
              }
  
              .nav-item {
                  padding: 15px 0;
                  cursor: pointer;
                  color: #666;
                  transition: color 0.3s ease;
                  font-size: 16px;
                  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
              }
  
              .nav-item.h2 {
                  padding-left: 20px;
                  font-size: 14px;
                  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
              }
              
              .nav-item.h3 {
                  padding-left: 40px;
                  font-size: 13px;
                  border-bottom: 1px solid rgba(0, 0, 0, 0.03);
              }
  
              .nav-item.active {
                  color: #000;
                  font-weight: 500;
              }
  
              .desktop-header {
                  display: none;
              }
          }
      `;
  document.head.appendChild(style);

  const nav = document.createElement("nav");
  nav.className = "nav-container collapsed";
  nav.innerHTML = `
          <div class="nav-header">
              <span class="current-section"></span>
              <a class="nav-close"><i class="fa-solid fa-xmark"></i></a>
          </div>
          <div class="desktop-header">
              <span class="current-section"></span>
              <button class="nav-collapse">
                  <a class="expand-icon"><i class="fa-solid fa-angle-down"></i></a>
                  <a class="collapse-icon"><i class="fa-solid fa-arrow-left-long"></i></a>
              </button>
          </div>
          <ul class="nav-list"></ul>
      `;
  document.body.appendChild(nav);

  // Get all h1, h2, and h3 elements throughout the document, regardless of nesting
  const h1Elements = Array.from(document.querySelectorAll("h1"));
  const allHeadings = Array.from(document.querySelectorAll("h1, h2, h3.toc"));
  const navList = nav.querySelector(".nav-list");

  function resetNavbarState() {
    nav.classList.remove("expanded");
    document.body.style.overflow = "";
  }

  // Sort headings by their position in the document
  allHeadings.sort((a, b) => {
    return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING
      ? -1
      : 1;
  });

  // Group h2 and h3 elements with their parent headings
  let currentH1 = null;
  let currentH2 = null;
  let h1Groups = [];

  allHeadings.forEach((heading) => {
    if (heading.tagName === "H1") {
      currentH1 = heading;
      currentH2 = null;
      h1Groups.push({
        h1: heading,
        h2s: [],
      });
    } else if (heading.tagName === "H2" && currentH1) {
      currentH2 = heading;
      // Find the last h1 group and add this h2 to it
      const lastGroup = h1Groups[h1Groups.length - 1];
      if (lastGroup) {
        lastGroup.h2s.push({
          h2: heading,
          h3s: [],
        });
      }
    } else if (heading.tagName === "H3" && currentH2 && currentH1) {
      // Find the last h1 group and the last h2 in that group
      const lastGroup = h1Groups[h1Groups.length - 1];
      if (lastGroup && lastGroup.h2s.length > 0) {
        const lastH2Group = lastGroup.h2s[lastGroup.h2s.length - 1];
        lastH2Group.h3s.push(heading);
      }
    }
  });

  // Create navigation items from the groups
  h1Groups.forEach((group) => {
    // Add h1 element
    const h1Item = document.createElement("li");
    h1Item.className = "nav-item h1";
    h1Item.textContent = group.h1.textContent.trim();
    h1Item.addEventListener("click", () => {
      group.h1.scrollIntoView({ behavior: "smooth" });
      if (window.innerWidth < 768) {
        resetNavbarState();
      }
    });
    navList.appendChild(h1Item);

    // Add all h2 elements for this h1
    group.h2s.forEach((h2Group) => {
      const h2Item = document.createElement("li");
      h2Item.className = "nav-item h2";
      h2Item.textContent = h2Group.h2.textContent.trim();
      h2Item.addEventListener("click", () => {
        h2Group.h2.scrollIntoView({ behavior: "smooth" });
        if (window.innerWidth < 768) {
          resetNavbarState();
        }
      });
      navList.appendChild(h2Item);

      // Add all h3 elements for this h2
      h2Group.h3s.forEach((h3) => {
        const h3Item = document.createElement("li");
        h3Item.className = "nav-item h3";
        h3Item.textContent = h3.textContent.trim();
        h3Item.addEventListener("click", () => {
          h3.scrollIntoView({ behavior: "smooth" });
          if (window.innerWidth < 768) {
            resetNavbarState();
          }
        });
        navList.appendChild(h3Item);
      });
    });
  });

  // Collapse button handling for desktop
  const collapseBtn = nav.querySelector(".nav-collapse");
  collapseBtn.addEventListener("click", () => {
    if (window.innerWidth >= 768) {
      nav.classList.toggle("collapsed");
    }
  });

  // Mobile navigation toggle
  const navHeader = nav.querySelector(".nav-header");
  const closeBtn = nav.querySelector(".nav-close");

  navHeader.addEventListener("click", (e) => {
    if (window.innerWidth < 768 && e.target !== closeBtn) {
      nav.classList.toggle("expanded");
      document.body.style.overflow = nav.classList.contains("expanded")
        ? "hidden"
        : "";
    }
  });

  // Scroll handling
  let lastScrollTop = 0;
  let ticking = false;

  function updateNavbar() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const isMobile = window.innerWidth < 768;

    // Find the active section based on scroll position
    let activeH1 = null;
    let activeH2 = null;
    let activeH3 = null;
    const scrollPosition = window.scrollY + 100; // Add offset for better detection

    // Check which heading is currently visible
    for (let i = 0; i < allHeadings.length; i++) {
      const heading = allHeadings[i];
      const headingTop = heading.getBoundingClientRect().top + window.scrollY;

      // Check if this heading is visible
      if (headingTop <= scrollPosition) {
        if (heading.tagName === "H1") {
          activeH1 = heading;
          activeH2 = null; // Reset active H2 when we find a new H1
          activeH3 = null; // Reset active H3 when we find a new H1
        } else if (heading.tagName === "H2" && activeH1) {
          activeH2 = heading;
          activeH3 = null; // Reset active H3 when we find a new H2
        } else if (heading.tagName === "H3" && activeH2 && activeH1) {
          activeH3 = heading;
        }
      } else {
        // If we've passed the visible section, break out
        break;
      }
    }

    // Update active states and section displays
    const navItems = nav.querySelectorAll(".nav-item");
    const currentSections = nav.querySelectorAll(".current-section");

    if (activeH1) {
      navItems.forEach((item) => {
        if (
          (item.classList.contains("h1") &&
            activeH1.textContent.trim() === item.textContent) ||
          (item.classList.contains("h2") &&
            activeH2 &&
            activeH2.textContent.trim() === item.textContent) ||
          (item.classList.contains("h3") &&
            activeH3 &&
            activeH3.textContent.trim() === item.textContent)
        ) {
          item.classList.add("active");
        } else {
          item.classList.remove("active");
        }
      });

      // Update all current section displays
      let displayText = activeH1.textContent.trim();

      if (activeH2) {
        displayText = `${displayText} > ${activeH2.textContent.trim()}`;

        if (activeH3) {
          displayText = `${displayText} > ${activeH3.textContent.trim()}`;
        }
      }

      currentSections.forEach((section) => {
        section.textContent = displayText;
      });

      nav.classList.add("has-active");
    } else {
      navItems.forEach((item) => item.classList.remove("active"));
      currentSections.forEach((section) => {
        section.textContent = "";
      });
      nav.classList.remove("has-active");
    }

    // Show/hide navigation based on scroll position
    const firstH1 = h1Elements[0];
    const firstH1Passed = firstH1 && window.scrollY + 10 >= firstH1.offsetTop;

    // Handle visibility
    if (isMobile) {
      if (!nav.classList.contains("expanded")) {
        if (!firstH1Passed) {
          nav.classList.remove("visible");
        } else {
          if (scrollTop > lastScrollTop) {
            if (scrollTop > 50) {
              nav.classList.remove("visible");
            }
          } else {
            nav.classList.add("visible");
          }
        }
      }
    } else {
      if (!firstH1Passed) {
        nav.classList.remove("visible");
      } else {
        nav.classList.add("visible");
      }
    }

    lastScrollTop = scrollTop;
    ticking = false;
  }

  window.addEventListener("scroll", () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        updateNavbar();
      });
      ticking = true;
    }
  });

  // Initial update
  updateNavbar();
});
