gsap.registerPlugin(ScrollTrigger);

// Animate the About Us Section Title (Slide and Fade In)
gsap.from(".about-header", {
  scrollTrigger: {
    trigger: ".about-header",
    start: "top 85%", // Trigger when the top of the section is 85% from the top of the viewport
    toggleActions: "play none none none" // Animation will play once when in view
  },
  opacity: 0,
  y: 40, // Slide from bottom
  duration: 1.2,
  ease: "power2.out"
});

// Animate the About Us Text (Slide and Fade In)
gsap.from(".about-text", {
  scrollTrigger: {
    trigger: ".about-text",
    start: "top 85%", // Trigger when the top of the section is 85% from the top of the viewport
    toggleActions: "play none none none" // Animation will play once when in view
  },
  opacity: 0,
  y: 40, // Slide from bottom
  duration: 1.2,
  ease: "power2.out",
  stagger: 0.2 // Add staggered animation for each text element
});

// Animate Team Member Cards (Slide and Fade In)
gsap.utils.toArray(".team-card").forEach((card, i) => {
  let x = i % 2 === 0 ? -100 : 100; // Alternate left/right
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 85%", // Trigger when the top of the section is 85% from the top of the viewport
      toggleActions: "play none none none" // Animation will play once when in view
    },
    opacity: 0,
    x: x, // Slide from left/right
    duration: 1.2,
    ease: "power2.out"
  });
});

// Animate the Guide Section (Slide and Fade In)
gsap.utils.toArray(".guide-card").forEach((card, i) => {
  let x = i % 2 === 0 ? -100 : 100; // Alternate left/right
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 85%", // Trigger when the top of the section is 85% from the top of the viewport
      toggleActions: "play none none none" // Animation will play once when in view
    },
    opacity: 0,
    x: x, // Slide from left/right
    duration: 1.2,
    ease: "power2.out"
  });
});

// Animate Achievements Section (Slide and Fade In)
gsap.from(".achievements-card", {
  scrollTrigger: {
    trigger: ".achievements-card",
    start: "top 85%", // Trigger when the top of the section is 85% from the top of the viewport
    toggleActions: "play none none none" // Animation will play once when in view
  },
  opacity: 0,
  y: 40, // Slide from bottom
  duration: 1.2,
  ease: "power2.out"
});
