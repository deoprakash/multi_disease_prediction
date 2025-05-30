function navigateToForm() {
    const model = document.getElementById("model").value;
    if (!model) {
        alert("Please select a model");
        return;
    }

    // Redirect to respective form pages
    switch(model) {
        case "cancer":
            window.location.href = "/cancer";
            break;
        case "cardio":
            window.location.href = "/cardio";
            break;
        case "iron":
            window.location.href = "/iron";
            break;
        case "obesity":
            window.location.href = "/obesity";
            break;
        case "sicklecell":
            window.location.href = "/sicklecell";
            break;
        case "diabetes":
            window.location.href = "/diabetes";
            break;
        case "brain_tumor":
            window.location.href = "/brain_tumor";
            break;
        case "ckd":
            window.location.href = "/ckd";
            break;
        case "common_disease":
            window.location.href = "/common_disease";
            break;
        default:
            alert("Model not found");
    }
}

const chatButton = document.getElementById('chat-button');
const chatIcon = document.querySelector('.chat-icon');
const chatContainer = document.getElementById('chat-container');
const chatClose = document.getElementById('chat-close');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatMessages = document.getElementById('chat-messages');

// Open chat only when clicking the chat icon
chatIcon.addEventListener('click', (e) => {
  e.stopPropagation(); // prevent bubbling
  chatContainer.style.display = 'flex';

  // Add welcome message if the chat is empty
  if (chatMessages.children.length === 0) {
    addBotMessage("Hello! How can I help you today?");
  }

  // Focus on input field
  chatInput.focus();
});

// Close chat when clicking ❌
chatClose.addEventListener('click', (e) => {
  e.stopPropagation(); // prevent bubbling
  chatContainer.style.display = 'none';
});

// Prevent click inside chat container from triggering anything
chatContainer.addEventListener('click', (e) => {
  e.stopPropagation(); // important to avoid unwanted triggers
});

// Helper to append a message
function appendMessage(text, sender = 'user') {
  const msg = document.createElement('div');
  msg.textContent = text;
  msg.style.margin = '5px 0';
  msg.style.textAlign = sender === 'bot' ? 'left' : 'right';
  msg.style.background = sender === 'bot' ? '#f1f1f1' : 'var(--accent)';
  msg.style.color = sender === 'bot' ? '#333' : 'white';
  msg.style.padding = '8px';
  msg.style.borderRadius = '8px';
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Send button handler
chatSend.addEventListener('click', () => {
  const text = chatInput.value.trim();
  if (!text) return;
  appendMessage(text, 'user');
  chatInput.value = '';

  // Call your backend
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text })
  })
    .then(r => r.json())
    .then(data => {
      appendMessage(data.response, 'bot');
    })
    .catch(err => {
      appendMessage('Sorry, something went wrong.', 'bot');
      console.error(err);
    });
});

// Also send on Enter
chatInput.addEventListener('keypress', e => {
  if (e.key === 'Enter') chatSend.click();
});

// Highlight active navbar item
document.addEventListener('DOMContentLoaded', function () {
  const currentLocation = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');

  navLinks.forEach(link => {
    if (link.getAttribute('href') === currentLocation) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
});

// Helper to add bot welcome message
function addBotMessage(message) {
  appendMessage(message, 'bot');
}


// animation section added

document.addEventListener('DOMContentLoaded', () => {
  gsap.registerPlugin(ScrollTrigger);

  // Hero Content - Slide in from bottom
  gsap.from(".hero-content", {
    scrollTrigger: {
      trigger: ".hero-content",
      start: "top 90%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 50,
    duration: 1,
    ease: "power3.out"
  });

  // Model selector - Scale in with bounce
  gsap.from(".model-selector", {
    scrollTrigger: {
      trigger: ".model-selector",
      start: "top 90%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    scale: 0.8,
    duration: 1.1,
    ease: "back.out(1.7)"
  });

  // Feature Boxes: Alternate directions
  const featureBoxes = gsap.utils.toArray(".feature-box");
  featureBoxes.forEach((box, i) => {
    let x = i % 2 === 0 ? -100 : 100; // Alternate left/right
    gsap.from(box, {
      scrollTrigger: {
        trigger: box,
        start: "top 85%",
        toggleActions: "play play play play"
      },
      opacity: 0,
      x: x,
      duration: 1,
      ease: "power2.out"
    });
  });

  // Quote section: slide up and fade
  gsap.from(".quote-text", {
    scrollTrigger: {
      trigger: ".quote-text",
      start: "top 90%",
    },
    opacity: 0,
    y: 40,
    duration: 1,
    ease: "power2.out"
  });

  gsap.from(".quote-author", {
    scrollTrigger: {
      trigger: ".quote-author",
      start: "top 90%",
    },
    opacity: 0,
    y: 20,
    delay: 0.3,
    duration: 1,
    ease: "power2.out"
  });

  // Chat button - bounce in from right
  gsap.set(".chat-button", { x: 100, opacity: 0 }); // Set initial state

gsap.to(".chat-button", 
  {
    x: 0,
    opacity: 1,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".chat-button",
      start: "top 90%",
      toggleActions: "play none none none",
      once: true
    }
  }
);


  // Navbar slide from top
  gsap.from(".navbar", {
    y: -60,
    opacity: 0,
    duration: 1,
    ease: "power3.out"
  });
});
