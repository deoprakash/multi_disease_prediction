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
