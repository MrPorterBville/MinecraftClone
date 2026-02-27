const searchInput = document.querySelector('#doc-search');
const cards = [...document.querySelectorAll('.doc-card')];
const navLinks = [...document.querySelectorAll('.sidebar a')];

function applyFilter() {
  const query = searchInput.value.trim().toLowerCase();

  cards.forEach((card) => {
    const text = card.textContent.toLowerCase();
    const title = (card.dataset.title || '').toLowerCase();
    const match = !query || text.includes(query) || title.includes(query);
    card.classList.toggle('hidden', !match);
  });

  navLinks.forEach((link) => {
    const targetId = link.getAttribute('href')?.slice(1);
    const target = cards.find((card) => card.id === targetId);
    link.classList.toggle('hidden', Boolean(target?.classList.contains('hidden')));
  });
}

searchInput.addEventListener('input', applyFilter);
